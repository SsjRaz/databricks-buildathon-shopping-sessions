#include "input.hpp"
#include "index.hpp"
#include "parquet.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>

namespace {
using input::Text;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr size_t partitions = 128;
constexpr size_t chunk_size = 4 * 1024 * 1024;
constexpr int64_t session_gap = 1800000;

template<class Row>
using Shards = std::array<std::vector<Row>, partitions>;

size_t shard(uint64_t value) { return value & (partitions - 1); }

template<class Fn>
void parallel(size_t jobs, size_t threads, Fn&& fn) {
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex lock;
    std::vector<std::jthread> workers;
    workers.reserve(std::min(jobs, threads));
    for (size_t lane = 0; lane < std::min(jobs, threads); ++lane) {
        workers.emplace_back([&, lane] {
            try {
                for (size_t job; (job = next.fetch_add(1, std::memory_order_relaxed)) < jobs;) {
                    fn(lane, job);
                }
            } catch (...) {
                std::lock_guard guard(lock);
                if (!error) error = std::current_exception();
                next.store(jobs, std::memory_order_relaxed);
            }
        });
    }
    workers.clear();
    if (error) std::rethrow_exception(error);
}

struct Person {
    Text name;
    uint64_t hash;
};

class LinkedIds {
    struct Node {
        Person person;
        uint32_t parent, size;
    };
    std::vector<Node> nodes;
    StringIndex ids;

    uint32_t index(Text id) {
        const uint64_t hash = text_hash(id);
        const auto [i, inserted] = ids.try_add(id, hash, nodes.size());
        if (inserted) nodes.push_back({{id, hash}, i, 1});
        return i;
    }
    uint32_t root(uint32_t i) {
        while (nodes[i].parent != i) {
            i = nodes[i].parent = nodes[nodes[i].parent].parent;
        }
        return i;
    }
    void join(Text a, Text b) {
        uint32_t first = root(index(a));
        uint32_t second = root(index(b));
        if (first == second) return;
        if (nodes[first].size < nodes[second].size) std::swap(first, second);
        nodes[second].parent = first;
        nodes[first].size += nodes[second].size;
        if (nodes[second].person.name < nodes[first].person.name) {
            nodes[first].person = nodes[second].person;
        }
    }

    static size_t estimate_ids(const std::vector<input::Mapping>& links) {
        size_t bytes = 0;
        for (const auto& file : links) bytes += file.bytes().size();
        // Reserve a starting capacity, with normal growth for different ID
        // lengths and link shapes. Cap the estimate for unusually long rows.
        return std::min<size_t>(bytes / 16, 4 * 1024 * 1024);
    }

public:
    explicit LinkedIds(const std::vector<input::Mapping>& links) : ids(estimate_ids(links)) {
        nodes.reserve(estimate_ids(links));
        for (const auto& file : links) {
            input::lookup<2>(file, [&](const auto& row) {
                if (!row[0].empty() && !row[1].empty()) join(row[0], row[1]);
            });
        }
        for (uint32_t i = 0; i < nodes.size(); ++i) nodes[i].parent = root(i);
        for (Node& node : nodes) node.person = nodes[node.parent].person;
    }
    Person person(Text id) const {
        const uint64_t hash = text_hash(id);
        const uint32_t index = ids.find(id, hash);
        return index == StringIndex::missing ? Person{id, hash} : nodes[index].person;
    }
};

struct Lookups {
    std::unordered_map<Text, uint32_t> products;
    std::unordered_map<Text, uint32_t> campaigns;
    std::vector<Text> category{"unknown"};
    std::vector<Text> campaign{"organic"};
    std::vector<Text> channel{"organic"};

    Lookups(const input::Mapping& product_file, const input::Mapping& campaign_file) {
        std::unordered_map<Text, uint32_t> categories;
        input::lookup<2>(product_file, [&](const auto& row) {
            const auto [it, inserted] = categories.try_emplace(row[1], category.size());
            if (inserted) category.push_back(row[1]);
            products.emplace(row[0], it->second);
        });
        input::lookup<2>(campaign_file, [&](const auto& row) {
            campaigns.emplace(row[0], campaign.size());
            campaign.push_back(row[0]);
            channel.push_back(row[1]);
        });
    }
    uint32_t category_for(Text product) const {
        const auto it = products.find(product);
        return it == products.end() ? 0 : it->second;
    }
};

// Keep invalid candidates through deduplication. An empty user marks a record
// that must be discarded only if it is the earliest copy.
struct Candidate {
    uint64_t id_hash;
    Text id, user;
    int64_t time = 0;
    int64_t ingest;
    uint32_t price = 0;
    uint32_t category = 0;
};

struct Event {
    Text person;
    int64_t time;
    uint32_t price, category;
};

struct Click {
    Text person, id;
    int64_t time;
    uint32_t campaign;
};

Candidate event_candidate(const std::array<Text, 7>& row, input::Window window,
                          const Lookups& lookups) {
    Candidate candidate{text_hash(row[0]), row[0], {}, 0, input::iso_time(row[3])};
    if (row[1].empty() || row[5].empty()) return candidate;
    const int64_t time = input::timestamp(row[2]);
    if (time < window.start || time >= window.end ||
        (candidate.ingest > time && static_cast<uint64_t>(candidate.ingest) -
             static_cast<uint64_t>(time) > input::day_ms)) {
        return candidate;
    }
    if (row[4] == "purchase") {
        const uint64_t price = input::decimal(row[6], 10000000);
        if (price == 0 || price == UINT64_MAX) return candidate;
        candidate.price = static_cast<uint32_t>(price);
        candidate.category = lookups.category_for(row[5]);
    } else if (row[4] != "view" && row[4] != "cart") {
        return candidate;
    }
    candidate.user = row[1];
    candidate.time = time;
    return candidate;
}

// Candidates in a partition all share the low hash bits. Use the
// remaining bits for linear probing, and compare IDs on hash matches.
void deduplicate(std::vector<Shards<Candidate>>& lanes, size_t part,
                 Shards<Event>& events, const LinkedIds& linked) {
    size_t count = 0;
    for (const auto& lane : lanes) count += lane[part].size();
    if (count == 0) return;

    struct Entry { Candidate* winner = nullptr; uint64_t hash = 0; };
    const size_t capacity = std::bit_ceil(count + count / 3 + 1);
    const size_t mask = capacity - 1;
    std::vector<Entry> table(capacity);
    for (auto& lane : lanes) {
        for (Candidate& candidate : lane[part]) {
            size_t slot = (candidate.id_hash / partitions) & mask;
            while (true) {
                Entry& entry = table[slot];
                if (!entry.winner) {
                    entry = {&candidate, candidate.id_hash};
                    break;
                }
                if (entry.hash == candidate.id_hash && entry.winner->id == candidate.id) {
                    if (candidate.ingest < entry.winner->ingest) {
                        entry.winner->user = {};
                        entry.winner = &candidate;
                    } else {
                        candidate.user = {};
                    }
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    // Walk the original buffers in input order instead of gathering and
    // sorting candidates or fetching the winners in hash-table order.
    for (auto& lane : lanes) {
        for (const Candidate& winner : lane[part]) {
            if (winner.user.empty()) continue;
            const Person person = linked.person(winner.user);
            events[shard(person.hash)].push_back({person.name, winner.time, winner.price, winner.category});
        }
        std::vector<Candidate>().swap(lane[part]);
    }
}

void read_clicks(Text chunk, Shards<Click>& clicks, const LinkedIds& linked,
                 const Lookups& lookups, input::Window window) {
    input::read<4>(chunk, [&](const auto& row) {
        if (row[0].empty() || row[1].empty()) return;
        const int64_t time = input::timestamp(row[3]);
        if (time < window.start - input::week_ms || time >= window.end) return;
        const auto campaign = lookups.campaigns.find(row[2]);
        if (campaign == lookups.campaigns.end()) return;
        const Person person = linked.person(row[1]);
        clicks[shard(person.hash)].push_back({person.name, row[0], time, campaign->second});
    });
}

struct TimedEvent {
    int64_t time;
    uint32_t price, category;
};

struct TimedClick {
    Text id;
    int64_t time;
    uint32_t campaign;
};

struct PersonGroup {
    Text name;
    size_t event_end = 0, click_end = 0;
};

struct GroupedPartition {
    std::vector<PersonGroup> people;
    std::unique_ptr<TimedEvent[]> events;
    std::unique_ptr<TimedClick[]> clicks;
};

// Bucket into contiguous per-person ranges, then sort only the timestamps
// within each range. Clicks for people without surviving events are discarded.
GroupedPartition group_people(std::vector<Shards<Event>>& event_lanes,
                              std::vector<Shards<Click>>& click_lanes, size_t part) {
    size_t event_count = 0, click_count = 0;
    for (const auto& lane : event_lanes) event_count += lane[part].size();
    for (const auto& lane : click_lanes) click_count += lane[part].size();
    GroupedPartition result;
    if (event_count == 0) return result;

    StringIndex index(event_count / 16);
    std::vector<uint32_t> event_person;
    std::vector<uint32_t> click_person;
    event_person.reserve(event_count);
    click_person.reserve(click_count);
    for (const auto& lane : event_lanes) {
        for (const Event& event : lane[part]) {
            const auto [id, inserted] =
                index.try_add(event.person, text_hash(event.person), result.people.size());
            if (inserted) result.people.push_back({event.person});
            ++result.people[id].event_end;
            event_person.push_back(id);
        }
    }
    for (const auto& lane : click_lanes) {
        for (const Click& click : lane[part]) {
            const uint32_t id = index.find(click.person, text_hash(click.person));
            if (id != StringIndex::missing) ++result.people[id].click_end;
            click_person.push_back(id);
        }
    }

    size_t events = 0, clicks = 0;
    for (auto& person : result.people) {
        const size_t person_events = std::exchange(person.event_end, events);
        const size_t person_clicks = std::exchange(person.click_end, clicks);
        events += person_events;
        clicks += person_clicks;
    }
    result.events = std::make_unique_for_overwrite<TimedEvent[]>(events);
    if (clicks != 0) result.clicks = std::make_unique_for_overwrite<TimedClick[]>(clicks);
    size_t i = 0;
    for (auto& lane : event_lanes) {
        for (const Event& event : lane[part]) {
            const size_t offset = result.people[event_person[i++]].event_end++;
            result.events[offset] = {event.time, event.price, event.category};
        }
        std::vector<Event>().swap(lane[part]);
    }
    i = 0;
    for (auto& lane : click_lanes) {
        for (const Click& click : lane[part]) {
            const uint32_t id = click_person[i++];
            if (id == StringIndex::missing) continue;
            result.clicks[result.people[id].click_end++] = {click.id, click.time, click.campaign};
        }
        std::vector<Click>().swap(lane[part]);
    }
    return result;
}

struct UserDay {
    Text person, category;
    int32_t day;
    int64_t sessions = 0, events = 0, revenue = 0;
};

struct Totals {
    int64_t sessions = 0, purchasing = 0, revenue = 0, assists = 0;

    void merge(const Totals& other) {
        sessions += other.sessions;
        purchasing += other.purchasing;
        revenue += other.revenue;
        assists += other.assists;
    }
};

struct CampaignDay {
    Text id, channel;
    int32_t day;
    Totals totals;
};

using CampaignTotals = std::unordered_map<uint64_t, Totals>;

uint64_t campaign_key(int32_t day, uint32_t campaign) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(day)) << 32) | campaign;
}

int32_t utc_day(int64_t ms) {
    return static_cast<int32_t>(ms / input::day_ms - (ms % input::day_ms < 0));
}

class Attribution {
    struct Count { uint32_t count = 0, index = 0; };
    std::vector<Count> counts;
    std::vector<uint32_t> active;
    std::span<const TimedClick> clicks;
    size_t left = 0, right = 0, latest = 0;

public:
    explicit Attribution(size_t campaigns) : counts(campaigns) {}
    void next_person(std::span<const TimedClick> person_clicks) {
        for (uint32_t campaign : active) counts[campaign].count = 0;
        active.clear();
        clicks = person_clicks;
        left = right = latest = 0;
    }

    void credit(int64_t start, int32_t day, int64_t revenue, CampaignTotals& totals) {
        const int64_t oldest = start - input::week_ms;
        while (left < right && clicks[left].time < oldest) {
            auto& slot = counts[clicks[left++].campaign];
            if (--slot.count == 0) {
                counts[active.back()].index = slot.index;
                active[slot.index] = active.back();
                active.pop_back();
            }
        }
        if (left == right) {
            while (right < clicks.size() && clicks[right].time < oldest) ++right;
            left = right;
        }
        while (right < clicks.size() && clicks[right].time <= start) {
            const TimedClick& click = clicks[right];
            // Click IDs are ascending within a timestamp, so only the first
            // click at that timestamp can win credit.
            if (right == left || click.time != clicks[right - 1].time) latest = right;
            auto& slot = counts[click.campaign];
            if (slot.count++ == 0) {
                slot.index = active.size();
                active.push_back(click.campaign);
            }
            ++right;
        }
        // Exact duplicate clicks cannot change the winning ID or the set of
        // active campaigns. Their multiplicity expires at the same instant.
        const uint32_t winner = left == right ? 0 : clicks[latest].campaign;
        auto& credited = totals[campaign_key(day, winner)];
        ++credited.sessions;
        credited.purchasing += revenue != 0;
        credited.revenue += revenue;
        for (uint32_t campaign : active) ++totals[campaign_key(day, campaign)].assists;
    }
};

void summarize(GroupedPartition& partition,
               const Lookups& lookups, std::vector<UserDay>& users, CampaignTotals& campaigns) {
    Attribution attribution(lookups.campaign.size());
    std::vector<int64_t> category_revenue(lookups.category.size());
    std::vector<uint32_t> purchased_categories;
    auto finish_day = [&](UserDay daily) {
        int64_t best = 0;
        for (uint32_t category : purchased_categories) {
            const int64_t revenue = std::exchange(category_revenue[category], 0);
            const Text name = lookups.category[category];
            if (revenue > best || (revenue == best && name < daily.category)) {
                best = revenue;
                daily.category = name;
            }
        }
        purchased_categories.clear();
        users.push_back(daily);
    };

    size_t event_start = 0, click_start = 0;
    for (const PersonGroup& group : partition.people) {
        const Text person = group.name;
        std::span events(partition.events.get() + event_start, group.event_end - event_start);
        event_start = group.event_end;
        std::span<TimedClick> clicks;
        if (click_start != group.click_end) {
            clicks = {partition.clicks.get() + click_start, group.click_end - click_start};
            click_start = group.click_end;
        }
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            return a.time < b.time;
        });
        if (clicks.size() > 1) {
            std::sort(clicks.begin(), clicks.end(), [](const auto& a, const auto& b) {
                return a.time != b.time ? a.time < b.time : a.id < b.id;
            });
        }
        attribution.next_person(clicks);

        size_t cursor = 0;
        UserDay daily{person, {}, utc_day(events[cursor].time)};
        do {
            const int64_t start = events[cursor].time;
            const int32_t day = utc_day(start);
            if (day != daily.day) {
                finish_day(daily);
                daily = {person, {}, day};
            }
            int64_t previous, revenue = 0;
            const size_t first_event = cursor;
            do {
                const TimedEvent& event = events[cursor++];
                previous = event.time;
                revenue += event.price;
                if (event.price != 0) {
                    auto& sum = category_revenue[event.category];
                    if (sum == 0) purchased_categories.push_back(event.category);
                    sum += event.price;
                }
            } while (cursor < events.size() && events[cursor].time - previous <= session_gap);
            ++daily.sessions;
            daily.events += cursor - first_event;
            daily.revenue += revenue;
            attribution.credit(start, day, revenue, campaigns);
        } while (cursor < events.size());
        finish_day(daily);
    }
}

const std::array<parquet::Field<UserDay>, 6> user_fields{{
    {{"user_id", parquet::String, parquet::Utf8}, nullptr, [](const UserDay& row) { return row.person; }},
    {{"day", parquet::Int32, parquet::Date}, [](const UserDay& row) -> int64_t { return row.day; }},
    {{"sessions", parquet::Int64}, [](const UserDay& row) { return row.sessions; }},
    {{"events", parquet::Int64}, [](const UserDay& row) { return row.events; }},
    {{"revenue_cents", parquet::Int64}, [](const UserDay& row) { return row.revenue; }},
    {{"top_category", parquet::String, parquet::Utf8, true}, nullptr,
     [](const UserDay& row) { return row.category; }},
}};

const std::array<parquet::Field<CampaignDay>, 7> campaign_fields{{
    {{"campaign_id", parquet::String, parquet::Utf8}, nullptr,
     [](const CampaignDay& row) { return row.id; }},
    {{"channel", parquet::String, parquet::Utf8}, nullptr,
     [](const CampaignDay& row) { return row.channel; }},
    {{"day", parquet::Int32, parquet::Date}, [](const CampaignDay& row) -> int64_t { return row.day; }},
    {{"sessions", parquet::Int64}, [](const CampaignDay& row) { return row.totals.sessions; }},
    {{"purchasing_sessions", parquet::Int64}, [](const CampaignDay& row) { return row.totals.purchasing; }},
    {{"revenue_cents", parquet::Int64}, [](const CampaignDay& row) { return row.totals.revenue; }},
    {{"assisted_sessions", parquet::Int64}, [](const CampaignDay& row) { return row.totals.assists; }},
}};

const char* setting(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

class Timer {
    bool enabled = std::getenv("SHOPPING_TIMING") != nullptr;
    Clock::time_point previous = Clock::now();
public:
    void mark(Text stage) {
        if (!enabled) return;
        const auto now = Clock::now();
        std::cerr << stage << ' ' << std::chrono::duration<double>(now - previous).count() << " s\n";
        previous = now;
    }
};

void run(const fs::path& data, const fs::path& output, size_t threads) {
    Timer timer;
    const input::Mapping manifest(data / "manifest.json");
    const input::Mapping products(data / "products.csv");
    const input::Mapping campaigns(data / "campaigns.csv");
    const auto link_files = input::map_parts(data / "links");
    const auto event_files = input::map_parts(data / "events");
    const auto click_files = input::map_parts(data / "clicks");
    const auto window = input::parse_window(manifest.bytes());
    const Lookups lookups(products, campaigns);
    const LinkedIds linked(link_files);
    timer.mark("lookups and links");

    const auto event_chunks = input::chunks(event_files, chunk_size);
    const auto click_chunks = input::chunks(click_files, chunk_size);
    size_t event_bytes = 0;
    for (Text chunk : event_chunks) event_bytes += chunk.size();
    // A capacity hint only. Vectors still grow normally for smaller rows or
    // uneven work distribution.
    const size_t candidate_hint = event_bytes / (64 * partitions * threads);
    std::vector<Shards<Candidate>> candidate_lanes(threads);
    std::vector<Shards<Event>> event_lanes(threads);
    std::vector<Shards<Click>> click_lanes(threads);

    parallel(event_chunks.size() + click_chunks.size(), threads, [&](size_t lane, size_t job) {
        if (job >= event_chunks.size()) {
            read_clicks(click_chunks[job - event_chunks.size()], click_lanes[lane], linked, lookups, window);
            return;
        }
        if (candidate_hint >= 1024 && candidate_lanes[lane][0].capacity() == 0) {
            for (auto& candidates : candidate_lanes[lane]) candidates.reserve(candidate_hint);
        }
        input::read<7>(event_chunks[job], [&](const auto& row) {
            if (row[0].empty()) return;
            auto candidate = event_candidate(row, window, lookups);
            candidate_lanes[lane][shard(candidate.id_hash)].push_back(candidate);
        });
    });
    timer.mark("parse events and clicks");

    parallel(partitions, threads, [&](size_t lane, size_t part) {
        deduplicate(candidate_lanes, part, event_lanes[lane], linked);
    });
    timer.mark("event dedup and routing");

    Shards<UserDay> user_rows;
    std::array<CampaignTotals, partitions> campaign_totals;
    parallel(partitions, threads, [&](size_t, size_t part) {
        auto grouped = group_people(event_lanes, click_lanes, part);
        summarize(grouped, lookups, user_rows[part], campaign_totals[part]);
    });
    timer.mark("sort, sessionize and attribute");

    CampaignTotals totals;
    for (const auto& part : campaign_totals) {
        for (const auto& [key, value] : part) totals[key].merge(value);
    }
    std::array<std::vector<CampaignDay>, 1> campaign_rows;
    campaign_rows[0].reserve(totals.size());
    for (const auto& [key, value] : totals) {
        const auto campaign = static_cast<uint32_t>(key);
        const auto day = static_cast<int32_t>(key >> 32);
        campaign_rows[0].push_back({lookups.campaign[campaign], lookups.channel[campaign], day, value});
    }
    parquet::write_table(output / "user_daily", std::span<const parquet::Field<UserDay>>(user_fields),
                         user_rows);
    parquet::write_table(output / "campaign_daily",
                         std::span<const parquet::Field<CampaignDay>>(campaign_fields), campaign_rows);
    timer.mark("merge and write Parquet");
}
} // namespace

int main() {
    try {
        const auto default_threads = std::to_string(std::max(1u, std::thread::hardware_concurrency()));
        const auto threads = std::stoul(setting("NCPU", default_threads.c_str()));
        run(setting("DATA_DIR", "/data"), setting("OUTPUT_DIR", "/output"),
            std::clamp<size_t>(threads, 1, partitions));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shopping-sessions: " << error.what() << '\n';
        return 1;
    }
}
