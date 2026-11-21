#include "input.hpp"
#include "parquet.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>

namespace {
using input::Text;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr size_t partitions = 64;
constexpr size_t chunk_size = 4 * 1024 * 1024;
constexpr int64_t session_gap = 1800000;

template<class Row>
using Shards = std::array<std::vector<Row>, partitions>;

uint64_t hash(Text value) { return std::hash<Text>{}(value); }
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

template<class Row>
std::vector<Row> gather(std::vector<Shards<Row>>& lanes, size_t partition) {
    size_t count = 0;
    std::vector<Row>* largest = &lanes.front()[partition];
    for (auto& lane : lanes) {
        auto& rows = lane[partition];
        count += rows.size();
        if (rows.size() > largest->size()) largest = &rows;
    }
    std::vector<Row> combined;
    if (largest->size() == count) {
        combined.swap(*largest);
    } else {
        combined.reserve(count);
        for (auto& lane : lanes) {
            auto& rows = lane[partition];
            combined.insert(combined.end(), rows.begin(), rows.end());
            std::vector<Row>().swap(rows);
        }
    }
    return combined;
}

class LinkedIds {
    struct Node {
        size_t parent, size;
        Text name;
    };
    std::vector<Node> nodes;
    std::unordered_map<Text, size_t> ids;

    size_t index(Text id) {
        const auto [it, inserted] = ids.try_emplace(id, nodes.size());
        if (inserted) nodes.push_back({it->second, 1, id});
        return it->second;
    }
    size_t root(size_t i) {
        while (nodes[i].parent != i) {
            i = nodes[i].parent = nodes[nodes[i].parent].parent;
        }
        return i;
    }
    void join(Text a, Text b) {
        size_t first = root(index(a));
        size_t second = root(index(b));
        if (first == second) return;
        if (nodes[first].size < nodes[second].size) std::swap(first, second);
        nodes[second].parent = first;
        nodes[first].size += nodes[second].size;
        nodes[first].name = std::min(nodes[first].name, nodes[second].name);
    }

public:
    explicit LinkedIds(const std::vector<input::Mapping>& links) {
        for (const auto& file : links) {
            input::lookup<2>(file, [&](const auto& row) {
                if (!row[0].empty() && !row[1].empty()) join(row[0], row[1]);
            });
        }
        for (auto& [id, i] : ids) i = root(i);
    }
    Text person(Text id) const {
        const auto it = ids.find(id);
        return it == ids.end() ? id : nodes[it->second].name;
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
    Candidate candidate{hash(row[0]), row[0], {}, 0, input::iso_time(row[3])};
    if (row[1].empty() || row[5].empty()) return candidate;
    const int64_t time = input::timestamp(row[2]);
    if (time < window.start || time >= window.end || candidate.ingest - time > input::day_ms) {
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

bool candidate_less(const Candidate& a, const Candidate& b) {
    if (a.id_hash != b.id_hash) return a.id_hash < b.id_hash;
    const int id = a.id.compare(b.id);
    return id == 0 ? a.ingest < b.ingest : id < 0;
}

bool event_less(const Event& a, const Event& b) {
    const int person = a.person.compare(b.person);
    return person == 0 ? a.time < b.time : person < 0;
}

bool click_less(const Click& a, const Click& b) {
    const int person = a.person.compare(b.person);
    if (person != 0) return person < 0;
    if (a.time != b.time) return a.time < b.time;
    return a.id < b.id;
}

void read_clicks(Text chunk, Shards<Click>& clicks, const LinkedIds& linked,
                 const Lookups& lookups, input::Window window) {
    input::read<4>(chunk, [&](const auto& row) {
        if (row[0].empty() || row[1].empty()) return;
        const int64_t time = input::timestamp(row[3]);
        if (time < window.start - input::week_ms || time >= window.end) return;
        const auto campaign = lookups.campaigns.find(row[2]);
        if (campaign == lookups.campaigns.end()) return;
        const Text person = linked.person(row[1]);
        clicks[shard(hash(person))].push_back({person, row[0], time, campaign->second});
    });
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
    std::span<const Click> clicks;
    size_t left = 0, right = 0, latest = 0;

public:
    explicit Attribution(size_t campaigns) : counts(campaigns) {}
    void next_person(std::span<const Click> person_clicks) {
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
            const Click& click = clicks[right];
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

void summarize(std::span<const Event> events, std::span<const Click> clicks,
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

    size_t cursor = 0;
    size_t click_cursor = 0;
    while (cursor < events.size()) {
        const Text person = events[cursor].person;
        while (click_cursor < clicks.size() && clicks[click_cursor].person < person) ++click_cursor;
        const size_t first = click_cursor;
        while (click_cursor < clicks.size() && clicks[click_cursor].person == person) ++click_cursor;
        attribution.next_person(clicks.subspan(first, click_cursor - first));

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
                const Event& event = events[cursor++];
                previous = event.time;
                revenue += event.price;
                if (event.price != 0) {
                    auto& sum = category_revenue[event.category];
                    if (sum == 0) purchased_categories.push_back(event.category);
                    sum += event.price;
                }
            } while (cursor < events.size() && events[cursor].time - previous <= session_gap &&
                     events[cursor].person == person);
            ++daily.sessions;
            daily.events += cursor - first_event;
            daily.revenue += revenue;
            attribution.credit(start, day, revenue, campaigns);
        } while (cursor < events.size() && events[cursor].person == person);
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
    std::vector<Shards<Candidate>> candidate_lanes(threads);
    std::vector<Shards<Event>> event_lanes(threads);
    std::vector<Shards<Click>> click_lanes(threads);

    parallel(event_chunks.size() + click_chunks.size(), threads, [&](size_t lane, size_t job) {
        if (job >= event_chunks.size()) {
            read_clicks(click_chunks[job - event_chunks.size()], click_lanes[lane], linked, lookups, window);
            return;
        }
        input::read<7>(event_chunks[job], [&](const auto& row) {
            if (row[0].empty()) return;
            auto candidate = event_candidate(row, window, lookups);
            candidate_lanes[lane][shard(candidate.id_hash)].push_back(candidate);
        });
    });
    timer.mark("parse events and clicks");

    parallel(partitions, threads, [&](size_t lane, size_t part) {
        auto candidates = gather(candidate_lanes, part);
        std::sort(candidates.begin(), candidates.end(), candidate_less);
        for (size_t i = 0; i < candidates.size();) {
            const Candidate& winner = candidates[i++];
            if (!winner.user.empty()) {
                const Text person = linked.person(winner.user);
                event_lanes[lane][shard(hash(person))].push_back(
                    {person, winner.time, winner.price, winner.category});
            }
            while (i < candidates.size() && winner.id_hash == candidates[i].id_hash &&
                   winner.id == candidates[i].id) ++i;
        }
    });
    timer.mark("event dedup and routing");

    Shards<UserDay> user_rows;
    std::array<CampaignTotals, partitions> campaign_totals;
    parallel(partitions, threads, [&](size_t, size_t part) {
        auto events = gather(event_lanes, part);
        if (events.empty()) return;
        auto clicks = gather(click_lanes, part);
        std::sort(events.begin(), events.end(), event_less);
        std::sort(clicks.begin(), clicks.end(), click_less);
        summarize(events, clicks, lookups, user_rows[part], campaign_totals[part]);
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
