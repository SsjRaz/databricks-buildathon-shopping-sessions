#include "input.hpp"
#include "index.hpp"
#include "parquet.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
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

// Packed names use all 63 low bits without a tag. Linked person ordinals
// and coordinates in the raw-name store have distinct high-bit tags.
constexpr uint64_t raw_name = uint64_t{3} << 62;
struct TimedEvent {
    int64_t time;
    uint32_t price, category;
};

struct LaneExtras {
    size_t lane;
    std::vector<Text> names;
    std::vector<TimedEvent> details;

    uint64_t routed_time(int64_t time, uint32_t price, uint32_t category) {
        if (price == 0 && time >= 0) return static_cast<uint64_t>(time);
        const size_t index = details.size();
        details.push_back({time, price, category});
        return nonpacked_alias | (index * partitions + lane);
    }
    // The low partition bits identify the producing worker (threads <= partitions).
    uint64_t add_name(Text id) {
        const size_t index = names.size();
        names.push_back(id);
        return raw_name | (index * partitions + lane);
    }
};

struct PersonKey {
    uint64_t value;
    bool linked() const { return (value >> 62) == 2; }
    bool packed() const { return (value & nonpacked_alias) == 0; }
    uint64_t packed_key() const { return value; }
    size_t ordinal() const { return value & ~nonpacked_alias; }
    Text unlinked_name(std::span<const LaneExtras> store) const {
        const uint64_t location = value & ~raw_name;
        return store[shard(location)].names[location / partitions];
    }
};

struct ResolvedPerson {
    PersonKey key;
    size_t partition;
};

class LinkedIds {
    struct Node {
        Text name;
        uint32_t parent, size;
    };
    std::vector<Node> nodes;
    PackedAliasIndex packed_ids;
    StringIndex other_ids;
    std::vector<size_t> canonical;
    std::array<std::vector<Text>, partitions> names_by_partition;

    uint32_t index(Text id) {
        const uint64_t packed = pack_alias(id);
        const auto [i, inserted] = packed == nonpacked_alias
            ? other_ids.try_add(id, text_hash(id), nodes.size())
            : packed_ids.try_add(packed, nodes.size());
        if (inserted) nodes.push_back({id, i, 1});
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
        nodes[first].name = std::min(nodes[first].name, nodes[second].name);
    }

    static size_t estimate_ids(const std::vector<input::Mapping>& links) {
        size_t bytes = 0;
        for (const auto& file : links) bytes += file.bytes().size();
        // Reserve a starting capacity, with normal growth for different ID
        // lengths and link shapes. Cap the estimate for unusually long rows.
        return std::min<size_t>(bytes / 16, 4 * 1024 * 1024);
    }

public:
    explicit LinkedIds(const std::vector<input::Mapping>& links) : packed_ids(estimate_ids(links)) {
        nodes.reserve(estimate_ids(links));
        for (const auto& file : links) {
            input::lookup<2>(file, [&](const auto& row) {
                if (!row[0].empty() && !row[1].empty()) join(row[0], row[1]);
            });
        }
        canonical.resize(nodes.size());
        for (uint32_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].parent != i) continue;
            const Text name = nodes[i].name;
            const size_t part = shard(text_hash(name));
            auto& names = names_by_partition[part];
            canonical[i] = names.size() * partitions + part;
            names.push_back(name);
        }
        for (uint32_t i = 0; i < nodes.size(); ++i) canonical[i] = canonical[root(i)];
        packed_ids.resolve(canonical);
        if (other_ids.empty()) std::vector<size_t>().swap(canonical);
        std::vector<Node>().swap(nodes);
    }
    ResolvedPerson packed_person(uint64_t packed) const {
        const uint64_t hash = packed_alias_hash(packed);
        const uint64_t code = packed_ids.find(packed, hash);
        if (code != PackedAliasIndex::missing) return {{nonpacked_alias | (code / partitions)}, shard(code)};
        return {{packed}, shard(hash)};
    }
    ResolvedPerson person(Text id, LaneExtras& extras) const {
        const uint64_t packed = pack_alias(id);
        if (packed != nonpacked_alias) return packed_person(packed);
        const uint64_t hash = text_hash(id);
        const uint32_t index = other_ids.find(id, hash);
        if (index == StringIndex::missing) return {{extras.add_name(id)}, shard(hash)};
        const size_t code = canonical[index];
        return {{nonpacked_alias | (code / partitions)}, shard(code)};
    }

    std::span<const Text> names(size_t partition) const {
        return names_by_partition[partition];
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

constexpr uint64_t wide_payload = nonpacked_alias;
constexpr uint64_t invalid_user = UINT64_MAX;

// Purchases and outlying values retain their full amounts, times and user IDs.
// Stable addresses are needed because later chunks may append to the deque.
struct WidePayload {
    Text user;
    int64_t time, ingest;
    uint32_t price, category;
};

struct CandidatePayload {
    uint64_t user;
    union {
        struct { uint32_t time, ingest; } delta;
        const WidePayload* wide;
    };

    int64_t ingest_time(int64_t start) const {
        return user == wide_payload ? wide->ingest : start + delta.ingest;
    }
};

struct PackedCandidate {
    uint64_t id;
    CandidatePayload payload;
};
static_assert(sizeof(PackedCandidate) == 24);

struct Candidate {
    uint64_t id_hash;
    Text id;
    CandidatePayload payload;
};

// Tag the eight-byte domain above every nine-byte code. Byte-identical IDs
// always enter the same domain, including when CSV quoting differs.
constexpr uint64_t nonpacked_event_id = UINT64_MAX;

uint64_t pack_event_id(Text id) {
    if (id.size() == 9) {
        const uint64_t key = pack_alias(id);
        return key == nonpacked_alias ? nonpacked_event_id : key;
    }
    if (id.size() != 8) return nonpacked_event_id;

    constexpr uint64_t byte_mask = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t bytes;
    std::memcpy(&bytes, id.data(), sizeof(bytes));
    if (bytes & ~byte_mask) return nonpacked_event_id;
#if defined(__BMI2__)
    return nonpacked_alias | _pext_u64(bytes, byte_mask);
#else
    uint64_t key = nonpacked_alias;
    for (size_t i = 0; i < id.size(); ++i) {
        key |= uint64_t{static_cast<unsigned char>(id[i])} << (7 * i);
    }
    return key;
#endif
}

// Direct timestamps cover non-purchasing events after the epoch. Purchases
// and negative timestamps point to complete records in the routing lane.
struct Event {
    PersonKey person;
    uint64_t time_or_detail;

    TimedEvent decode(std::span<const LaneExtras> extras) const {
        if ((time_or_detail & nonpacked_alias) == 0)
            return {static_cast<int64_t>(time_or_detail), 0, 0};
        const uint64_t location = time_or_detail & ~nonpacked_alias;
        return extras[shard(location)].details[location / partitions];
    }
};
static_assert(sizeof(Event) == 16);

struct Click {
    PersonKey person;
    Text id;
    int64_t time;
    uint32_t campaign;
};

CandidatePayload event_payload(const std::array<Text, 7>& row, input::Window window,
                               const Lookups& lookups, std::deque<WidePayload>& wide_rows) {
    const int64_t ingest = input::iso_time(row[3]);
    CandidatePayload result{};
    result.user = invalid_user;
    int64_t time = window.start;
    Text user;
    uint32_t purchase_price = 0, category = 0;

    if (!row[1].empty() && !row[5].empty()) {
        time = input::timestamp(row[2]);
        if (time >= window.start && time < window.end &&
            (ingest <= time || static_cast<uint64_t>(ingest) - static_cast<uint64_t>(time) <= input::day_ms)) {
            if (row[4] == "view" || row[4] == "cart") {
                user = row[1];
            } else if (row[4] == "purchase") {
                const uint64_t price = input::decimal(row[6], 10000000);
                if (price != 0 && price != UINT64_MAX) {
                    user = row[1];
                    purchase_price = static_cast<uint32_t>(price);
                    category = lookups.category_for(row[5]);
                }
            }
        }
    }
    if (user.empty()) time = window.start;
    else result.user = pack_alias(user);

    const uint64_t ingest_delta = static_cast<uint64_t>(ingest) - static_cast<uint64_t>(window.start);
    const uint64_t time_delta = static_cast<uint64_t>(time) - static_cast<uint64_t>(window.start);
    if (purchase_price || result.user == wide_payload || ingest < window.start || ingest_delta > UINT32_MAX || time_delta > UINT32_MAX) {
        result.user = wide_payload;
        result.wide = &wide_rows.emplace_back(WidePayload{user, time, ingest, purchase_price, category});
    } else {
        result.delta = {static_cast<uint32_t>(time_delta), static_cast<uint32_t>(ingest_delta)};
    }
    return result;
}

// Deduplicate the exact numeric and full-byte domains independently. In
// each domain candidates in a partition share the low hash bits, so probe
// with the remaining bits. A hash match in the byte domain requires equality
// of the original ID. In the numeric domain the table stores the exact ID.
template<class Row>
void deduplicate(std::vector<Shards<Row>>& lanes, size_t part,
                 Shards<Event>& events, const LinkedIds& linked, int64_t start, LaneExtras& extras) {
    constexpr bool packed = std::is_same_v<Row, PackedCandidate>;
    size_t count = 0;
    for (const auto& lane : lanes) count += lane[part].size();
    if (count == 0) return;

    struct Entry { Row* winner = nullptr; uint64_t key = 0; };
    const size_t capacity = std::bit_ceil(count + count / 3 + 1);
    const size_t mask = capacity - 1;
    std::vector<Entry> table(capacity);
    for (auto& lane : lanes) {
        for (Row& candidate : lane[part]) {
            const uint64_t key = [](const Row& row) {
                if constexpr (packed) return row.id;
                else return row.id_hash;
            }(candidate);
            const uint64_t hash = packed ? packed_alias_hash(key) : key;
            size_t slot = (hash / partitions) & mask;
            while (true) {
                Entry& entry = table[slot];
                if (!entry.winner) {
                    entry = {&candidate, key};
                    break;
                }
                const bool matches = [&] {
                    if (entry.key != key) return false;
                    if constexpr (packed) return true;
                    else return entry.winner->id == candidate.id;
                }();
                if (matches) {
                    if (candidate.payload.ingest_time(start) < entry.winner->payload.ingest_time(start)) {
                        entry.winner->payload.user = invalid_user;
                        entry.winner = &candidate;
                    } else {
                        candidate.payload.user = invalid_user;
                    }
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    // Walk each domain's contiguous buffers instead of gathering and sorting
    // candidates or fetching the winners in hash-table order.
    for (auto& lane : lanes) {
        for (const Row& row : lane[part]) {
            const CandidatePayload& winner = row.payload;
            if (winner.user == invalid_user) continue;
            ResolvedPerson person;
            uint64_t time_code;
            if (winner.user == wide_payload) {
                if (winner.wide->user.empty()) continue;
                person = linked.person(winner.wide->user, extras);
                time_code = extras.routed_time(winner.wide->time, winner.wide->price, winner.wide->category);
            } else {
                // The packed key reconstructs the exact name on first use.
                person = linked.packed_person(winner.user);
                time_code = extras.routed_time(start + winner.delta.time, 0, 0);
            }
            events[person.partition].push_back({person.key, time_code});
        }
        std::vector<Row>().swap(lane[part]);
    }
}

void read_clicks(Text chunk, Shards<Click>& clicks, const LinkedIds& linked,
                 const Lookups& lookups, input::Window window, LaneExtras& extras) {
    input::read<4>(chunk, [&](const auto& row) {
        if (row[0].empty() || row[1].empty()) return;
        const int64_t time = input::timestamp(row[3]);
        if (time < window.start - input::week_ms || time >= window.end) return;
        const auto campaign = lookups.campaigns.find(row[2]);
        if (campaign == lookups.campaigns.end()) return;
        const ResolvedPerson person = linked.person(row[1], extras);
        clicks[person.partition].push_back({person.key, row[0], time, campaign->second});
    });
}

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

using PackedNames = std::deque<std::array<char, 9>>;

Text unpack_name(PackedNames& names, uint64_t key) {
    auto& bytes = names.emplace_back();
#if defined(__BMI2__)
    const uint64_t first = _pdep_u64(key, 0x7f7f7f7f7f7f7f7fULL);
    std::memcpy(bytes.data(), &first, sizeof(first));
    bytes[8] = static_cast<char>(key >> 56);
#else
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>((key >> (7 * i)) & 0x7f);
#endif
    return {bytes.data(), bytes.size()};
}

// Bucket into contiguous per-person ranges, then sort only the timestamps
// within each range. Clicks for people without surviving events are discarded.
GroupedPartition group_people(std::vector<Shards<Event>>& event_lanes,
                              std::vector<Shards<Click>>& click_lanes, size_t part,
                              std::span<const Text> linked_names, PackedNames& packed_names,
                              std::span<const LaneExtras> extras) {
    size_t event_count = 0, click_count = 0;
    for (const auto& lane : event_lanes) event_count += lane[part].size();
    for (const auto& lane : click_lanes) click_count += lane[part].size();
    GroupedPartition result;
    if (event_count == 0) return result;

    // The packed hash's low bits are already shared by all names in this
    // partition. Probe the packed index with the remaining bits.
    PackedAliasIndex packed_groups(event_count / 32, std::countr_zero(partitions));
    StringIndex raw_groups(event_count / 256);
    std::vector<uint32_t> linked_groups(linked_names.size(), StringIndex::missing);
    std::vector<uint32_t> event_person;
    std::vector<uint32_t> click_person;
    event_person.reserve(event_count);
    click_person.reserve(click_count);
    for (const auto& lane : event_lanes) {
        for (const Event& event : lane[part]) {
            const PersonKey person = event.person;
            uint32_t id;
            if (person.linked()) {
                auto& group = linked_groups[person.ordinal()];
                if (group == StringIndex::missing) {
                    group = result.people.size();
                    result.people.push_back({linked_names[person.ordinal()]});
                }
                id = group;
            } else if (person.packed()) {
                const auto [group, inserted] = packed_groups.try_add(person.packed_key(), result.people.size());
                if (inserted) result.people.push_back({unpack_name(packed_names, person.packed_key())});
                id = group;
            } else {
                const Text name = person.unlinked_name(extras);
                const auto [group, inserted] = raw_groups.try_add(name, text_hash(name), result.people.size());
                if (inserted) result.people.push_back({name});
                id = group;
            }
            ++result.people[id].event_end;
            event_person.push_back(id);
        }
    }
    for (const auto& lane : click_lanes) {
        for (const Click& click : lane[part]) {
            const PersonKey person = click.person;
            uint32_t id;
            if (person.linked()) {
                id = linked_groups[person.ordinal()];
            } else if (person.packed()) {
                const uint64_t key = person.packed_key();
                const uint64_t group = packed_groups.find(key, packed_alias_hash(key));
                id = group == PackedAliasIndex::missing ? StringIndex::missing
                                                       : static_cast<uint32_t>(group);
            } else {
                const Text name = person.unlinked_name(extras);
                id = raw_groups.find(name, text_hash(name));
            }
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
            result.events[offset] = event.decode(extras);
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
    timer.mark("lookups");
    std::promise<LinkedIds> build_links;
    const auto links_ready = build_links.get_future().share();

    const auto event_chunks = input::chunks(event_files, chunk_size);
    const auto click_chunks = input::chunks(click_files, chunk_size);
    size_t event_bytes = 0;
    for (Text chunk : event_chunks) event_bytes += chunk.size();
    // A capacity hint only. Vectors still grow normally for smaller rows or
    // uneven work distribution.
    const size_t candidate_hint = event_bytes / (64 * partitions * threads);
    std::vector<LaneExtras> extras(threads);
    for (size_t i = 0; i < threads; ++i) extras[i].lane = i;
    std::vector<std::deque<WidePayload>> wide_rows(threads);
    std::vector<Shards<PackedCandidate>> packed_candidate_lanes(threads);
    std::vector<Shards<Candidate>> candidate_lanes(threads);
    std::vector<Shards<Event>> event_lanes(threads);
    std::vector<Shards<Click>> click_lanes(threads);

    parallel(1 + event_chunks.size() + click_chunks.size(), threads, [&](size_t lane, size_t job) {
        if (job == 0) {
            try {
                build_links.set_value(LinkedIds(link_files));
            } catch (...) {
                build_links.set_exception(std::current_exception());
                throw;
            }
            return;
        }
        --job;
        if (job >= event_chunks.size()) {
            const auto ready = links_ready;
            read_clicks(click_chunks[job - event_chunks.size()], click_lanes[lane],
                        ready.get(), lookups, window, extras[lane]);
            return;
        }
        if (candidate_hint >= 1024 && packed_candidate_lanes[lane][0].capacity() == 0) {
            for (auto& candidates : packed_candidate_lanes[lane]) candidates.reserve(candidate_hint);
        }
        input::read<7>(event_chunks[job], [&](const auto& row) {
            const Text id = row[0];
            if (id.empty()) return;

            const uint64_t packed = pack_event_id(id);
            const auto payload = event_payload(row, window, lookups, wide_rows[lane]);
            if (packed != nonpacked_event_id) {
                const size_t part = shard(packed_alias_hash(packed));
                packed_candidate_lanes[lane][part].push_back({packed, payload});
            } else {
                const uint64_t hash = text_hash(id);
                candidate_lanes[lane][shard(hash)].push_back({hash, id, payload});
            }
        });
    });
    timer.mark("build links and parse events and clicks");
    const LinkedIds& linked = links_ready.get();

    const size_t event_hint = event_bytes / (72 * partitions * threads);
    parallel(partitions, threads, [&](size_t lane, size_t part) {
        if (event_hint >= 1024 && event_lanes[lane][0].capacity() == 0) {
            for (auto& events : event_lanes[lane]) events.reserve(event_hint);
        }
        deduplicate(packed_candidate_lanes, part, event_lanes[lane], linked, window.start, extras[lane]);
        deduplicate(candidate_lanes, part, event_lanes[lane], linked, window.start, extras[lane]);
    });
    timer.mark("event dedup and routing");

    wide_rows.clear();
    std::array<PackedNames, partitions> packed_names;
    Shards<UserDay> user_rows;
    std::array<CampaignTotals, partitions> campaign_totals;
    parallel(partitions, threads, [&](size_t, size_t part) {
        auto grouped = group_people(event_lanes, click_lanes, part, linked.names(part), packed_names[part], extras);
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
    // Write a fixed set of parts, including empty parts, so a subsequent run
    // in the same output directory replaces every prior user table part.
    constexpr size_t output_parts = 8;
    static_assert(partitions % output_parts == 0);
    constexpr size_t shards_per_part = partitions / output_parts;
    const auto user_output = output / "user_daily";
    const auto campaign_output = output / "campaign_daily";
    fs::create_directories(user_output);
    fs::create_directories(campaign_output);
    const std::span<const std::vector<UserDay>> rows(user_rows);
    parallel(output_parts, threads, [&](size_t, size_t part) {
        parquet::write_table(user_output / ("part-" + std::to_string(part) + ".parquet"),
                             std::span<const parquet::Field<UserDay>>(user_fields),
                             rows.subspan(part * shards_per_part, shards_per_part));
    });
    parquet::write_table(campaign_output / "part-0.parquet",
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
