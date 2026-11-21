#include "input.hpp"
#include "index.hpp"
#include "parquet.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
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

// Nine bytes occupy the low 63 bits. Eight-byte codes set the high bit
// and use only the low 56 bits. Use a separate miss code, since eight NUL
// bytes are represented by the high bit alone.
constexpr uint64_t packed_eight_tag = uint64_t{1} << 63;
constexpr uint64_t nonpacked_id = UINT64_MAX;
constexpr uint64_t name_kind_mask = uint64_t{7} << 61;
constexpr uint64_t linked_name_tag = uint64_t{5} << 61;
constexpr uint64_t raw_name_tag = uint64_t{6} << 61;
constexpr uint64_t name_handle_mask = ~name_kind_mask;

uint64_t pack_short_id(Text id) {
    if (id.size() == 9) {
        const uint64_t key = pack_alias(id);
        return key == nonpacked_alias ? nonpacked_id : key;
    }
    if (id.size() != 8) return nonpacked_id;

    constexpr uint64_t byte_mask = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t bytes;
    std::memcpy(&bytes, id.data(), sizeof(bytes));
    if (bytes & ~byte_mask) return nonpacked_id;
#if defined(__BMI2__)
    return packed_eight_tag | _pext_u64(bytes, byte_mask);
#else
    uint64_t key = packed_eight_tag;
    for (size_t i = 0; i < id.size(); ++i) {
        key |= uint64_t{static_cast<unsigned char>(id[i])} << (7 * i);
    }
    return key;
#endif
}

using RawNameLanes = std::vector<std::vector<Text>>;
using DecodedNames = std::deque<std::array<char, 9>>;

// Both packed name ranges sort below the linked and raw handle ranges.
class PersonKey {
    uint64_t value;

    explicit PersonKey(uint64_t value) : value(value) {}

public:
    static PersonKey from_link(size_t ordinal) {
        // Linked graph nodes are indexed by uint32_t, so every ordinal fits.
        return PersonKey(linked_name_tag | ordinal);
    }

    static PersonKey from_packed(uint64_t name) { return PersonKey(name); }

    static PersonKey from_raw(Text name, std::vector<Text>& names, size_t lane) {
        if (names.size() > name_handle_mask / partitions) {
            throw std::overflow_error("too many raw person names");
        }
        const uint64_t handle = names.size() * partitions + lane;
        names.push_back(name);
        return PersonKey(raw_name_tag | handle);
    }

    bool linked() const { return (value & name_kind_mask) == linked_name_tag; }
    bool packed() const { return value < linked_name_tag; }
    uint64_t packed_key() const { return value; }
    size_t ordinal() const { return value & name_handle_mask; }

    Text raw_name(const RawNameLanes& names) const {
        const uint64_t handle = value & name_handle_mask;
        return names[shard(handle)][handle / partitions];
    }
};
static_assert(sizeof(PersonKey) == sizeof(uint64_t));

Text decode_name(DecodedNames& names, uint64_t key) {
    // User rows keep a view into this arena through Parquet writing.
    auto& bytes = names.emplace_back();
    const size_t length = (key & packed_eight_tag) != 0 ? 8 : 9;
#if defined(__BMI2__)
    constexpr uint64_t byte_mask = 0x7f7f7f7f7f7f7f7fULL;
    const uint64_t first_eight = _pdep_u64(key, byte_mask);
    std::memcpy(bytes.data(), &first_eight, sizeof(first_eight));
    if (length == 9) bytes[8] = static_cast<char>(key >> 56);
#else
    for (size_t i = 0; i < length; ++i) {
        bytes[i] = static_cast<char>((key >> (7 * i)) & 0x7f);
    }
#endif
    return Text(bytes.data(), length);
}

struct ResolvedPerson {
    PersonKey key;
    size_t partition;
};

ResolvedPerson routed_alias(uint64_t packed) {
    return {PersonKey::from_packed(packed), shard(packed_alias_hash(packed))};
}

ResolvedPerson routed_alias(Text name, uint64_t packed, std::vector<Text>& raw_names, size_t lane) {
    if (packed != nonpacked_id) return routed_alias(packed);
    return {PersonKey::from_raw(name, raw_names, lane), shard(text_hash(name))};
}

class LinkedIds {
    struct Node {
        uint32_t parent, size;
    };
    static_assert(sizeof(Node) == 8);
    using Edge = std::array<uint32_t, 2>;

    struct PackedEndpoint {
        const char* name;
        uint64_t key;
        size_t destination;

        Text text() const { return {name, (key & packed_eight_tag) != 0 ? size_t{8} : size_t{9}}; }
    };
    static_assert(sizeof(PackedEndpoint) == 24);

    std::vector<Node> nodes;
    std::vector<Text> alias_names;
    std::vector<PackedAliasIndex> packed_ids;
    StringIndex other_ids;
    std::vector<size_t> canonical;
    std::array<std::vector<Text>, partitions> names_by_partition;

    uint32_t next_index() const {
        if (nodes.size() >= StringIndex::missing) throw std::overflow_error("too many linked aliases");
        return static_cast<uint32_t>(nodes.size());
    }

    uint32_t register_alias(Text id, std::pair<uint32_t, bool> result) {
        const auto [i, inserted] = result;
        if (inserted) {
            nodes.push_back({i, 1});
            alias_names.push_back(id);
        }
        return i;
    }
    uint32_t root(uint32_t i) {
        while (nodes[i].parent != i) {
            i = nodes[i].parent = nodes[nodes[i].parent].parent;
        }
        return i;
    }
    void join(const Edge& edge) {
        uint32_t first = root(edge[0]);
        uint32_t second = root(edge[1]);
        if (first == second) return;
        if (nodes[first].size < nodes[second].size) std::swap(first, second);
        nodes[second].parent = first;
        nodes[first].size += nodes[second].size;
    }

    void build_forest(const std::vector<input::Mapping>& links, size_t bytes, size_t expected_aliases) {
        constexpr size_t edge_reserve_limit = 4 * 1024 * 1024;
        const size_t endpoint_hint = std::min(bytes / 8, 2 * edge_reserve_limit);
        const size_t per_shard = (endpoint_hint + partitions - 1) / partitions;
        std::array<std::vector<PackedEndpoint>, partitions> endpoints;
        std::vector<Edge> edges;
        edges.reserve(std::min(bytes / 18, edge_reserve_limit));
        for (const auto& file : links) {
            input::lookup<2>(file, [&](const auto& row) {
                if (row[0].empty() || row[1].empty()) return;
                const size_t ordinal = edges.size();
                auto& edge = edges.emplace_back();
                for (size_t side = 0; side < row.size(); ++side) {
                    const Text name = row[side];
                    const uint64_t packed = pack_short_id(name);
                    if (packed == nonpacked_id) {
                        edge[side] = register_alias(name, other_ids.try_add(name, text_hash(name), next_index()));
                    } else {
                        auto& pending = endpoints[shard(packed_alias_hash(packed))];
                        if (pending.capacity() == 0) pending.reserve(per_shard);
                        // One size_t token includes the side. The Edge array's
                        // element width already bounds ordinal below SIZE_MAX/2.
                        pending.push_back({name.data(), packed, 2 * ordinal + side});
                    }
                }
            });
        }
        const size_t alias_hint = (expected_aliases + partitions - 1) / partitions;
        packed_ids.reserve(partitions);
        for (auto& pending : endpoints) {
            auto& ids = packed_ids.emplace_back(std::min(alias_hint, pending.size()), std::countr_zero(partitions));
            for (const auto& endpoint : pending) {
                const auto node = register_alias(endpoint.text(), ids.try_add(endpoint.key, next_index()));
                edges[endpoint.destination / 2][endpoint.destination % 2] = node;
            }
            std::vector<PackedEndpoint>().swap(pending);
        }
        for (const auto& edge : edges) join(edge);
    }

public:
    explicit LinkedIds(const std::vector<input::Mapping>& links) {
        size_t bytes = 0;
        for (const auto& file : links) bytes += file.bytes().size();
        const size_t expected = std::min<size_t>(bytes / 16, 4 * 1024 * 1024);
        nodes.reserve(expected);
        alias_names.reserve(expected);
        build_forest(links, bytes, expected);
        // The forest is now fixed. Compress to final roots while reducing
        // names into root slots; a non-root slot still holds its original ID.
        for (uint32_t i = 0; i < nodes.size(); ++i) {
            auto& node = nodes[i];
            if (node.parent == i) continue;
            node.parent = root(node.parent);
            auto& name = alias_names[node.parent];
            if (alias_names[i] < name) name = alias_names[i];
        }
        canonical.resize(nodes.size());
        for (uint32_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].parent != i) continue;
            const Text name = alias_names[i];
            const size_t part = shard(text_hash(name));
            auto& names = names_by_partition[part];
            canonical[i] = names.size() * partitions + part;
            names.push_back(name);
        }
        for (uint32_t i = 0; i < nodes.size(); ++i) canonical[i] = canonical[nodes[i].parent];
        for (auto& ids : packed_ids) ids.resolve(canonical);
        if (other_ids.empty()) std::vector<size_t>().swap(canonical);
        std::vector<Node>().swap(nodes);
        std::vector<Text>().swap(alias_names);
    }
    ResolvedPerson person(PersonKey alias, size_t alias_partition, const RawNameLanes& raw_names) const {
        uint64_t code;
        if (alias.packed()) {
            const uint64_t key = alias.packed_key();
            code = packed_ids[alias_partition].find(key, packed_alias_hash(key));
            if (code == PackedAliasIndex::missing) return {alias, alias_partition};
        } else {
            const Text name = alias.raw_name(raw_names);
            const uint32_t index = other_ids.find(name, text_hash(name));
            if (index == StringIndex::missing) return {alias, alias_partition};
            code = canonical[index];
        }
        return {PersonKey::from_link(code / partitions), shard(code)};
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

// Pack short user IDs while the CSV bytes are hot. Keep the original
// pointer for unlinked and nonpacked IDs, and keep full lengths for fallbacks.
struct ParsedAlias {
    const char* bytes = nullptr;
    uint64_t key_or_length = nonpacked_alias;

    ParsedAlias() = default;
    explicit ParsedAlias(Text id) : bytes(id.data()), key_or_length(pack_alias(id)) {
        if (key_or_length == nonpacked_alias) key_or_length |= id.size();
    }

    bool empty() const { return bytes == nullptr; }
    uint64_t packed() const { return key_or_length & nonpacked_alias ? nonpacked_alias : key_or_length; }
    Text text() const {
        return {bytes, key_or_length & nonpacked_alias ? key_or_length & ~nonpacked_alias : 9};
    }
};
static_assert(sizeof(ParsedAlias) == sizeof(Text));

uint64_t pack_parsed_alias(const ParsedAlias& alias) {
    const uint64_t key = alias.packed();
    if (key != nonpacked_alias) return key;
    const Text name = alias.text();
    return name.size() == 8 ? pack_short_id(name) : nonpacked_id;
}

// Keep invalid candidates through deduplication. An empty user marks a record
// that must be discarded only if it is the earliest copy.
struct CandidatePayload {
    ParsedAlias user;
    int64_t time = 0;
    int64_t ingest;
    uint32_t price = 0;
    uint32_t category = 0;
};

struct TimeOrigins {
    int64_t event_ms;
    int64_t ingest_seconds;
};

struct WideCandidateTimes {
    int64_t event_ms;
    int64_t ingest_ms;
};
static_assert(alignof(WideCandidateTimes) >= 2);

// Store unsigned relative event milliseconds below the biased signed ingest
// seconds. The top bit distinguishes an exact full-width fallback.
struct PackedCandidate {
    static constexpr uint64_t wide = uint64_t{1} << 63;
    static constexpr uint64_t invalid_alias = UINT64_MAX;
    static constexpr int64_t ingest_bias = int64_t{1} << 30;

    uint64_t id;
    uint64_t alias;
    uint64_t times;

    const WideCandidateTimes& wide_times() const {
        return *reinterpret_cast<const WideCandidateTimes*>(
            static_cast<uintptr_t>((times & ~wide) << 1));
    }

    int64_t ingest_time(TimeOrigins origin) const {
        if (times & wide) return wide_times().ingest_ms;
        const int64_t delta = static_cast<int64_t>(times >> 32) - ingest_bias;
        return (origin.ingest_seconds + delta) * 1000;
    }

    bool earlier_than(const PackedCandidate& other, TimeOrigins origin) const {
        if (((times | other.times) & wide) == 0) return (times >> 32) < (other.times >> 32);
        return ingest_time(origin) < other.ingest_time(origin);
    }

    int64_t event_time(TimeOrigins origin) const {
        return times & wide ? wide_times().event_ms : origin.event_ms + static_cast<uint32_t>(times);
    }
};
static_assert(sizeof(PackedCandidate) == 24);

uint64_t pack_candidate_times(const CandidatePayload& payload, TimeOrigins origin,
                              std::deque<WideCandidateTimes>& overflow) {
    // Invalid events still enter dedup with their original ingest. Their
    // event time is never consumed because their user is empty.
    const uint64_t event_delta = payload.user.empty() ? 0 :
        static_cast<uint64_t>(payload.time) - static_cast<uint64_t>(origin.event_ms);
    const int64_t ingest_delta = payload.ingest / 1000 - origin.ingest_seconds;

    if (event_delta <= UINT32_MAX && payload.ingest % 1000 == 0 &&
        ingest_delta >= -PackedCandidate::ingest_bias && ingest_delta < PackedCandidate::ingest_bias) {
        const auto biased = static_cast<uint64_t>(ingest_delta + PackedCandidate::ingest_bias);
        return (biased << 32) | event_delta;
    }

    // A deque preserves the addresses of existing elements during appends.
    // Alignment leaves the pointer's low bit zero. Shift it away to make
    // room for the wide marker without restricting the address range.
    overflow.push_back({payload.time, payload.ingest});
    const auto pointer = reinterpret_cast<uintptr_t>(&overflow.back());
    return PackedCandidate::wide | (static_cast<uint64_t>(pointer) >> 1);
}

struct PackedPurchase {
    size_t position;
    uint32_t price, category;
};

// Raw alias handles are local to the parsing lane. Dedup compares only
// event IDs and times. Routing later reads the originating lane's Text view.
uint64_t pack_candidate_alias(const ParsedAlias& alias, std::vector<Text>& raw_names) {
    if (alias.empty()) return PackedCandidate::invalid_alias;
    const uint64_t key = pack_parsed_alias(alias);
    if (key != nonpacked_id) return key;
    if (raw_names.size() > name_handle_mask) {
        throw std::overflow_error("too many raw candidate aliases");
    }
    const uint64_t handle = raw_names.size();
    raw_names.push_back(alias.text());
    return raw_name_tag | handle;
}

struct Candidate {
    uint64_t id_hash;
    Text id;
    CandidatePayload payload;
};

struct TimedEvent {
    int64_t timestamp;
    uint32_t cents, category_id;

    static TimedEvent make(int64_t time, uint32_t price, uint32_t category, int64_t) {
        return {time, price, category};
    }

    int64_t clock() const { return timestamp; }
    int64_t absolute_time(int64_t) const { return timestamp; }
    int64_t sort_key() const { return timestamp; }
    uint32_t price() const { return cents; }
    uint32_t category() const { return category_id; }
};

// One timeline uses one layout. Compact windows guarantee that every
// surviving event's offset and category fit, without testing each row.
// The word carries 32 timestamp bits, 8 category bits and 24 price bits.
struct CompactTimedEvent {
    uint64_t word;

    static bool fits(input::Window window, size_t categories) {
        const uint64_t span = static_cast<uint64_t>(window.end) - static_cast<uint64_t>(window.start);
        return span <= (uint64_t{1} << 32) && categories <= 256;
    }

    static CompactTimedEvent make(int64_t time, uint32_t price, uint32_t category, int64_t origin) {
        // The rule's maximum price is 10,000,000, less than 2^24.
        const uint64_t offset = static_cast<uint64_t>(time) - static_cast<uint64_t>(origin);
        return {(offset << 32) | (uint64_t{category} << 24) | price};
    }

    int64_t clock() const { return static_cast<int64_t>(word >> 32); }
    int64_t absolute_time(int64_t origin) const { return origin + clock(); }
    uint64_t sort_key() const { return word; } // Payload bits break irrelevant timestamp ties.
    uint32_t price() const { return static_cast<uint32_t>(word) & 0xffffff; }
    uint32_t category() const { return static_cast<uint32_t>(word >> 24) & 0xff; }
};

template<class TimelineEvent>
struct Event {
    PersonKey alias;
    TimelineEvent timed;
};
static_assert(sizeof(TimedEvent) == 16);
static_assert(sizeof(CompactTimedEvent) == 8);
static_assert(sizeof(Event<TimedEvent>) == 24);
static_assert(sizeof(Event<CompactTimedEvent>) == 16);

struct TimedClick {
    static constexpr uint32_t wide_id = UINT32_MAX;

    union {
        const char* bytes;
        const Text* full;
    } id;
    int64_t time;
    uint32_t campaign;
    uint32_t length;

    Text id_text() const {
        return length == wide_id ? *id.full : Text(id.bytes, length);
    }
};

struct Click {
    PersonKey alias;
    TimedClick row;
};
static_assert(sizeof(TimedClick) == 24);
static_assert(sizeof(Click) == 32);

TimedClick make_timed_click(Text id, int64_t time, uint32_t campaign, std::deque<Text>& wide_ids) {
    if (id.size() < TimedClick::wide_id) {
        return {{.bytes = id.data()}, time, campaign, static_cast<uint32_t>(id.size())};
    }
    // The source mapping owns the bytes. The deque keeps the full-width
    // Text view stable while its parsing lane appends more IDs.
    wide_ids.push_back(id);
    return {{.full = &wide_ids.back()}, time, campaign, TimedClick::wide_id};
}

CandidatePayload event_payload(const std::array<Text, 7>& row, input::Window window,
                               const Lookups& lookups) {
    CandidatePayload candidate{{}, 0, input::iso_time(row[3])};
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
    candidate.user = ParsedAlias(row[1]);
    candidate.time = time;
    return candidate;
}

// Deduplicate the exact numeric and full-byte domains independently. In
// each domain candidates in a partition share the low hash bits, so probe
// with the remaining bits. A hash match in the byte domain requires equality
// of the original ID. In the numeric domain the table stores the exact ID.
template<class Row, class TimelineEvent>
void deduplicate(std::vector<Shards<Row>>& lanes, size_t part, Shards<Event<TimelineEvent>>& events,
                 std::vector<Text>& raw_names, size_t worker, TimeOrigins origin,
                 const RawNameLanes& raw_candidate_names,
                 std::span<Shards<PackedPurchase>> purchases = {}) {
    constexpr bool packed = std::is_same_v<Row, PackedCandidate>;
    const auto earlier = [&](const Row& candidate, const Row& winner) {
        if constexpr (packed) return candidate.earlier_than(winner, origin);
        else return candidate.payload.ingest < winner.payload.ingest;
    };
    const auto discard = [](Row& row) {
        if constexpr (packed) row.alias = PackedCandidate::invalid_alias;
        else row.payload.user = {};
    };
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
                    if (earlier(candidate, *entry.winner)) {
                        discard(*entry.winner);
                        entry.winner = &candidate;
                    } else {
                        discard(candidate);
                    }
                    break;
                }
                slot = (slot + 1) & mask;
            }
        }
    }

    // Walk each domain's contiguous buffers instead of gathering and sorting
    // candidates or fetching the winners in hash-table order.
    for (size_t lane = 0; lane < lanes.size(); ++lane) {
        auto& rows = lanes[lane][part];
        if constexpr (packed) {
            const auto& candidate_names = raw_candidate_names[lane];
            const auto route = [&](const PackedCandidate& row, uint32_t price, uint32_t category) {
                const uint64_t alias = row.alias;
                if (alias == PackedCandidate::invalid_alias) return;
                const auto destination = (alias & name_kind_mask) == raw_name_tag
                    ? routed_alias(candidate_names[alias & name_handle_mask], nonpacked_id, raw_names, worker)
                    : routed_alias(alias);
                events[destination.partition].push_back({destination.key,
                    TimelineEvent::make(row.event_time(origin), price, category, origin.event_ms)});
            };

            // A losing candidate still consumes the extras at its position.
            auto& charges = purchases[lane][part];
            size_t purchase = 0;
            size_t next = charges.empty() ? rows.size() : charges.front().position;
            for (size_t index = 0; index < rows.size(); ++index) {
                if (index == next) {
                    const auto& charge = charges[purchase++];
                    next = purchase == charges.size() ? rows.size() : charges[purchase].position;
                    route(rows[index], charge.price, charge.category);
                } else {
                    route(rows[index], 0, 0);
                }
            }
            std::vector<PackedPurchase>().swap(charges);
        } else {
            for (const Row& row : rows) {
                const auto& winner = row.payload;
                if (winner.user.empty()) continue;
                const auto destination =
                    routed_alias(winner.user.text(), pack_parsed_alias(winner.user), raw_names, worker);
                events[destination.partition].push_back({destination.key,
                    TimelineEvent::make(winner.time, winner.price, winner.category, origin.event_ms)});
            }
        }
        std::vector<Row>().swap(rows);
    }
}

void read_clicks(Text chunk, Shards<Click>& clicks, const Lookups& lookups, input::Window window,
                 std::vector<Text>& raw_names, size_t worker, std::deque<Text>& wide_ids) {
    input::read<4>(chunk, [&](const auto& row) {
        if (row[0].empty() || row[1].empty()) return;
        const int64_t time = input::timestamp(row[3]);
        if (time < window.start - input::week_ms || time >= window.end) return;
        const auto campaign = lookups.campaigns.find(row[2]);
        if (campaign == lookups.campaigns.end()) return;
        const auto destination = routed_alias(row[1], pack_short_id(row[1]), raw_names, worker);
        const TimedClick click = make_timed_click(row[0], time, campaign->second, wide_ids);
        clicks[destination.partition].push_back({destination.key, click});
    });
}


struct AliasGroup {
    PersonKey alias;
    size_t event_end = 0, click_end = 0;
};

template<class TimelineEvent>
struct AliasStorage {
    std::unique_ptr<TimelineEvent[]> events;
    std::unique_ptr<TimedClick[]> clicks;
};

template<class TimelineEvent>
struct PersonFragment {
    PersonKey person;
    std::span<TimelineEvent> events;
    std::span<TimedClick> clicks;
    PersonFragment* next = nullptr;
};

template<class TimelineEvent>
using FragmentShards = std::array<std::vector<PersonFragment<TimelineEvent>>, partitions>;

// Gather each raw alias exactly once. Click-only aliases are retained
// because another alias of their canonical person can have events.
template<class TimelineEvent>
AliasStorage<TimelineEvent> group_aliases(std::vector<Shards<Event<TimelineEvent>>>& event_lanes,
                                         std::vector<Shards<Click>>& click_lanes, size_t part,
                                         const RawNameLanes& raw_names, const LinkedIds& linked,
                                         FragmentShards<TimelineEvent>& fragments) {
    size_t event_count = 0, click_count = 0;
    for (const auto& lane : event_lanes) event_count += lane[part].size();
    for (const auto& lane : click_lanes) click_count += lane[part].size();
    AliasStorage<TimelineEvent> storage;
    if (event_count == 0 && click_count == 0) return storage;

    // All packed aliases here share the low hash bits used by routing.
    const size_t row_count = event_count + click_count;
    PackedAliasIndex packed_groups(row_count / 32, std::countr_zero(partitions));
    StringIndex raw_groups(row_count / 256);
    std::vector<AliasGroup> groups;
    const auto group_for = [&](PersonKey alias) {
        const auto [id, inserted] = [&] {
            if (alias.packed()) return packed_groups.try_add(alias.packed_key(), groups.size());
            const Text name = alias.raw_name(raw_names);
            return raw_groups.try_add(name, text_hash(name), groups.size());
        }();
        if (inserted) groups.push_back({alias});
        return id;
    };

    std::vector<uint32_t> event_group, click_group;
    event_group.reserve(event_count);
    click_group.reserve(click_count);
    for (const auto& lane : event_lanes) {
        for (const auto& event : lane[part]) {
            const uint32_t id = group_for(event.alias);
            ++groups[id].event_end;
            event_group.push_back(id);
        }
    }
    for (const auto& lane : click_lanes) {
        for (const Click& click : lane[part]) {
            const uint32_t id = group_for(click.alias);
            ++groups[id].click_end;
            click_group.push_back(id);
        }
    }

    size_t events = 0, clicks = 0;
    for (auto& group : groups) {
        const size_t group_events = std::exchange(group.event_end, events);
        const size_t group_clicks = std::exchange(group.click_end, clicks);
        events += group_events;
        clicks += group_clicks;
    }
    if (events != 0) storage.events = std::make_unique_for_overwrite<TimelineEvent[]>(events);
    if (clicks != 0) storage.clicks = std::make_unique_for_overwrite<TimedClick[]>(clicks);
    size_t i = 0;
    for (auto& lane : event_lanes) {
        for (const auto& event : lane[part]) {
            const size_t offset = groups[event_group[i++]].event_end++;
            storage.events[offset] = event.timed;
        }
        std::vector<Event<TimelineEvent>>().swap(lane[part]);
    }
    i = 0;
    for (auto& lane : click_lanes) {
        for (const Click& click : lane[part]) {
            const size_t offset = groups[click_group[i++]].click_end++;
            storage.clicks[offset] = click.row;
        }
        std::vector<Click>().swap(lane[part]);
    }

    size_t event_begin = 0, click_begin = 0;
    for (const auto& group : groups) {
        const auto person = linked.person(group.alias, part, raw_names);
        std::span<TimelineEvent> event_span;
        std::span<TimedClick> click_span;
        if (event_begin != group.event_end) {
            event_span = {storage.events.get() + event_begin, group.event_end - event_begin};
        }
        if (click_begin != group.click_end) {
            click_span = {storage.clicks.get() + click_begin, group.click_end - click_begin};
        }
        event_begin = group.event_end;
        click_begin = group.click_end;
        // An unlinked alias cannot share its clicks with another fragment.
        if (!event_span.empty() || person.key.linked()) {
            fragments[person.partition].push_back({person.key, event_span, click_span});
        }
    }
    return storage;
}

struct UserDay {
    Text person, category;
    int32_t day;
    int64_t sessions = 0, events = 0, revenue = 0;
};

struct CreditTotals {
    int64_t sessions = 0, purchasing = 0, revenue = 0;

    void credit(int64_t amount) {
        ++sessions;
        purchasing += amount != 0;
        revenue += amount;
    }

    void merge(const CreditTotals& other) {
        sessions += other.sessions;
        purchasing += other.purchasing;
        revenue += other.revenue;
    }
};

struct Totals : CreditTotals {
    int64_t assists = 0;

    void merge(const Totals& other) {
        CreditTotals::merge(other);
        assists += other.assists;
    }
};

struct CampaignDay {
    Text id, channel;
    int32_t day;
    Totals totals;
};

using SparseCampaignTotals = std::unordered_map<uint64_t, Totals>;

uint64_t campaign_key(int32_t day, uint32_t campaign) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(day)) << 32) | campaign;
}

int64_t utc_day_number(int64_t ms) {
    return ms / input::day_ms - (ms % input::day_ms < 0);
}

int32_t utc_day(int64_t ms) {
    return static_cast<int32_t>(utc_day_number(ms));
}

struct DayCampaignTotals {
    CreditTotals* credits = nullptr;
    int64_t* assists = nullptr;
    SparseCampaignTotals* sparse = nullptr;
    uint64_t day_key = 0;
};

// Workers keep independent totals. Within a small validated day/campaign
// domain, the assist loop touches only the compact counter array.
class CampaignTotals {
    size_t campaign_count;
    int64_t first_day = 0;
    std::vector<CreditTotals> credits;
    std::vector<int64_t> assists;
    SparseCampaignTotals sparse;

public:
    CampaignTotals(input::Window window, size_t campaigns) : campaign_count(campaigns) {
        constexpr size_t max_cells = 65536;
        const int64_t first = utc_day_number(window.start);
        const int64_t last = utc_day_number(window.end - 1);
        const auto days = static_cast<uint64_t>(last - first + 1);
        if (first >= std::numeric_limits<int32_t>::min() &&
            last <= std::numeric_limits<int32_t>::max() &&
            campaign_count != 0 && days <= max_cells / campaign_count) {
            first_day = first;
            credits.resize(days * campaign_count);
            assists.resize(credits.size());
        }
    }

    DayCampaignTotals day(int32_t day) {
        if (credits.empty()) return {nullptr, nullptr, &sparse, campaign_key(day, 0)};
        const auto offset = static_cast<size_t>(static_cast<int64_t>(day) - first_day) * campaign_count;
        return {credits.data() + offset, assists.data() + offset};
    }

    void merge(const CampaignTotals& other) {
        if (credits.empty()) {
            for (const auto& [key, value] : other.sparse) sparse[key].merge(value);
            return;
        }
        for (size_t i = 0; i < credits.size(); ++i) {
            credits[i].merge(other.credits[i]);
            assists[i] += other.assists[i];
        }
    }

    void append_rows(const Lookups& lookups, std::vector<CampaignDay>& rows) const {
        if (credits.empty()) {
            rows.reserve(sparse.size());
            for (const auto& [key, value] : sparse) {
                const auto campaign = static_cast<uint32_t>(key);
                const auto day = static_cast<int32_t>(key >> 32);
                rows.push_back({lookups.campaign[campaign], lookups.channel[campaign], day, value});
            }
            return;
        }
        rows.reserve(credits.size());
        for (size_t offset = 0; offset < credits.size(); offset += campaign_count) {
            const auto day = static_cast<int32_t>(first_day + static_cast<int64_t>(offset / campaign_count));
            for (size_t campaign = 0; campaign < campaign_count; ++campaign) {
                const size_t i = offset + campaign;
                if (credits[i].sessions == 0 && assists[i] == 0) continue;
                rows.push_back({lookups.campaign[campaign], lookups.channel[campaign], day,
                                {credits[i], assists[i]}});
            }
        }
    }
};

class Attribution {
    struct Count { uint32_t count = 0, index = 0; };
    std::vector<Count> counts;
    std::vector<uint32_t> active;
    std::span<const TimedClick> clicks;
    size_t left = 0, right = 0, latest = 0;

    void record(uint32_t winner, int64_t revenue, const DayCampaignTotals& totals) const {
        if (totals.credits != nullptr) {
            totals.credits[winner].credit(revenue);
            for (uint32_t campaign : active) ++totals.assists[campaign];
        } else {
            auto& sparse = *totals.sparse;
            sparse[totals.day_key | winner].credit(revenue);
            for (uint32_t campaign : active) ++sparse[totals.day_key | campaign].assists;
        }
    }

public:
    explicit Attribution(size_t campaigns) : counts(campaigns) {}
    void next_person(std::span<const TimedClick> person_clicks) {
        for (uint32_t campaign : active) counts[campaign].count = 0;
        active.clear();
        clicks = person_clicks;
        left = right = latest = 0;
    }

    void credit_single(int64_t start, int64_t revenue, std::span<const TimedClick> person_clicks,
                       const DayCampaignTotals& totals) {
        next_person(person_clicks);
        const TimedClick* winner = nullptr;
        for (const auto& click : person_clicks) {
            if (click.time > start ||
                static_cast<uint64_t>(start) - static_cast<uint64_t>(click.time) > input::week_ms) {
                continue;
            }
            if (winner == nullptr || click.time > winner->time ||
                (click.time == winner->time && click.id_text() < winner->id_text())) {
                winner = &click;
            }
            auto& count = counts[click.campaign].count;
            if (count == 0) {
                count = 1;
                active.push_back(click.campaign);
            }
        }
        record(winner == nullptr ? 0 : winner->campaign, revenue, totals);
    }

    void credit(int64_t start, int64_t revenue, const DayCampaignTotals& totals) {
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
        record(winner, revenue, totals);
    }
};

template<class TimelineEvent>
void summarize_person_fragments(std::array<FragmentShards<TimelineEvent>, partitions>& fragments,
                                size_t part, std::span<const Text> linked_names, const RawNameLanes& raw_names,
                                DecodedNames& decoded_names, const Lookups& lookups, input::Window window,
                                std::vector<UserDay>& users, CampaignTotals& campaigns) {
    const int64_t origin = window.start;
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

    const auto prepare_clicks = [&](std::span<TimedClick> clicks) {
        if (clicks.size() > 1) {
            std::sort(clicks.begin(), clicks.end(), [](const auto& a, const auto& b) {
                return a.time != b.time ? a.time < b.time : a.id_text() < b.id_text();
            });
        }
        attribution.next_person(clicks);
    };

    constexpr size_t minimum_bucket_events = 16;
    constexpr size_t maximum_buckets = 2048;
    constexpr size_t maximum_categories = 64;
    constexpr uint64_t width = session_gap;
    static_assert(maximum_buckets <= UINT16_MAX + 1);
    static_assert(maximum_buckets * width <= (uint64_t{1} << 32));
    const uint64_t span = static_cast<uint64_t>(window.end) - static_cast<uint64_t>(origin);
    const uint64_t buckets = 1 + (span - 1) / width; // The manifest window is nonempty.
    const size_t categories = lookups.category.size();
    const auto use_buckets = [&](size_t event_count) {
        return event_count >= minimum_bucket_events && buckets <= maximum_buckets &&
               categories <= maximum_categories;
    };

    struct Bin {
        uint32_t first, last;
        size_t count = 0;
        uint64_t purchased = 0;
    };
    std::vector<Bin> bins;
    std::vector<uint16_t> touched;
    std::unique_ptr<int64_t[]> by_category;
    const auto summarize_buckets = [&](Text person, std::span<TimedClick> clicks, auto&& visit_spans) {
        if (bins.empty()) {
            bins.resize(buckets);
            touched.reserve(buckets);
        }
        touched.clear();
        visit_spans([&](std::span<const TimelineEvent> events) {
            for (const auto& event : events) {
                const auto offset = static_cast<uint32_t>(static_cast<uint64_t>(event.absolute_time(origin)) -
                                                          static_cast<uint64_t>(origin));
                const size_t bucket = offset / width;
                auto& bin = bins[bucket];
                if (bin.count == 0) {
                    touched.push_back(static_cast<uint16_t>(bucket));
                    bin.first = bin.last = offset;
                } else {
                    bin.first = std::min(bin.first, offset);
                    bin.last = std::max(bin.last, offset);
                }
                ++bin.count;
                if (const uint32_t price = event.price(); price != 0) {
                    if (!by_category) by_category = std::make_unique_for_overwrite<int64_t[]>(buckets * categories);
                    const uint32_t category = event.category();
                    auto& subtotal = by_category[bucket * categories + category];
                    const uint64_t bit = uint64_t{1} << category;
                    if (bin.purchased & bit) {
                        subtotal += price;
                    } else {
                        subtotal = price;
                        bin.purchased |= bit;
                    }
                }
            }
        });

        // Events in one half-open half-hour bucket cannot split a session.
        // Only boundaries between occupied buckets matter.
        UserDay daily{person, {}, 0};
        DayCampaignTotals day_totals;
        int64_t start = 0, revenue = 0;
        uint64_t last = 0;
        bool started = false;
        bool multiple_sessions = false;
        std::sort(touched.begin(), touched.end());
        for (size_t i : touched) {
            auto& bin = bins[i];
            if (!started || bin.first - last > width) {
                if (started) {
                    if (!multiple_sessions) {
                        prepare_clicks(clicks);
                        multiple_sessions = true;
                    }
                    attribution.credit(start, revenue, day_totals);
                }
                // Eligible windows bound every relative offset to maximum_buckets * width.
                start = origin + static_cast<int64_t>(bin.first);
                revenue = 0;
                const int32_t day = utc_day(start);
                if (!started || day != daily.day) {
                    if (started) finish_day(daily);
                    daily = {person, {}, day};
                    day_totals = campaigns.day(day);
                }
                started = true;
                ++daily.sessions;
            }
            last = bin.last;
            daily.events += bin.count;
            // A cleared mask makes the previous subtotals unreachable.
            for (uint64_t remaining = bin.purchased; remaining != 0; remaining &= remaining - 1) {
                const size_t category = std::countr_zero(remaining);
                const int64_t subtotal = by_category[i * categories + category];
                auto& sum = category_revenue[category];
                if (sum == 0) purchased_categories.push_back(category);
                sum += subtotal;
                daily.revenue += subtotal;
                revenue += subtotal;
            }
            bin.purchased = 0;
            bin.count = 0;
        }
        if (multiple_sessions) {
            attribution.credit(start, revenue, day_totals);
        } else {
            attribution.credit_single(start, revenue, clicks, day_totals);
        }
        finish_day(daily);
    };

    const auto summarize_timeline = [&](Text person, std::span<TimelineEvent> events, std::span<TimedClick> clicks) {
        if (events.empty()) return;
        if (use_buckets(events.size())) {
            summarize_buckets(person, clicks, [&](const auto& collect) { collect(events); });
            return;
        }
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            return a.sort_key() < b.sort_key();
        });
        prepare_clicks(clicks);

        size_t cursor = 0;
        UserDay daily{person, {}, utc_day(events[cursor].absolute_time(origin))};
        DayCampaignTotals day_totals = campaigns.day(daily.day);
        do {
            const int64_t start = events[cursor].absolute_time(origin);
            const int32_t day = utc_day(start);
            if (day != daily.day) {
                finish_day(daily);
                daily = {person, {}, day};
                day_totals = campaigns.day(day);
            }
            int64_t previous, revenue = 0;
            const size_t first_event = cursor;
            do {
                const auto& event = events[cursor++];
                previous = event.clock();
                const uint32_t price = event.price();
                revenue += price;
                if (price != 0) {
                    const uint32_t category = event.category();
                    auto& sum = category_revenue[category];
                    if (sum == 0) purchased_categories.push_back(category);
                    sum += price;
                }
            } while (cursor < events.size() &&
                     static_cast<uint64_t>(events[cursor].clock()) - static_cast<uint64_t>(previous) <=
                         static_cast<uint64_t>(session_gap));
            ++daily.sessions;
            daily.events += cursor - first_event;
            daily.revenue += revenue;
            attribution.credit(start, revenue, day_totals);
        } while (cursor < events.size());
        finish_day(daily);
    };

    // Each canonical partition owns its fragments and their row ranges.
    // Unlinked names occur once by construction; linked ordinals directly
    // index their fragment chains without a second string lookup.
    std::vector<PersonFragment<TimelineEvent>*> linked_heads(linked_names.size(), nullptr);
    for (auto& source : fragments) {
        for (auto& fragment : source[part]) {
            const PersonKey person = fragment.person;
            if (person.linked()) {
                auto& head = linked_heads[person.ordinal()];
                fragment.next = head;
                head = &fragment;
            } else {
                const Text name = person.packed() ? decode_name(decoded_names, person.packed_key())
                                                  : person.raw_name(raw_names);
                summarize_timeline(name, fragment.events, fragment.clicks);
            }
        }
    }

    std::vector<TimelineEvent> merged_events;
    std::vector<TimedClick> merged_clicks;
    for (size_t ordinal = 0; ordinal < linked_heads.size(); ++ordinal) {
        const auto* head = linked_heads[ordinal];
        if (head == nullptr) continue;
        if (head->next == nullptr) {
            summarize_timeline(linked_names[ordinal], head->events, head->clicks);
            continue;
        }
        size_t event_count = 0, click_count = 0;
        for (auto* fragment = head; fragment != nullptr; fragment = fragment->next) {
            event_count += fragment->events.size();
            click_count += fragment->clicks.size();
        }
        if (event_count == 0) continue;

        const bool bucket_events = use_buckets(event_count);
        if (!bucket_events) {
            merged_events.clear();
            merged_events.reserve(event_count);
        }
        merged_clicks.clear();
        merged_clicks.reserve(click_count);
        for (auto* fragment = head; fragment != nullptr; fragment = fragment->next) {
            if (!bucket_events && !fragment->events.empty()) {
                merged_events.insert(merged_events.end(), fragment->events.begin(), fragment->events.end());
            }
            if (!fragment->clicks.empty()) {
                merged_clicks.insert(merged_clicks.end(), fragment->clicks.begin(), fragment->clicks.end());
            }
        }
        if (bucket_events) {
            summarize_buckets(linked_names[ordinal], merged_clicks, [&](const auto& collect) {
                for (auto* fragment = head; fragment != nullptr; fragment = fragment->next) {
                    collect(fragment->events);
                }
            });
        } else {
            summarize_timeline(linked_names[ordinal], merged_events, merged_clicks);
        }
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
    std::unique_ptr<LinkedIds> linked_ids;

    const auto event_chunks = input::chunks(event_files, chunk_size);
    const auto click_chunks = input::chunks(click_files, chunk_size);
    size_t event_bytes = 0;
    for (Text chunk : event_chunks) event_bytes += chunk.size();
    // A capacity hint only. Vectors still grow normally for smaller rows or
    // uneven work distribution.
    const size_t candidate_hint = event_bytes / (64 * partitions * threads);
    const TimeOrigins time_origin{window.start, window.start / 1000};
    std::vector<std::deque<WideCandidateTimes>> wide_candidate_times(threads);
    RawNameLanes raw_candidate_names(threads);
    std::vector<Shards<PackedCandidate>> packed_candidate_lanes(threads);
    std::vector<Shards<PackedPurchase>> packed_purchase_lanes(threads);
    std::vector<Shards<Candidate>> candidate_lanes(threads);
    std::vector<Shards<Click>> click_lanes(threads);
    std::vector<std::deque<Text>> wide_click_ids(threads);
    RawNameLanes raw_names(threads);

    parallel(1 + event_chunks.size() + click_chunks.size(), threads, [&](size_t lane, size_t job) {
        if (job == 0) {
            linked_ids = std::make_unique<LinkedIds>(link_files);
            return;
        }
        --job;
        if (job >= event_chunks.size()) {
            read_clicks(click_chunks[job - event_chunks.size()], click_lanes[lane],
                        lookups, window, raw_names[lane], lane, wide_click_ids[lane]);
            return;
        }
        if (candidate_hint >= 1024 && packed_candidate_lanes[lane][0].capacity() == 0) {
            for (auto& candidates : packed_candidate_lanes[lane]) candidates.reserve(candidate_hint);
        }
        input::read<7>(event_chunks[job], [&](const auto& row) {
            const Text id = row[0];
            if (id.empty()) return;

            const uint64_t packed = pack_short_id(id);
            const auto payload = event_payload(row, window, lookups);
            if (packed != nonpacked_id) {
                const size_t part = shard(packed_alias_hash(packed));
                auto& candidates = packed_candidate_lanes[lane][part];
                if (payload.price != 0) {
                    packed_purchase_lanes[lane][part].push_back(
                        {candidates.size(), payload.price, payload.category});
                }
                const uint64_t times = pack_candidate_times(payload, time_origin, wide_candidate_times[lane]);
                const uint64_t alias = pack_candidate_alias(payload.user, raw_candidate_names[lane]);
                candidates.push_back({packed, alias, times});
            } else {
                const uint64_t hash = text_hash(id);
                candidate_lanes[lane][shard(hash)].push_back({hash, id, payload});
            }
        });
    });
    timer.mark("build links and parse events and clicks");
    const LinkedIds& linked = *linked_ids;

    std::array<DecodedNames, partitions> decoded_names;
    Shards<UserDay> user_rows;
    std::vector<CampaignTotals> campaign_totals;
    campaign_totals.reserve(threads);
    for (size_t lane = 0; lane < threads; ++lane) {
        campaign_totals.emplace_back(window, lookups.campaign.size());
    }
    const auto build_timelines = [&]<class TimelineEvent>() {
        std::vector<Shards<Event<TimelineEvent>>> event_lanes(threads);
        const size_t event_hint = event_bytes / (72 * partitions * threads);
        parallel(partitions, threads, [&](size_t lane, size_t part) {
            if (event_hint >= 1024 && event_lanes[lane][0].capacity() == 0) {
                for (auto& events : event_lanes[lane]) events.reserve(event_hint);
            }
            deduplicate(packed_candidate_lanes, part, event_lanes[lane], raw_names[lane], lane,
                        time_origin, raw_candidate_names, packed_purchase_lanes);
            deduplicate(candidate_lanes, part, event_lanes[lane], raw_names[lane], lane,
                        time_origin, raw_candidate_names);
        });
        raw_candidate_names.clear();
        wide_candidate_times.clear();
        timer.mark("event dedup and alias routing");

        // Fragment spans reference these arrays through the final sort pass.
        std::array<AliasStorage<TimelineEvent>, partitions> alias_rows;
        auto fragments = std::make_unique<std::array<FragmentShards<TimelineEvent>, partitions>>();
        parallel(partitions, threads, [&](size_t, size_t part) {
            alias_rows[part] = group_aliases<TimelineEvent>(
                event_lanes, click_lanes, part, raw_names, linked, (*fragments)[part]);
        });
        timer.mark("group aliases and resolve fragments");
        parallel(partitions, threads, [&](size_t lane, size_t part) {
            summarize_person_fragments<TimelineEvent>(
                *fragments, part, linked.names(part), raw_names, decoded_names[part],
                lookups, window, user_rows[part], campaign_totals[lane]);
        });
    };
    if (CompactTimedEvent::fits(window, lookups.category.size())) {
        build_timelines.operator()<CompactTimedEvent>();
    } else {
        build_timelines.operator()<TimedEvent>();
    }
    timer.mark("merge fragments, sort, sessionize and attribute");

    CampaignTotals& totals = campaign_totals.front();
    for (size_t lane = 1; lane < campaign_totals.size(); ++lane) {
        totals.merge(campaign_totals[lane]);
    }
    std::array<std::vector<CampaignDay>, 1> campaign_rows;
    totals.append_rows(lookups, campaign_rows[0]);
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
