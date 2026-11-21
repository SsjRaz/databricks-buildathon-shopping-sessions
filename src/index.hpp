#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__BMI2__)
#include <immintrin.h>
#endif

inline uint64_t text_hash(std::string_view value) {
    return std::hash<std::string_view>{}(value);
}

// An open-addressed string-to-index map. String storage is owned by the input
// mappings, and a hash match always requires an exact byte comparison.
class StringIndex {
    struct Slot {
        std::string_view key;
        uint64_t hash = 0;
        uint32_t index = 0;
    };

    std::vector<Slot> slots;
    size_t used = 0;

    size_t locate(std::string_view key, uint64_t hash) const {
        const size_t mask = slots.size() - 1;
        // Both halves contribute to the initial slot even when the lower
        // hash bits have already been used to partition the input.
        size_t i = (hash ^ (hash >> 32)) & mask;
        while (!slots[i].key.empty() && (slots[i].hash != hash || slots[i].key != key)) {
            i = (i + 1) & mask;
        }
        return i;
    }

    void grow() {
        auto old = std::move(slots);
        slots.resize(old.size() * 2);
        for (const Slot& entry : old) {
            if (!entry.key.empty()) slots[locate(entry.key, entry.hash)] = entry;
        }
    }

public:
    static constexpr uint32_t missing = UINT32_MAX;

    explicit StringIndex(size_t expected = 0)
        : slots(std::bit_ceil(std::max<size_t>(16, expected + expected / 2))) {}

    bool empty() const { return used == 0; }

    uint32_t find(std::string_view key, uint64_t hash) const {
        const auto& slot = slots[locate(key, hash)];
        return slot.key.empty() ? missing : slot.index;
    }

    std::pair<uint32_t, bool> try_add(std::string_view key, uint64_t hash, uint32_t index) {
        if (key.empty() || index == missing) throw std::logic_error("invalid string index entry");
        size_t i = locate(key, hash);
        if (!slots[i].key.empty()) return {slots[i].index, false};
        if (used + 1 > slots.size() * 2 / 3) {
            grow();
            i = locate(key, hash);
        }
        slots[i] = {key, hash, index};
        ++used;
        return {index, true};
    }
};


// Nine ASCII bytes fit exactly in 63 bits. The remaining high bit identifies
// inputs requiring the full string index. This does not normalize any bytes.
inline constexpr uint64_t nonpacked_alias = uint64_t{1} << 63;

inline uint64_t pack_alias(std::string_view key) {
    if (key.size() != 9) return nonpacked_alias;
    constexpr uint64_t byte_mask = 0x7f7f7f7f7f7f7f7fULL;
    uint64_t first_eight;
    std::memcpy(&first_eight, key.data(), sizeof(first_eight));
    const auto ninth = static_cast<unsigned char>(key[8]);
    if ((first_eight & ~byte_mask) || (ninth & 0x80)) return nonpacked_alias;
#if defined(__BMI2__)
    return _pext_u64(first_eight, byte_mask) | (uint64_t{ninth} << 56);
#else
    uint64_t packed = 0;
    for (size_t i = 0; i < key.size(); ++i) {
        packed |= uint64_t{static_cast<unsigned char>(key[i])} << (7 * i);
    }
    return packed;
#endif
}

inline uint64_t packed_alias_hash(uint64_t key) {
    const uint64_t mixed = (key ^ (key >> 29)) * 0x9e3779b97f4a7c15ULL;
    return mixed ^ (mixed >> 32);
}

// Probe exact keys in a compact array. A miss need not read the values array.
class PackedAliasIndex {
public:
    static constexpr uint64_t missing = UINT64_MAX;

private:
    std::vector<uint64_t> keys;
    std::unique_ptr<uint64_t[]> values;
    size_t initial_capacity;
    unsigned partition_bits;
    size_t used = 0;

    size_t locate(uint64_t key, uint64_t hash) const {
        const size_t mask = keys.size() - 1;
        size_t i = (hash >> partition_bits) & mask;
        while (keys[i] != missing && keys[i] != key) i = (i + 1) & mask;
        return i;
    }

    void allocate(size_t capacity) {
        keys.assign(capacity, missing);
        values = std::make_unique_for_overwrite<uint64_t[]>(capacity);
    }

    void grow() {
        auto old_keys = std::move(keys);
        auto old_values = std::move(values);
        allocate(old_keys.size() * 2);
        for (size_t i = 0; i < old_keys.size(); ++i) {
            const uint64_t key = old_keys[i];
            if (key == missing) continue;
            const size_t slot = locate(key, packed_alias_hash(key));
            keys[slot] = key;
            values[slot] = old_values[i];
        }
    }

public:
    explicit PackedAliasIndex(size_t expected, unsigned partition_bits = 0)
        : initial_capacity(std::bit_ceil(std::max<size_t>(16, expected + expected / 2))),
          partition_bits(partition_bits) {}

    uint64_t find(uint64_t key, uint64_t hash) const {
        if (keys.empty()) return missing;
        const size_t slot = locate(key, hash);
        return keys[slot] == missing ? missing : values[slot];
    }

    std::pair<uint32_t, bool> try_add(uint64_t key, uint32_t index) {
        if (key == missing || index == StringIndex::missing) throw std::logic_error("invalid alias index entry");
        if (keys.empty()) allocate(initial_capacity);
        const uint64_t hash = packed_alias_hash(key);
        size_t i = locate(key, hash);
        if (keys[i] != missing) return {static_cast<uint32_t>(values[i]), false};
        if (used + 1 > keys.size() * 2 / 3) {
            grow();
            i = locate(key, hash);
        }
        keys[i] = key;
        values[i] = index;
        ++used;
        return {index, true};
    }

    void resolve(std::span<const size_t> canonical) {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] != missing) values[i] = canonical[values[i]];
        }
    }
};
