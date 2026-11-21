#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

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
