#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <limits>
#include <string_view>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace input {
using Text = std::string_view;
namespace fs = std::filesystem;
constexpr int64_t day_ms = 86400000;
constexpr int64_t week_ms = 7 * day_ms;
constexpr int64_t invalid_time = std::numeric_limits<int64_t>::min();

class Mapping {
    Text content;
public:
    explicit Mapping(const fs::path& path) {
        const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), path.string());
        struct stat status{};
        if (fstat(fd, &status) < 0) {
            const int error = errno;
            close(fd);
            throw std::system_error(error, std::generic_category(), path.string());
        }
        if (status.st_size == 0) {
            close(fd);
            return;
        }
        void* start = mmap(nullptr, status.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        const int error = errno;
        close(fd);
        if (start == MAP_FAILED) throw std::system_error(error, std::generic_category(), path.string());
        content = {static_cast<const char*>(start), static_cast<size_t>(status.st_size)};
    }
    Mapping(Mapping&& other) noexcept : content(std::exchange(other.content, {})) {}
    Mapping& operator=(Mapping&&) = delete;
    Mapping(const Mapping&) = delete;
    ~Mapping() {
        if (!content.empty()) munmap(const_cast<char*>(content.data()), content.size());
    }
    Text bytes() const { return content; }
};

inline const char* find_byte(const char* start, const char* end, char value) {
    const auto* found = static_cast<const char*>(std::memchr(start, value, end - start));
    return found ? found : end;
}

inline std::vector<Mapping> map_parts(const fs::path& directory) {
    std::vector<Mapping> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.path().extension() == ".csv" || entry.path().filename() == ".csv") {
            if (entry.is_regular_file()) files.emplace_back(entry.path());
        }
    }
    return files;
}

inline std::vector<Text> chunks(const std::vector<Mapping>& files, size_t chunk_size) {
    std::vector<Text> result;
    for (const auto& file : files) {
        const Text bytes = file.bytes();
        if (bytes.empty()) continue;
        const char* end = bytes.data() + bytes.size();
        const char* p = find_byte(bytes.data(), end, '\n');
        if (p != end) ++p; // Skip exactly the first line of each file.
        while (p < end) {
            const char* cut = end;
            if (static_cast<size_t>(end - p) > chunk_size) {
                cut = find_byte(p + chunk_size, end, '\n');
                if (cut != end) ++cut;
            }
            result.emplace_back(p, cut - p);
            p = cut;
        }
    }
    return result;
}

template<class Fn>
void lines(Text bytes, Fn&& fn) {
    if (bytes.empty()) return;
    const char* end = bytes.data() + bytes.size();
    const char* start = bytes.data();
    while (start < end) {
        const char* newline = find_byte(start, end, '\n');
        fn(Text(start, newline - start));
        if (newline == end) break;
        start = newline + 1;
    }
}

// Commas always delimit and only a pair of quotes around the entire field is
// syntax. In particular, carriage returns are ordinary bytes.
template<size_t Count>
bool fields(Text line, std::array<Text, Count>& out) {
    if (line.empty()) return false;
    const char* p = line.data();
    const char* end = p + line.size();
    for (size_t i = 0; i < Count; ++i) {
        const char* start = p;
        const char* stop = find_byte(p, end, ',');
        if ((stop == end) != (i + 1 == Count)) return false;
        p = stop == end ? end : stop + 1;
        if (start != stop && *start == '"') {
            if (stop - start < 2 || stop[-1] != '"') return false;
            ++start;
            --stop;
        }
        if (find_byte(start, stop, '"') != stop) return false;
        out[i] = Text(start, stop - start);
    }
    return true;
}

template<size_t Count, class Fn>
void read(Text bytes, Fn&& fn) {
    std::array<Text, Count> record;
#ifdef __AVX2__
    if (bytes.empty()) return;
    const char* block = bytes.data();
    const char* end = block + bytes.size();
    const char* first = block;
    size_t field = 0;
    size_t quotes = 0;
    bool malformed = false;

    auto finish_field = [&]<bool EndOfLine, bool EndOfFile = false>(const char* last) {
        if (!malformed && (field == Count - 1) == EndOfLine) {
            if (quotes == 0) {
                record[field] = Text(first, last - first);
            } else if (quotes == 2 && *first == '"' && last[-1] == '"') {
                record[field] = Text(first + 1, last - first - 2);
            } else {
                malformed = true;
            }
        } else {
            malformed = true;
        }
        if constexpr (EndOfLine) {
            if (!malformed) fn(record);
            field = 0;
            malformed = false;
        } else {
            ++field;
        }
        if constexpr (!EndOfFile) first = last + 1;
        quotes = 0;
    };

    const auto comma = _mm256_set1_epi8(',');
    const auto newline = _mm256_set1_epi8('\n');
    const auto quote = _mm256_set1_epi8('"');
    auto scan = [&](const char* position, __m256i values) {
        const auto commas = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(values, comma)));
        const auto newlines = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(values, newline)));
        auto quote_bits = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(values, quote)));
        for (uint32_t separators = commas | newlines; separators != 0; separators &= separators - 1) {
            const unsigned offset = std::countr_zero(separators);
            const uint32_t bit = uint32_t{1} << offset;
            const uint32_t earlier = bit - 1;
            quotes += std::popcount(quote_bits & earlier);
            quote_bits &= ~earlier;
            if (newlines & bit) finish_field.template operator()<true>(position + offset);
            else finish_field.template operator()<false>(position + offset);
        }
        quotes += std::popcount(quote_bits);
    };

    while (end - block >= 32) {
        scan(block, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)));
        block += 32;
    }
    if (block != end) {
        alignas(32) char tail[32]{};
        std::memcpy(tail, block, end - block);
        scan(block, _mm256_load_si256(reinterpret_cast<const __m256i*>(tail)));
    }
    if (first != end || field != 0) finish_field.template operator()<true, true>(end);
#else
    lines(bytes, [&](Text line) {
        if (fields(line, record)) fn(record);
    });
#endif
}

template<size_t Count, class Fn>
void lookup(const Mapping& file, Fn&& fn) {
    const auto bytes = file.bytes();
    if (bytes.empty()) return;
    const char* end = bytes.data() + bytes.size();
    const char* header = find_byte(bytes.data(), end, '\n');
    if (header != end) read<Count>(Text(header + 1, end - header - 1), std::forward<Fn>(fn));
}

inline bool digit(char c) { return static_cast<unsigned>(c - '0') < 10; }

inline int pair(const char* p) {
    if (!digit(p[0]) || !digit(p[1])) return -1;
    return (p[0] - '0') * 10 + p[1] - '0';
}

inline bool leap(int year) { return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0); }

// Proleptic Gregorian date converted to days relative to 1970-01-01.
inline int days_since_epoch(int year, unsigned month, unsigned day) {
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned y = static_cast<unsigned>(year - era * 400);
    const unsigned m = month <= 2 ? month + 9 : month - 3;
    return era * 146097 + static_cast<int>(y * 365 + y / 4 - y / 100 +
           (153 * m + 2) / 5 + day - 1) - 719468;
}

inline constexpr int32_t invalid_day = std::numeric_limits<int32_t>::min();

inline int32_t calendar_day(const char* p) {
    struct Entry {
        uint64_t prefix = 0;
        uint16_t suffix = 0;
        int32_t days = invalid_day;
    };
    thread_local std::array<Entry, 64> cache;
    uint64_t prefix;
    uint16_t suffix;
    std::memcpy(&prefix, p, sizeof(prefix));
    std::memcpy(&suffix, p + sizeof(prefix), sizeof(suffix));

    // Index by the day of the month for the common case where most rows are
    // from one short window. Compare all ten date bytes before reusing a day.
    const unsigned slot = ((p[8] & 15) * 10 + (p[9] & 15)) & (cache.size() - 1);
    auto& entry = cache[slot];
    if (entry.prefix == prefix && entry.suffix == suffix) return entry.days;

    int32_t result = invalid_day;
    if (p[4] == '-' && p[7] == '-') {
        const int century = pair(p);
        const int year_suffix = pair(p + 2);
        const int month = pair(p + 5);
        const int day = pair(p + 8);
        if (century >= 0 && year_suffix >= 0 && month >= 1 && month <= 12 && day >= 1) {
            const int year = century * 100 + year_suffix;
            constexpr int month_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            if (day <= month_days[month - 1] + (month == 2 && leap(year))) {
                result = days_since_epoch(year, month, day);
            }
        }
    }
    entry = {prefix, suffix, result};
    return result;
}

inline int64_t iso_time(Text text) {
    const bool zulu = text.size() == 20;
    if (!zulu && text.size() != 25) return invalid_time;
    const char* p = text.data();
    if (p[10] != 'T' || p[13] != ':' || p[16] != ':') return invalid_time;
    const int32_t day = calendar_day(p);
    const int hour = pair(p + 11);
    const int minute = pair(p + 14);
    const int second = pair(p + 17);
    if (day == invalid_day || hour < 0 || hour > 23 || minute < 0 || minute > 59 ||
        second < 0 || second > 59) {
        return invalid_time;
    }
    int offset = 0;
    if (zulu) {
        if (p[19] != 'Z') return invalid_time;
    } else {
        if ((p[19] != '+' && p[19] != '-') || p[22] != ':') return invalid_time;
        const int h = pair(p + 20);
        const int m = pair(p + 23);
        if (h < 0 || h > 23 || m < 0 || m > 59) return invalid_time;
        offset = (h * 60 + m) * (p[19] == '-' ? -1 : 1);
    }
    return day * day_ms + ((hour * 60 + minute - offset) * 60 + second) * int64_t{1000};
}

inline uint64_t decimal(Text text, uint64_t limit) {
    if (text.empty()) return UINT64_MAX;
    uint64_t result = 0;
    const uint64_t prefix = limit / 10;
    const uint64_t last = limit % 10;
    for (const char c : text) {
        const auto d = static_cast<unsigned>(c - '0');
        if (d > 9 || result > prefix || (result == prefix && d > last)) return UINT64_MAX;
        result = result * 10 + d;
    }
    return result;
}

// Check eight ASCII bytes at once, then combine adjacent decimal groups.
inline uint32_t eight_digits(const char* p) {
    uint64_t bytes;
    std::memcpy(&bytes, p, sizeof(bytes));
    constexpr uint64_t high = 0xf0f0f0f0f0f0f0f0ULL;
    constexpr uint64_t ascii = 0x3030303030303030ULL;
    if ((bytes & high) != ascii || ((bytes + 0x0606060606060606ULL) & high) != ascii) {
        return UINT32_MAX;
    }
    bytes &= ~high;
    bytes = (bytes * 10 + (bytes >> 8)) & 0x00ff00ff00ff00ffULL;
    bytes = (bytes * 100 + (bytes >> 16)) & 0x0000ffff0000ffffULL;
    return static_cast<uint32_t>(bytes * 10000 + (bytes >> 32));
}

inline int64_t timestamp(Text text) {
    if (text.size() == 13) {
        const uint32_t first = eight_digits(text.data());
        const uint32_t last = eight_digits(text.data() + 5);
        if (first == UINT32_MAX || last == UINT32_MAX) return invalid_time;
        return int64_t{first} * 100000 + last % 100000;
    }
    if ((text.size() == 20 || text.size() == 25) && text[4] == '-') return iso_time(text);
    const uint64_t value = decimal(text, INT64_MAX);
    return value == UINT64_MAX ? invalid_time : static_cast<int64_t>(value);
}

struct Window {
    int64_t start, end;
};

inline Text manifest_value(Text bytes, Text key) {
    const auto key_pos = bytes.find(key);
    if (key_pos == Text::npos) return {};
    const auto colon = bytes.find(':', key_pos + key.size());
    if (colon == Text::npos) return {};
    const auto open = bytes.find('"', colon + 1);
    if (open == Text::npos) return {};
    const auto close = bytes.find('"', open + 1);
    if (close == Text::npos) return {};
    return bytes.substr(open + 1, close - open - 1);
}

inline Window parse_window(Text manifest) {
    const int64_t start = timestamp(manifest_value(manifest, "\"window_start\""));
    const int64_t end = timestamp(manifest_value(manifest, "\"window_end\""));
    if (start == invalid_time || end == invalid_time || start >= end) {
        throw std::runtime_error("invalid manifest window");
    }
    return {start, end};
}
} // namespace input
