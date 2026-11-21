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
#if defined(__AVX2__) && defined(__BMI2__)
    static_assert(Count > 0 && Count < 63, "field masks must fit in one word");
    const char* start = bytes.data();
    if (bytes.empty()) return;
    const char* end = start + bytes.size();
    while (end - start >= 128) {
        // Load two contiguous 64-byte masks. Most records fit in this window.
        // Mask off subsequent records before interpreting an index.
        auto masks = [](const char* p, uint64_t& separators, uint64_t& quotes, uint64_t& lfs) {
            const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            const auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
            auto mask = [a,b](char target) {
                const auto x = _mm256_set1_epi8(target);
                const auto lo = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, x)));
                const auto hi = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, x)));
                return lo | (uint64_t{hi} << 32);
            };
            lfs = mask('\n');
            separators = mask(',') | lfs;
            quotes = mask('"');
        };
        uint64_t lo, qlo, lflo, hi = 0, qhi = 0, lfhi = 0;
        masks(start, lo, qlo, lflo);
        if (lflo == 0) masks(start + 64, hi, qhi, lfhi);
        const char* newline;
        if (lflo != 0) {
            const auto offset = std::countr_zero(lflo);
            newline = start + offset;
            const uint64_t through = lflo ^ (lflo - 1);
            lo &= through;
            qlo &= through;
        } else if (lfhi != 0) {
            const auto offset = std::countr_zero(lfhi);
            newline = start + 64 + offset;
            const uint64_t through = lfhi ^ (lfhi - 1);
            hi &= through;
            qhi &= through;
        } else {
            newline = find_byte(start + 128, end, '\n');
            if (fields(Text(start, newline - start), record)) fn(record);
            start = newline == end ? end : newline + 1;
            continue;
        }
        const size_t low_count = std::popcount(lo);
        if (low_count + std::popcount(hi) == Count) {
            const uint64_t low_before = (lo << 1) | 1;
            const uint64_t low_after = (lo >> 1) | (hi << 63);
            const uint64_t high_before = (hi << 1) | (lo >> 63);
            const uint64_t high_after = hi >> 1;
            const bool interior_or_single = (qlo & ~(low_before ^ low_after)) ||
                                             (qhi & ~(high_before ^ high_after));
            // The next and previous quote of each delimiter describe the
            // beginning and end of adjacent fields, including the first.
            const uint64_t opening = ((qlo & 1) |
                (_pext_u64((qlo >> 1) | (qhi << 63), lo) << 1) |
                (_pext_u64(qhi >> 1, hi) << (low_count + 1)));
            const uint64_t closing = _pext_u64(qlo << 1, lo) |
                         (_pext_u64((qhi << 1) | (qlo >> 63), hi) << low_count);
            if (!interior_or_single && opening == closing) {
                std::array<unsigned, Count> ends;
                // Extra low-word positions are replaced by the high word.
                for (size_t i = 0; i < Count; ++i) {
                    ends[i] = std::countr_zero(lo);
                    lo &= lo - 1;
                }
                for (size_t i = 0; hi; ++i) {
                    ends[low_count + i] = 64 + std::countr_zero(hi);
                    hi &= hi - 1;
                }
                size_t first = 0;
                for (size_t i = 0; i < Count; ++i) {
                    const size_t q = (opening >> i) & 1;
                    record[i] = Text(start + first + q, ends[i] - first - 2 * q);
                    first = ends[i] + 1;
                }
                fn(record);
            }
        }
        start = newline + 1;
    }
    lines(Text(start, end - start), [&](Text line) {
        if (fields(line, record)) fn(record);
    });
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
    const unsigned slot = ((p[8] & 3) << 4) | (p[9] & 15);
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

// Compare and decode HH:MM:SS from a single, bounded eight-byte load.
// XOR turns the expected delimiters into zeroes and ASCII digits into
// nibbles. A high nibble or a carry out of the permitted low digit rejects
// that byte. A separate hour check covers the upper bound of 23.
inline uint64_t clock_digits(const char* p) {
    uint64_t bytes;
    std::memcpy(&bytes, p, sizeof(bytes));
    return bytes ^ 0x30303a30303a3030ULL;
}

inline bool valid_clock(uint64_t digits) {
    constexpr uint64_t high = 0xf0f0f0f0f0f0f0f0ULL;
    constexpr uint64_t bias = 0x060a0f060a0f060dULL;
    return ((digits | (digits + bias)) & high) == 0 &&
           ((digits & 0xff) != 2 || ((digits >> 8) & 0xff) <= 3);
}

inline int clock_seconds(uint64_t digits) {
#ifdef __AVX2__
    // The two colon lanes have zero weight. The intermediate values are
    // hour, tens of minutes, units of minutes and seconds.
    const __m128i pairs = _mm_maddubs_epi16(_mm_cvtsi64_si128(digits),
                                            _mm_setr_epi8(10, 1, 0, 10, 1, 0, 10, 1,
                                                          0, 0, 0, 0, 0, 0, 0, 0));
    const __m128i parts = _mm_madd_epi16(pairs, _mm_setr_epi16(3600, 60, 60, 1, 0, 0, 0, 0));
    return _mm_cvtsi128_si32(parts) + _mm_extract_epi32(parts, 1);
#else
    const uint64_t pairs = (digits * 10 + (digits >> 8));
    return static_cast<int>(pairs & 255) * 3600 +
           static_cast<int>((pairs >> 24) & 255) * 60 +
           static_cast<int>((pairs >> 48) & 255);
#endif
}

inline int64_t iso_time(Text text) {
    const bool zulu = text.size() == 20;
    if (!zulu && text.size() != 25) return invalid_time;
    const char* p = text.data();
    const uint64_t digits = clock_digits(p + 11);
    const int32_t day = calendar_day(p);
    if (p[10] != 'T' || !valid_clock(digits) || day == invalid_day) return invalid_time;
    int seconds = clock_seconds(digits);
    if (zulu) {
        if (p[19] != 'Z') return invalid_time;
    } else {
        if ((p[19] != '+' && p[19] != '-') || p[22] != ':') return invalid_time;
        const int h = pair(p + 20);
        const int m = pair(p + 23);
        if (h < 0 || h > 23 || m < 0 || m > 59) return invalid_time;
        const int offset = (h * 60 + m) * 60;
        seconds += p[19] == '-' ? offset : -offset;
    }
    return day * day_ms + seconds * int64_t{1000};
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
#ifdef __AVX2__
        // Two in-bounds loads cover all 13 bytes. Validate the digits
        // together, then expand the leading five to an eight-digit group.
        const __m128i input = _mm_unpacklo_epi64(_mm_loadu_si64(text.data()),
                                                  _mm_loadu_si64(text.data() + 5));
        const __m128i digits = _mm_sub_epi8(input, _mm_set1_epi8('0'));
        const __m128i excess = _mm_subs_epu8(digits, _mm_set1_epi8(9));
        if (!_mm_testz_si128(excess, excess)) return invalid_time;
        const __m128i grouped = _mm_shuffle_epi8(digits,
            _mm_setr_epi8(-1, -1, -1, 0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15));
        const __m128i pairs = _mm_maddubs_epi16(grouped, _mm_set1_epi16(0x010a));
        const __m128i fours = _mm_madd_epi16(pairs, _mm_set1_epi32(0x00010064));
        const __m128i eights = _mm_madd_epi16(_mm_packs_epi32(fours, fours),
                                              _mm_set1_epi32(0x00012710));
        return int64_t{_mm_cvtsi128_si32(eights)} * 100000000 + _mm_extract_epi32(eights, 1);
#else
        const uint32_t first = eight_digits(text.data());
        const uint32_t last = eight_digits(text.data() + 5);
        if (first == UINT32_MAX || last == UINT32_MAX) return invalid_time;
        return int64_t{first} * 100000 + last % 100000;
#endif
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
