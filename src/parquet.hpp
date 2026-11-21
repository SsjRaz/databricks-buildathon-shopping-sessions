#pragma once

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// The writer uses Parquet data page v1 with plain values and no compression.
// The footer and page headers use the Thrift compact protocol.
namespace parquet {
using Bytes = std::vector<char>;
enum Wire { Stop = 0, I32 = 5, I64 = 6, Binary = 8, List = 9, Struct = 12 };
enum Type { Int32 = 1, Int64 = 2, String = 6 };
enum Logical { None = -1, Utf8 = 0, Date = 6 };

inline void varint(Bytes& out, uint64_t value) {
    while (value >= 128) {
        out.push_back(static_cast<char>(value | 128));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

inline void little(Bytes& out, uint32_t value) {
    static_assert(std::endian::native == std::endian::little);
    const char* first = reinterpret_cast<const char*>(&value);
    out.insert(out.end(), first, first + sizeof(value));
}

class Compact {
    int previous = 0;
public:
    Bytes bytes;
    void field(int id, Wire type) {
        const int delta = id - previous;
        if (delta < 1 || delta > 15) throw std::logic_error("invalid compact field order");
        bytes.push_back(static_cast<char>((delta << 4) | type));
        previous = id;
    }
    void integer(int64_t value) {
        varint(bytes, (static_cast<uint64_t>(value) << 1) ^ (value < 0 ? UINT64_MAX : 0));
    }
    void number(int id, int64_t value, Wire type = I32) {
        field(id, type);
        integer(value);
    }
    void string(std::string_view value) {
        varint(bytes, value.size());
        bytes.insert(bytes.end(), value.begin(), value.end());
    }
    void string(int id, std::string_view value) {
        field(id, Binary);
        string(value);
    }
    void list(int id, Wire item, size_t count) {
        field(id, List);
        const size_t short_count = count < 15 ? count : 15;
        bytes.push_back(static_cast<char>((short_count << 4) | item));
        if (count >= 15) varint(bytes, count);
    }
    void structure(Compact content) {
        bytes.insert(bytes.end(), content.bytes.begin(), content.bytes.end());
        bytes.push_back(Stop);
    }
    void structure(int id, Compact content) {
        field(id, Struct);
        structure(std::move(content));
    }
};

struct Column {
    std::string_view name;
    Type type;
    Logical logical = None;
    bool nullable = false;
};

template<class Row>
struct Field {
    Column column;
    int64_t (*number)(const Row&) = nullptr;
    std::string_view (*string)(const Row&) = nullptr;
};

class Writer {
    struct Chunk {
        int64_t offset;
        int64_t size;
    };
    struct Group {
        size_t rows;
        int64_t size = 0;
        std::vector<Chunk> chunks;
    };
    std::ofstream file;
    std::vector<Column> columns;
    std::vector<Group> groups;
    int64_t position = 0;
    int64_t rows = 0;

    void write(std::span<const char> bytes) {
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        position += bytes.size();
    }

    static Compact schema(Column column) {
        Compact entry;
        entry.number(1, column.type);
        entry.number(3, column.nullable ? 1 : 0);
        entry.string(4, column.name);
        if (column.logical != None) entry.number(6, column.logical);
        return entry;
    }

    static Compact chunk_metadata(Column column, Chunk chunk, size_t count) {
        Compact meta;
        meta.number(1, column.type);
        meta.list(2, I32, 2);
        meta.integer(0); // PLAIN
        meta.integer(3); // RLE, for definition levels
        meta.list(3, Binary, 1);
        meta.string(column.name);
        meta.number(4, 0); // UNCOMPRESSED
        meta.number(5, count, I64);
        meta.number(6, chunk.size, I64);
        meta.number(7, chunk.size, I64);
        meta.number(9, chunk.offset, I64);
        Compact entry;
        entry.number(2, chunk.offset, I64);
        entry.structure(3, std::move(meta));
        return entry;
    }

public:
    template<class Row>
    Writer(const std::filesystem::path& path, std::span<const Field<Row>> fields) {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(path, std::ios::binary);
        for (const auto& field : fields) columns.push_back(field.column);
        write(std::span("PAR1", 4));
    }

    template<class Row>
    void append(std::span<const Field<Row>> fields, std::span<const Row> batch) {
        if (batch.empty()) return;
        Group group{batch.size(), 0, {}};
        Bytes data;
        for (const auto& field : fields) {
            data.clear();
            data.reserve(batch.size() * 16);
            if (field.column.nullable) {
                const size_t blocks = (batch.size() + 7) / 8;
                Bytes levels;
                varint(levels, (blocks << 1) | 1);
                const size_t start = levels.size();
                levels.resize(start + blocks, 0);
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (field.string(batch[i]).data() != nullptr) {
                        levels[start + i / 8] |= static_cast<char>(1 << (i % 8));
                    }
                }
                little(data, static_cast<uint32_t>(levels.size()));
                data.insert(data.end(), levels.begin(), levels.end());
            }

            if (field.column.type == String) {
                for (const Row& row : batch) {
                    const auto value = field.string(row);
                    if (field.column.nullable && value.data() == nullptr) continue;
                    little(data, static_cast<uint32_t>(value.size()));
                    if (!value.empty()) data.insert(data.end(), value.begin(), value.end());
                }
            } else if (field.column.type == Int32) {
                for (const Row& row : batch) little(data, static_cast<uint32_t>(field.number(row)));
            } else {
                for (const Row& row : batch) {
                    const uint64_t value = field.number(row);
                    little(data, static_cast<uint32_t>(value));
                    little(data, static_cast<uint32_t>(value >> 32));
                }
            }

            Compact page;
            page.number(1, batch.size());
            page.number(2, 0); // PLAIN
            page.number(3, 3); // RLE
            page.number(4, 3); // RLE
            Compact header;
            header.number(1, 0); // DATA_PAGE
            header.number(2, data.size());
            header.number(3, data.size());
            header.structure(5, std::move(page));
            header.bytes.push_back(Stop);
            const int64_t offset = position;
            write(header.bytes);
            write(data);
            group.chunks.push_back({offset, position - offset});
            group.size += position - offset;
        }
        rows += group.rows;
        groups.push_back(std::move(group));
    }

    void finish() {
        Compact footer;
        footer.number(1, 1);
        footer.list(2, Struct, columns.size() + 1);
        Compact root;
        root.string(4, "schema");
        root.number(5, columns.size());
        footer.structure(std::move(root));
        for (Column column : columns) footer.structure(schema(column));
        footer.number(3, rows, I64);
        footer.list(4, Struct, groups.size());
        for (const auto& group : groups) {
            Compact entry;
            entry.list(1, Struct, columns.size());
            for (size_t i = 0; i < columns.size(); ++i) {
                entry.structure(chunk_metadata(columns[i], group.chunks[i], group.rows));
            }
            entry.number(2, group.size, I64);
            entry.number(3, group.rows, I64);
            footer.structure(std::move(entry));
        }
        footer.string(6, "shopping-sessions");
        footer.bytes.push_back(Stop);
        write(footer.bytes);
        Bytes length;
        little(length, static_cast<uint32_t>(footer.bytes.size()));
        write(length);
        write(std::span("PAR1", 4));
        file.close();
    }
};

template<class Row, class Groups>
void write_table(const std::filesystem::path& path, std::span<const Field<Row>> fields,
                 const Groups& groups) {
    std::filesystem::create_directories(path);
    Writer writer(path / "part-0.parquet", fields);
    constexpr size_t batch_size = 65536;
    for (const auto& group : groups) {
        std::span<const Row> rows(group);
        while (!rows.empty()) {
            const size_t count = std::min(batch_size, rows.size());
            writer.append(fields, rows.first(count));
            rows = rows.subspan(count);
        }
    }
    writer.finish();
}
} // namespace parquet
