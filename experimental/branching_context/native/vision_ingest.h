#pragma once
#include "strata/core/vision_layout.hpp"
#include "strata/core/vision_payload.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>

namespace cow_vision {
using namespace strata::core;
struct Ingest {
    std::vector<int32_t> tokens;
    std::vector<VisionPlacement> placements;
    std::vector<VisionEmbedding> embeddings;
    VisionLayout layout;
    std::vector<const float*> rows;
    uint64_t read_bytes = 0;
};

struct RestoreInput {
    std::vector<int32_t> tokens;
    std::vector<VisionPlacement> placements;
};

// No embedding files or row pointers exist in this restore transport. Its
// canonical provenance and full input digest are checked by the Python adapter.
inline RestoreInput restore_metadata(const std::string& path, int64_t capacity,
                                     int64_t vocab, int64_t width) {
    const auto raw = read_vision_bytes(path, 32 * 1024 * 1024);
    std::istringstream input(std::string(raw.begin(), raw.end()));
    std::string version, extra;
    int64_t count = -1, images = -1;
    if (!(input >> version >> count >> images) || version != "COW_VISION_RESTORE_V1" ||
        count <= 0 || count > capacity || images < 0 || images > count)
        throw std::invalid_argument("invalid vision restore metadata extent");
    RestoreInput result;
    result.tokens.reserve(static_cast<size_t>(count));
    for (int64_t i = 0, token; i < count; ++i) {
        if (!(input >> token) || token < 0 || token >= vocab)
            throw std::invalid_argument("invalid vision restore token");
        result.tokens.push_back(static_cast<int32_t>(token));
    }
    for (int64_t i = 0; i < images; ++i) {
        VisionPlacement image;
        if (!(input >> image.start >> image.rows >> image.grid_width >> image.grid_height
                    >> image.width >> image.pad_token >> image.embedding_sha256))
            throw std::invalid_argument("incomplete vision restore placement");
        result.placements.push_back(std::move(image));
    }
    if (input >> extra) throw std::invalid_argument("trailing vision restore metadata");
    (void)vision_layout(result.tokens, result.placements, capacity, vocab, width, 1024);
    return result;
}

inline Ingest prepare(const std::string& path, const std::vector<int32_t>& previous,
                      const std::vector<VisionPlacement>& existing,
                      int64_t capacity, int64_t vocab, int64_t width) {
    // Trusted local pipe manifest; public clients will supply immutable spans,
    // never server paths. The row/payload limits apply before device access.
    const auto raw = read_vision_bytes(path, 32 * 1024 * 1024);
    std::istringstream input(std::string(raw.begin(), raw.end()));
    std::string version, extra;
    int64_t consumed = -1, count = -1, images = -1;
    if (!(input >> version >> consumed >> count >> images) || version != "COW_VISION_INPUT_V1" ||
        consumed != static_cast<int64_t>(previous.size()) || count <= 0 ||
        count > capacity - consumed || images <= 0 || images > 8)
        throw std::invalid_argument("vision ingest manifest boundary or count mismatch");
    Ingest result;
    result.tokens.reserve(static_cast<size_t>(count));
    for (int64_t i = 0, token; i < count; ++i) {
        if (!(input >> token) || token < 0 || token >= vocab)
            throw std::invalid_argument("invalid vision ingest token");
        result.tokens.push_back(static_cast<int32_t>(token));
    }
    result.placements = existing;
    std::vector<std::string> files;
    for (int64_t i = 0; i < images; ++i) {
        VisionPlacement placement;
        std::string file;
        if (!(input >> placement.start >> placement.rows >> placement.grid_width >> placement.grid_height
                    >> placement.width >> placement.pad_token >> placement.embedding_sha256 >> std::quoted(file)) ||
            placement.start < consumed)
            throw std::invalid_argument("invalid or historical image placement");
        result.placements.push_back(placement);
        files.push_back(std::move(file));
    }
    if (input >> extra) throw std::invalid_argument("trailing vision ingest metadata");
    std::vector<int32_t> all = previous;
    all.insert(all.end(), result.tokens.begin(), result.tokens.end());
    result.layout = vision_layout(all, result.placements, capacity, vocab, width, 1024);
    result.embeddings.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& placement = result.placements[existing.size() + i];
        auto image = read_vision_sve(files[i], placement.embedding_sha256, width, 1024);
        if (image.rows != placement.rows || image.grid_width != placement.grid_width ||
            image.grid_height != placement.grid_height)
            throw std::invalid_argument("embedding grid differs from input span");
        result.read_bytes += 20 + image.values.size() * sizeof(float);
        result.embeddings.push_back(std::move(image));
    }
    result.rows.assign(all.size(), nullptr);
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& image = result.placements[existing.size() + i];
        for (int64_t row = 0; row < image.rows; ++row)
            result.rows[static_cast<size_t>(image.start + row)] =
                result.embeddings[i].values.data() + row * width;
    }
    return result;
}

} // namespace cow_vision
