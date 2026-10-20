#pragma once
#include "strata/core/branch_sha256.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

// A transient projection of canonical input spans. Placement is explicit: a
// literal image-pad token alone never authorizes embedding substitution.
struct VisionPlacement {
    int64_t start = 0, rows = 0, grid_width = 0, grid_height = 0, width = 0;
    int32_t pad_token = -1;
    std::string embedding_sha256;
    bool operator==(const VisionPlacement&) const = default;
};
struct VisionRow { int64_t image = -1, row = -1; };
struct VisionLayout {
    int64_t consumed = 0, rotary_cursor = 0, image_rows = 0;
    std::vector<int32_t> positions; // [capacity][t,h,w], including following text
    std::vector<VisionRow> rows;    // [consumed], no payload or owning pointers
};

inline VisionLayout vision_layout(const std::vector<int32_t>& ids,
                                 const std::vector<VisionPlacement>& images,
                                 int64_t capacity, int64_t vocab, int64_t width,
                                 int64_t max_image_rows) {
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    auto reject = [](const char* reason) { throw std::invalid_argument(reason); };
    if (capacity <= 0 || capacity > limit || static_cast<uint64_t>(capacity) > SIZE_MAX / 12 ||
        vocab <= 0 || vocab > limit || width <= 0 || width > limit ||
        max_image_rows <= 0 || max_image_rows > limit || ids.size() > static_cast<uint64_t>(capacity))
        reject("invalid vision layout limits or context capacity");
    for (int32_t token : ids)
        if (token < 0 || token >= vocab) reject("vision token is outside the vocabulary");
    int64_t previous_end = 0;
    for (const auto& image : images) {
        if (image.start < previous_end || image.start > static_cast<int64_t>(ids.size()) ||
            image.rows <= 0 || image.rows > max_image_rows ||
            image.grid_width <= 0 || image.grid_width > limit ||
            image.grid_height <= 0 || image.grid_height > limit ||
            image.grid_width * image.grid_height != image.rows || image.width != width ||
            image.rows > static_cast<int64_t>(ids.size()) - image.start ||
            image.pad_token < 0 || image.pad_token >= vocab)
            reject("invalid image placement, grid, width or row bound");
        if (image.embedding_sha256.size() != 64 ||
            !std::all_of(image.embedding_sha256.begin(), image.embedding_sha256.end(),
                         [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }))
            reject("invalid full embedding digest");
        for (int64_t row = 0; row < image.rows; ++row)
            if (ids[static_cast<size_t>(image.start + row)] != image.pad_token)
                reject("explicit image placement contains a non-pad token");
        previous_end = image.start + image.rows;
    }
    // Public consumption boundaries require closing/control text. Native
    // batches can still cut image rows within this complete operation.
    if (!images.empty() && previous_end == static_cast<int64_t>(ids.size()))
        reject("consumed prompt cannot end inside or at the end of an image span");

    VisionLayout plan;
    plan.consumed = static_cast<int64_t>(ids.size());
    plan.positions.resize(static_cast<size_t>(capacity) * 3);
    plan.rows.resize(ids.size());
    auto put = [&](int64_t cell, int64_t t, int64_t h, int64_t w) {
        const auto offset = static_cast<size_t>(cell) * 3;
        plan.positions[offset] = static_cast<int32_t>(t);
        plan.positions[offset + 1] = static_cast<int32_t>(h);
        plan.positions[offset + 2] = static_cast<int32_t>(w);
    };
    int64_t cell = 0, rotary = 0;
    size_t next = 0;
    while (cell < plan.consumed) {
        if (next == images.size() || cell != images[next].start) {
            put(cell++, rotary, rotary, rotary);
            ++rotary;
            continue;
        }
        const auto& image = images[next];
        for (int64_t row = 0; row < image.rows; ++row) {
            put(cell + row, rotary, rotary + row / image.grid_width, rotary + row % image.grid_width);
            plan.rows[static_cast<size_t>(cell + row)] = {static_cast<int64_t>(next), row};
        }
        cell += image.rows;
        rotary += std::max(image.grid_width, image.grid_height);
        plan.image_rows += image.rows;
        ++next;
    }
    plan.rotary_cursor = rotary;
    for (; cell < capacity; ++cell, ++rotary) put(cell, rotary, rotary, rotary);
    return plan;
}

// Versioned consumed-position identity, independent of unused capacity. The
// caller separately binds the full input identities and execution profile.
inline std::string vision_position_digest(const VisionLayout& plan) {
    if (plan.consumed < 0 || plan.rotary_cursor < 0 || plan.rotary_cursor > plan.consumed ||
        static_cast<uint64_t>(plan.consumed) > plan.positions.size() / 3)
        throw std::invalid_argument("invalid consumed position plan");
    constexpr char domain[] = "COW-MROPE-v1";
    std::vector<uint8_t> bytes(domain, domain + sizeof(domain));
    auto little = [&](uint64_t value, unsigned count) {
        for (unsigned i = 0; i < count; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    };
    little(static_cast<uint64_t>(plan.consumed), 8);
    little(static_cast<uint64_t>(plan.rotary_cursor), 8);
    for (size_t i = 0; i < static_cast<size_t>(plan.consumed) * 3; ++i)
        little(static_cast<uint32_t>(plan.positions[i]), 4);
    BranchDigest digest;
    branch_sha256(bytes.data(), bytes.size(), digest);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (uint8_t byte : digest.bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}

// Native model-input binding includes the full payload identity, not just pad
// IDs or grid coordinates. Canonical provenance is bound by the outer receipt.
inline std::string vision_input_digest(const std::vector<int32_t>& ids,
                                       const std::vector<VisionPlacement>& images) {
    constexpr char domain[] = "COW-VISION-INPUT-v1";
    std::vector<uint8_t> bytes(domain, domain + sizeof(domain));
    auto little = [&](uint64_t value, unsigned count) {
        for (unsigned i = 0; i < count; ++i) bytes.push_back(static_cast<uint8_t>(value >> (8*i)));
    };
    little(ids.size(), 8);
    for (auto token : ids) little(static_cast<uint32_t>(token), 4);
    little(images.size(), 8);
    for (const auto& image : images) {
        little(image.start, 8); little(image.rows, 8);
        little(image.grid_width, 8); little(image.grid_height, 8);
        little(image.width, 8); little(image.pad_token, 8);
        bytes.insert(bytes.end(), image.embedding_sha256.begin(), image.embedding_sha256.end());
    }
    BranchDigest digest;
    branch_sha256(bytes.data(), bytes.size(), digest);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (uint8_t byte : digest.bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}

} // namespace strata::core
