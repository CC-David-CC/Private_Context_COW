#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cow_vision_dry {
struct TextWindow {
    std::vector<uint32_t> tokens;
    std::vector<size_t> boundaries;
    size_t text_count = 0;
};

// A transient projection of the native session's existing IDs/placements.
// Never classify an ID as an image by its vocabulary value. Walk placements
// once and copy only last_n actual text IDs, without allocating a prefix copy.
template<class Placement>
TextWindow project(const std::vector<int32_t>& ids,
                   const std::vector<Placement>& images, size_t last_n) {
    TextWindow result;
    size_t previous_end = 0, rows = 0;
    for (const auto& image : images) {
        if (image.start < 0 || image.rows <= 0 ||
                static_cast<uint64_t>(image.start) > ids.size() ||
                static_cast<uint64_t>(image.rows) > ids.size() - static_cast<size_t>(image.start) ||
                static_cast<size_t>(image.start) < previous_end)
            throw std::invalid_argument("invalid native image placement for DRY projection");
        previous_end = static_cast<size_t>(image.start + image.rows);
        rows += static_cast<size_t>(image.rows);
    }
    result.text_count = ids.size() - rows;
    if (last_n == 0) return result;
    const size_t offset = result.text_count - std::min(last_n, result.text_count);
    result.tokens.reserve(std::min(last_n, result.text_count));
    size_t text_position = 0, cell = 0;
    const auto append = [&](size_t end) {
        const size_t length = end - cell;
        const size_t skip = offset > text_position ? std::min(offset - text_position, length) : 0;
        for (size_t i = cell + skip; i < end; ++i) {
            if (ids[i] < 0) throw std::invalid_argument("negative text ID in DRY projection");
            result.tokens.push_back(static_cast<uint32_t>(ids[i]));
        }
        text_position += length;
    };
    for (const auto& image : images) {
        append(static_cast<size_t>(image.start));
        if (text_position >= offset) {
            const size_t boundary = text_position - offset;
            if (result.boundaries.empty() || result.boundaries.back() != boundary)
                result.boundaries.push_back(boundary);
        }
        cell = static_cast<size_t>(image.start + image.rows);
    }
    append(ids.size());
    return result;
}
} // namespace cow_vision_dry
