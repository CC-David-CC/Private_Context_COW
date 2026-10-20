#pragma once

#include <cstddef>
#include <stdexcept>

namespace strata_vision {

// SVE1 represents exactly one rectangular image. Never silently serialize the
// last chunk of a tiled/multi-chunk encoder result as if it were the whole input.
template<class IsImage>
std::size_t single_image_chunk_index(std::size_t count, IsImage is_image) {
    std::size_t selected = count;
    for (std::size_t i = 0; i < count; ++i) {
        if (!is_image(i)) continue;
        if (selected != count) throw std::invalid_argument("multiple image chunks are unsupported");
        selected = i;
    }
    if (selected == count) throw std::invalid_argument("no image chunk");
    return selected;
}

} // namespace strata_vision
