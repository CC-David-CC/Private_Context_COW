#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace grammar_select {

struct Result {
    std::size_t token;
    std::size_t logits_count;
    std::size_t mask_bytes;
};

// Select the lowest-ID legal token with the greatest finite raw score.
// Every score is checked, including masked-out positions.
inline Result select(const float* scores, std::size_t vocab,
                     const std::uint8_t* mask, std::size_t mask_bytes,
                     std::size_t consumed, std::size_t expected_consumed) {
    if (vocab == 0 || scores == nullptr) {
        throw std::invalid_argument("empty logits");
    }
    if (consumed == 0 || expected_consumed == 0 || consumed != expected_consumed) {
        throw std::invalid_argument("consumed count mismatch");
    }
    const std::size_t required = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
    if (mask == nullptr || mask_bytes != required) {
        throw std::invalid_argument("mask width mismatch");
    }
    const unsigned used = static_cast<unsigned>(vocab % 8);
    if (used != 0) {
        const auto allowed = static_cast<std::uint8_t>((1u << used) - 1u);
        if ((mask[mask_bytes - 1] & static_cast<std::uint8_t>(~allowed)) != 0) {
            throw std::invalid_argument("nonzero mask padding bits");
        }
    }

    bool found = false;
    std::size_t best = 0;
    float best_score = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        const float value = scores[i];
        static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
                      "grammar selector requires IEEE-754 binary32 logits");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            throw std::invalid_argument("nonfinite logits");
        }
        const bool legal = (mask[i / 8] & static_cast<std::uint8_t>(1u << (i % 8))) != 0;
        if (legal && (!found || value > best_score)) {
            found = true;
            best = i;
            best_score = value;
        }
    }
    if (!found) {
        throw std::invalid_argument("empty grammar support");
    }
    return {best, vocab, required};
}

}  // namespace grammar_select
