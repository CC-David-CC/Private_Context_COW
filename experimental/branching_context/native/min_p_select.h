#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <vector>

#ifdef MIN_P_SELECT_ENABLE_V4
#include "dynatemp_select.h"
#include "typical_select.h"
#endif
#ifdef MIN_P_SELECT_ENABLE_V5
#include "xtc_select.h"
#endif

namespace min_p_select {

constexpr uint64_t kDraw53Range = uint64_t(1) << 53;

struct Result {
    size_t token = 0;
    size_t legal_count = 0;
    size_t kept_count = 0;
    size_t mask_bytes = 0;
    double total_weight = 0.0;
    size_t dry_penalized_candidates = 0;
    size_t min_p_kept_count = 0;
    size_t top_k_kept_count = 0;
    size_t typical_kept_count = 0;
    double effective_temperature = 0.0;
    bool typical_computed = false;
    bool dynatemp_computed = false;
#ifdef MIN_P_SELECT_ENABLE_V5
    // Private handoff from the unchanged v4 preparation stages into XTC v5.
    // These are not serialized by v4 commands.
    std::vector<float> v5_adjusted_logits;
    std::vector<double> v5_post_top_p_weights;
    std::vector<uint32_t> v5_post_top_p_ids;
#endif
};

struct DryConfig {
    double multiplier = 0.0;
    double base = 1.0;
    uint32_t allowed = 0;
    uint32_t last_n = 0; // zero disables DRY
};

inline bool legal(const uint8_t* mask, size_t token) noexcept {
    return (mask[token / 8] & static_cast<uint8_t>(1u << (token % 8))) != 0;
}

// Pure read-only categorical choice over a fully scored vocabulary.
// Callers pass a branch-stream draw; this function performs no RNG work and
// does not alter logits, masks, or native session state.
inline Result select(const float* logits, size_t vocab, const uint8_t* mask,
                     size_t mask_bytes, double temperature, double min_p,
                     uint64_t draw53) {
    if (vocab == 0 || logits == nullptr || mask == nullptr)
        throw std::invalid_argument("Min-P requires nonempty logits and mask");
    const size_t expected_mask_bytes = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
    if (mask_bytes != expected_mask_bytes)
        throw std::invalid_argument("Min-P mask width mismatch");
    if (!std::isfinite(temperature) || temperature <= 0.0)
        throw std::invalid_argument("Min-P temperature must be positive and finite");
    if (!std::isfinite(min_p) || min_p < 0.0 || min_p > 1.0)
        throw std::invalid_argument("Min-P probability must be finite in [0,1]");
    if (draw53 >= kDraw53Range)
        throw std::invalid_argument("Min-P draw53 is outside the 53-bit range");
    const size_t used_bits = vocab % 8;
    if (used_bits != 0) {
        const uint8_t allowed = static_cast<uint8_t>((1u << used_bits) - 1u);
        if ((mask[mask_bytes - 1] & static_cast<uint8_t>(~allowed)) != 0)
            throw std::invalid_argument("Min-P mask has nonzero padding bits");
    }

    Result result;
    result.mask_bytes = mask_bytes;
    double max_legal = -std::numeric_limits<double>::infinity();
    for (size_t token = 0; token < vocab; ++token) {
        const double score = static_cast<double>(logits[token]);
        if (!std::isfinite(score))
            throw std::invalid_argument("Min-P logits contain a nonfinite score");
        if (legal(mask, token)) {
            ++result.legal_count;
            if (score > max_legal) max_legal = score;
        }
    }
    if (result.legal_count == 0)
        throw std::invalid_argument("Min-P legal support is empty");

    const double cutoff = min_p == 0.0
        ? -std::numeric_limits<double>::infinity()
        : max_legal + std::log(min_p);
    for (size_t token = 0; token < vocab; ++token) {
        if (legal(mask, token) && static_cast<double>(logits[token]) >= cutoff) {
            const double weight = std::exp((static_cast<double>(logits[token]) - max_legal) / temperature);
            if (!std::isfinite(weight) || weight < 0.0)
                throw std::invalid_argument("Min-P produced an invalid weight");
            ++result.kept_count;
            result.total_weight += weight;
        }
    }
    if (result.kept_count == 0 || !std::isfinite(result.total_weight) || result.total_weight <= 0.0)
        throw std::invalid_argument("Min-P retained support has no positive weight");

    const double unit = static_cast<double>(draw53) / static_cast<double>(kDraw53Range);
    const double target = unit * result.total_weight;
    double cumulative = 0.0;
    size_t last_positive = vocab;
    for (size_t token = 0; token < vocab; ++token) {
        if (!legal(mask, token) || static_cast<double>(logits[token]) < cutoff) continue;
        const double weight = std::exp((static_cast<double>(logits[token]) - max_legal) / temperature);
        if (weight <= 0.0) continue;
        last_positive = token;
        cumulative += weight;
        if (cumulative > target) {
            result.token = token;
            return result;
        }
    }
    if (last_positive == vocab)
        throw std::invalid_argument("Min-P has no positive-weight token");
    result.token = last_positive;
    return result;
}

inline bool finite_binary32(float value) noexcept {
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
                  "Min-P selector requires IEEE-754 binary32 logits");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

inline bool finite_binary64(double value) noexcept {
    static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
                  "Min-P selector requires IEEE-754 binary64 parameters");
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

// Apply llama.cpp-style DRY penalties to a private float32 copy. The supplied
// history is already the native session's trailing min(consumed,last_n) IDs;
// callers must validate that binding before invoking this pure helper.
inline std::vector<float> apply_dry_penalties(
        const float * logits, size_t vocab,
        const std::vector<uint32_t> & history,
        const std::vector<uint32_t> & breakers,
        const DryConfig & config,
        size_t * penalized_candidates = nullptr,
        const std::vector<size_t> * sequence_boundaries = nullptr) {
    if (penalized_candidates != nullptr) *penalized_candidates = 0;
    if (vocab == 0 || logits == nullptr)
        throw std::invalid_argument("DRY requires a nonempty logits vector");
    if (!finite_binary64(config.multiplier) || config.multiplier < 0.0 ||
        !finite_binary64(config.base) || config.base < 1.0)
        throw std::invalid_argument("DRY multiplier/base are outside their finite ranges");
    if (config.last_n == 0) {
        if (!history.empty() || !breakers.empty())
            throw std::invalid_argument("disabled DRY requires empty history and breakers");
    } else if (history.size() > config.last_n) {
        throw std::invalid_argument("DRY history exceeds last_n");
    }
    for (uint32_t token : history) {
        if (static_cast<size_t>(token) >= vocab)
            throw std::invalid_argument("DRY history token is outside vocabulary");
    }
    uint32_t previous = 0;
    bool first = true;
    for (uint32_t token : breakers) {
        if (static_cast<size_t>(token) >= vocab)
            throw std::invalid_argument("DRY breaker token is outside vocabulary");
        if (!first && token <= previous)
            throw std::invalid_argument("DRY breakers must be strictly sorted and unique");
        if (std::find(history.begin(), history.end(), token) == history.end())
            throw std::invalid_argument("DRY breaker token is absent from history");
        previous = token;
        first = false;
    }

    const size_t n = history.size();
    if (sequence_boundaries != nullptr) {
        size_t previous = 0;
        bool first = true;
        for (size_t boundary : *sequence_boundaries) {
            if (boundary > n || (!first && boundary <= previous))
                throw std::invalid_argument("DRY sequence boundaries must be sorted unique positions in history");
            previous = boundary; first = false;
        }
    }
    std::vector<float> adjusted(logits, logits + vocab);
    for (float score : adjusted) {
        if (!finite_binary32(score))
            throw std::invalid_argument("DRY logits contain a nonfinite score");
    }
    if (config.last_n == 0 || config.multiplier == 0.0 || history.size() < 2)
        return adjusted;

    // Boundaries are positions between real text tokens, not invented token
    // IDs. Keep the original text-only arithmetic/path when none are supplied.
    std::vector<size_t> run_start;
    if (sequence_boundaries != nullptr && !sequence_boundaries->empty()) {
        if (sequence_boundaries->back() == n) return adjusted;
        run_start.resize(n, 0);
        size_t boundary_index = 0, start = 0;
        for (size_t index = 0; index < n; ++index) {
            if (boundary_index < sequence_boundaries->size() &&
                    (*sequence_boundaries)[boundary_index] == index)
                start = (*sequence_boundaries)[boundary_index++];
            run_start[index] = start;
        }
    }
    size_t repeat_limit = n;
    for (size_t reverse_index = 0; reverse_index < n; ++reverse_index) {
        if (std::binary_search(breakers.begin(), breakers.end(), history[n - 1 - reverse_index])) {
            repeat_limit = reverse_index;
            break;
        }
    }
    if (!run_start.empty()) repeat_limit = std::min(repeat_limit, n - run_start.back());
    if (repeat_limit < config.allowed)
        return adjusted;

    // Z values over the reversed history give each prior position's longest
    // match against the current suffix in linear time, including overlaps.
    std::vector<size_t> z(n, 0);
    z[0] = n;
    size_t left = 0, right = 0;
    for (size_t i = 1; i < n; ++i) {
        if (i <= right)
            z[i] = std::min(right - i + 1, z[i - left]);
        while (i + z[i] < n && history[n - 1 - z[i]] == history[n - 1 - (i + z[i])])
            ++z[i];
        if (z[i] != 0 && i + z[i] - 1 > right) {
            left = i;
            right = i + z[i] - 1;
        }
    }

    std::vector<size_t> max_repeat(vocab, 0);
    std::vector<uint8_t> has_repeat(vocab, 0);
    for (size_t reverse_index = 1; reverse_index < n; ++reverse_index) {
        const size_t original_candidate = n - reverse_index;
        const uint32_t token = history[original_candidate];
        if (!run_start.empty() && original_candidate == run_start[original_candidate]) continue;
        size_t repeat = std::min(z[reverse_index], repeat_limit);
        if (!run_start.empty()) repeat = std::min(repeat, original_candidate - run_start[original_candidate]);
        if (!has_repeat[token] || repeat > max_repeat[token]) {
            max_repeat[token] = repeat;
            has_repeat[token] = 1;
        }
    }

    const double max_float = static_cast<double>(std::numeric_limits<float>::max());
    for (size_t token = 0; token < vocab; ++token) {
        if (!has_repeat[token] || max_repeat[token] < config.allowed ||
            std::binary_search(breakers.begin(), breakers.end(), static_cast<uint32_t>(token)))
            continue;
        const double exponent = static_cast<double>(max_repeat[token] - config.allowed);
        double penalty = config.multiplier * std::pow(config.base, exponent);
        if (!finite_binary64(penalty) || penalty > max_float)
            penalty = max_float;
        double value = static_cast<double>(adjusted[token]) - penalty;
        if (!finite_binary64(value) || value < -max_float)
            value = -max_float;
        if (value > max_float)
            value = max_float;
        const float updated = static_cast<float>(value);
        if (updated != adjusted[token] && penalized_candidates != nullptr)
            ++*penalized_candidates;
        adjusted[token] = updated;
    }
    return adjusted;
}

inline Result select(const float* logits, size_t vocab, const uint8_t* mask,
                     size_t mask_bytes, double temperature, double min_p,
                     uint64_t draw53, size_t top_k, double top_p);

inline Result select_with_dry(const float * logits, size_t vocab,
        const uint8_t * mask, size_t mask_bytes,
        double temperature, double min_p, uint64_t draw53,
        size_t top_k, double top_p,
        const std::vector<uint32_t> & history,
        const std::vector<uint32_t> & breakers,
        const DryConfig & config) {
    size_t penalized_candidates = 0;
    const std::vector<float> adjusted = apply_dry_penalties(logits, vocab, history, breakers,
        config, &penalized_candidates);
    Result result = select(adjusted.data(), vocab, mask, mask_bytes, temperature, min_p,
                           draw53, top_k, top_p);
    result.dry_penalized_candidates = penalized_candidates;
    return result;
}

// Versioned sampler extension. The all-default Top-K/Top-P configuration is
// routed through the original v1 implementation, preserving its arithmetic
// and token choices exactly. Non-default requests apply filters in this order:
// Min-P on raw logits, Top-K on raw logits, temperature, then Top-P.
inline Result select(const float* logits, size_t vocab, const uint8_t* mask,
                     size_t mask_bytes, double temperature, double min_p,
                     uint64_t draw53, size_t top_k, double top_p) {
    if (temperature > 0.0 && top_k == 0 && top_p == 1.0)
        return select(logits, vocab, mask, mask_bytes, temperature, min_p, draw53);

    if (vocab == 0 || logits == nullptr || mask == nullptr)
        throw std::invalid_argument("Min-P requires nonempty logits and mask");
    const size_t required = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
    if (mask_bytes != required)
        throw std::invalid_argument("Min-P mask width mismatch");
    if (!finite_binary64(temperature) || temperature < 0.0)
        throw std::invalid_argument("Min-P temperature must be nonnegative and finite");
    if (!finite_binary64(min_p) || min_p < 0.0 || min_p > 1.0)
        throw std::invalid_argument("Min-P probability must be finite in [0,1]");
    if (!finite_binary64(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("Top-P must be finite in (0,1]");
    if (draw53 >= kDraw53Range)
        throw std::invalid_argument("Min-P draw53 is outside the 53-bit range");
    const size_t used_bits = vocab % 8;
    if (used_bits != 0) {
        const uint8_t allowed = static_cast<uint8_t>((1u << used_bits) - 1u);
        if ((mask[required - 1] & static_cast<uint8_t>(~allowed)) != 0)
            throw std::invalid_argument("Min-P mask has nonzero padding bits");
    }

    Result result;
    result.mask_bytes = required;
    double max_legal = -std::numeric_limits<double>::infinity();
    size_t max_token = vocab;
    for (size_t token = 0; token < vocab; ++token) {
        if (!finite_binary32(logits[token]))
            throw std::invalid_argument("Min-P logits contain a nonfinite score");
        if (legal(mask, token)) {
            ++result.legal_count;
            if (max_token == vocab || static_cast<double>(logits[token]) > max_legal) {
                max_legal = static_cast<double>(logits[token]);
                max_token = token;
            }
        }
    }
    if (result.legal_count == 0)
        throw std::invalid_argument("Min-P legal support is empty");
    if (temperature == 0.0) {
        result.token = max_token;
        result.kept_count = 1;
        result.total_weight = 1.0;
        return result;
    }

    const double cutoff = min_p == 0.0
        ? -std::numeric_limits<double>::infinity()
        : max_legal + std::log(min_p);
    std::vector<size_t> ranked;
    ranked.reserve(result.legal_count);
    for (size_t token = 0; token < vocab; ++token) {
        if (legal(mask, token) && static_cast<double>(logits[token]) >= cutoff)
            ranked.push_back(token);
    }
    if (ranked.empty())
        throw std::invalid_argument("Min-P retained support is empty");

    const auto rank_before = [logits](size_t lhs, size_t rhs) {
        if (logits[lhs] != logits[rhs]) return logits[lhs] > logits[rhs];
        return lhs < rhs;
    };
    std::sort(ranked.begin(), ranked.end(), rank_before);
    if (top_k != 0 && ranked.size() > top_k)
        ranked.resize(top_k);
    if (ranked.empty())
        throw std::invalid_argument("Top-K retained support is empty");

    const double max_selected = static_cast<double>(logits[ranked.front()]);
    std::vector<double> weights(vocab, 0.0);
    double total_weight = 0.0;
    for (size_t token : ranked) {
        const double weight = std::exp((static_cast<double>(logits[token]) - max_selected) / temperature);
        if (!std::isfinite(weight) || weight < 0.0)
            throw std::invalid_argument("Min-P produced an invalid weight");
        weights[token] = weight;
    }
    // Keep the denominator's binary64 accumulation in token-ID order. The
    // Top-P prefix itself remains ranked by descending weight below.
    for (size_t token = 0; token < vocab; ++token)
        total_weight += weights[token];
    if (!std::isfinite(total_weight) || total_weight <= 0.0)
        throw std::invalid_argument("Min-P retained support has no positive weight");

    if (top_p < 1.0) {
        const double target_mass = top_p * total_weight;
        double cumulative_mass = 0.0;
        size_t prefix_count = 0;
        for (size_t token : ranked) {
            cumulative_mass += weights[token];
            ++prefix_count;
            if (cumulative_mass >= target_mass) break;
        }
        prefix_count = prefix_count == 0 ? 1 : prefix_count;
        for (size_t index = prefix_count; index < ranked.size(); ++index)
            weights[ranked[index]] = 0.0;
        ranked.resize(prefix_count);
    }
    result.kept_count = ranked.size();
    total_weight = 0.0;
    for (size_t token = 0; token < vocab; ++token)
        total_weight += weights[token];
    result.total_weight = total_weight;
    if (result.kept_count == 0 || !std::isfinite(total_weight) || total_weight <= 0.0)
        throw std::invalid_argument("Top-P retained support has no positive weight");
    const double unit = static_cast<double>(draw53) / static_cast<double>(kDraw53Range);
    const double target = unit * total_weight;
    double cumulative = 0.0;
    size_t last_positive = vocab;
    for (size_t token = 0; token < vocab; ++token) {
        const double weight = weights[token];
        if (weight <= 0.0) continue;
        last_positive = token;
        cumulative += weight;
        if (cumulative > target) {
            result.token = token;
            return result;
        }
    }
    if (last_positive == vocab)
        throw std::invalid_argument("Min-P has no positive-weight token");
    result.token = last_positive;
    return result;
}

#ifdef MIN_P_SELECT_ENABLE_V4
// Combined v4 path. The older v1/v2/v3 overloads above are intentionally
// unchanged; callers opt into this ordered chain explicitly.
inline Result select_v4(const float * logits, size_t vocab,
        const uint8_t * mask, size_t mask_bytes,
        double temperature, double min_p, size_t top_k, double top_p,
        uint64_t draw53, const std::vector<uint32_t> & history,
        const std::vector<uint32_t> & breakers, const DryConfig & dry,
        double typical_p, double dyna_range, double dyna_exponent
#ifdef MIN_P_SELECT_ENABLE_V5
        , bool capture_v5_handoff = false
#endif
        ) {
    if (vocab == 0 || logits == nullptr || mask == nullptr)
        throw std::invalid_argument("sampler v4 requires nonempty logits and mask");
    const size_t required = vocab / 8 + (vocab % 8 != 0 ? 1 : 0);
    if (mask_bytes != required)
        throw std::invalid_argument("sampler v4 mask width mismatch");
    if (!finite_binary64(temperature) || temperature <= 0.0)
        throw std::invalid_argument("sampler v4 temperature must be positive and finite");
    if (!finite_binary64(min_p) || min_p < 0.0 || min_p > 1.0)
        throw std::invalid_argument("sampler v4 Min-P must be finite in [0,1]");
    if (!finite_binary64(top_p) || top_p <= 0.0 || top_p > 1.0)
        throw std::invalid_argument("sampler v4 Top-P must be finite in (0,1]");
    if (!finite_binary64(typical_p) || typical_p <= 0.0 || typical_p > 1.0)
        throw std::invalid_argument("sampler v4 Typical p must be finite in (0,1]");
    if (!finite_binary64(dyna_range) || dyna_range < 0.0 ||
        !finite_binary64(dyna_exponent) || dyna_exponent <= 0.0 ||
        !finite_binary64(temperature + dyna_range))
        throw std::invalid_argument("sampler v4 DynaTemp parameters are invalid");
    if (draw53 >= kDraw53Range)
        throw std::invalid_argument("sampler v4 draw53 is outside the 53-bit range");
    const size_t used_bits = vocab % 8;
    if (used_bits != 0) {
        const uint8_t allowed = static_cast<uint8_t>((1u << used_bits) - 1u);
        if ((mask[required - 1] & static_cast<uint8_t>(~allowed)) != 0)
            throw std::invalid_argument("sampler v4 mask has nonzero padding bits");
    }

    const bool dry_active = dry.multiplier > 0.0 && dry.last_n > 0;
    if (!dry_active && (!history.empty() || !breakers.empty()))
        throw std::invalid_argument("inactive sampler v4 DRY requires empty history and breakers");
    size_t penalized_candidates = 0;
    const std::vector<float> adjusted = apply_dry_penalties(
        logits, vocab, history, breakers, dry, &penalized_candidates);

    Result result;
    result.mask_bytes = required;
    result.dry_penalized_candidates = penalized_candidates;
    std::vector<uint32_t> legal_ids;
    legal_ids.reserve(vocab);
    for (size_t token = 0; token < vocab; ++token) {
        if (legal(mask, token)) legal_ids.push_back(static_cast<uint32_t>(token));
    }
    if (legal_ids.empty())
        throw std::invalid_argument("sampler v4 legal support is empty");

    double max_legal = -std::numeric_limits<double>::infinity();
    for (uint32_t token : legal_ids)
        max_legal = std::max(max_legal, static_cast<double>(adjusted[token]));
    const double cutoff = min_p == 0.0
        ? -std::numeric_limits<double>::infinity()
        : max_legal + std::log(min_p);
    std::vector<uint32_t> min_p_ids;
    min_p_ids.reserve(legal_ids.size());
    for (uint32_t token : legal_ids) {
        if (static_cast<double>(adjusted[token]) >= cutoff) min_p_ids.push_back(token);
    }
    if (min_p_ids.empty())
        throw std::invalid_argument("sampler v4 Min-P support is empty");
    result.legal_count = legal_ids.size();
    result.min_p_kept_count = min_p_ids.size();

    std::vector<uint32_t> top_k_ids = min_p_ids;
    if (top_k != 0 && top_k_ids.size() > top_k) {
        std::sort(top_k_ids.begin(), top_k_ids.end(), [&adjusted](uint32_t lhs, uint32_t rhs) {
            if (adjusted[lhs] != adjusted[rhs]) return adjusted[lhs] > adjusted[rhs];
            return lhs < rhs;
        });
        top_k_ids.resize(top_k);
        std::sort(top_k_ids.begin(), top_k_ids.end());
    }
    result.top_k_kept_count = top_k_ids.size();

    const typical_select::Result typical = typical_select::retain(
        adjusted, top_k_ids, typical_p, 1);
    result.typical_kept_count = typical.retained.size();
    result.typical_computed = typical_p < 1.0;

    const dynatemp_select::Result dyna = dynatemp_select::effective_temperature(
        adjusted, typical.retained, temperature, dyna_range, dyna_exponent);
    result.effective_temperature = dyna.effective_temperature;
    result.dynatemp_computed = dyna_range > 0.0;

    if (dyna.effective_temperature == 0.0) {
        uint32_t best = typical.retained.front();
        for (uint32_t token : typical.retained) {
            if (adjusted[token] > adjusted[best] ||
                (adjusted[token] == adjusted[best] && token < best)) best = token;
        }
        result.token = best;
        result.kept_count = 1;
        result.total_weight = 1.0;
#ifdef MIN_P_SELECT_ENABLE_V5
        if (capture_v5_handoff) {
            result.v5_adjusted_logits = adjusted;
            result.v5_post_top_p_weights.assign(vocab, 0.0);
            result.v5_post_top_p_weights[best] = 1.0;
            result.v5_post_top_p_ids.assign(1, best);
        }
#endif
        return result;
    }

    double max_selected = -std::numeric_limits<double>::infinity();
    for (uint32_t token : typical.retained)
        max_selected = std::max(max_selected, static_cast<double>(adjusted[token]));
    std::vector<double> weights(vocab, 0.0);
    for (uint32_t token : typical.retained) {
        const double weight = std::exp((static_cast<double>(adjusted[token]) - max_selected) /
                                       dyna.effective_temperature);
        if (!finite_binary64(weight) || weight < 0.0)
            throw std::invalid_argument("sampler v4 produced an invalid Top-P weight");
        weights[token] = weight;
    }
    double total_weight = 0.0;
    for (size_t token = 0; token < vocab; ++token) total_weight += weights[token];
    if (!finite_binary64(total_weight) || total_weight <= 0.0)
        throw std::invalid_argument("sampler v4 categorical support has no positive mass");

    std::vector<uint32_t> top_p_ids = typical.retained;
    if (top_p < 1.0) {
        std::sort(top_p_ids.begin(), top_p_ids.end(), [&adjusted](uint32_t lhs, uint32_t rhs) {
            if (adjusted[lhs] != adjusted[rhs]) return adjusted[lhs] > adjusted[rhs];
            return lhs < rhs;
        });
        const double target_mass = top_p * total_weight;
        double cumulative_mass = 0.0;
        size_t prefix_count = 0;
        for (uint32_t token : top_p_ids) {
            cumulative_mass += weights[token];
            ++prefix_count;
            if (cumulative_mass >= target_mass) break;
        }
        if (prefix_count == 0) prefix_count = 1;
        for (size_t index = prefix_count; index < top_p_ids.size(); ++index)
            weights[top_p_ids[index]] = 0.0;
        top_p_ids.resize(prefix_count);
    }
    std::sort(top_p_ids.begin(), top_p_ids.end());
    result.kept_count = top_p_ids.size();
    total_weight = 0.0;
    for (size_t token = 0; token < vocab; ++token) total_weight += weights[token];
    if (!finite_binary64(total_weight) || total_weight <= 0.0)
        throw std::invalid_argument("sampler v4 retained support has no positive mass");
    result.total_weight = total_weight;

#ifdef MIN_P_SELECT_ENABLE_V5
    if (capture_v5_handoff) {
        result.v5_adjusted_logits = adjusted;
        result.v5_post_top_p_weights = weights;
        result.v5_post_top_p_ids = top_p_ids;
    }
#endif

    const double target = (static_cast<double>(draw53) / static_cast<double>(kDraw53Range)) * total_weight;
    double cumulative = 0.0;
    uint32_t last_positive = vocab;
    for (uint32_t token : top_p_ids) {
        const double weight = weights[token];
        if (weight <= 0.0) continue;
        last_positive = token;
        cumulative += weight;
        if (cumulative > target) {
            result.token = token;
            return result;
        }
    }
    if (last_positive == vocab)
        throw std::invalid_argument("sampler v4 has no positive categorical weight");
    result.token = last_positive;
    return result;
}
#endif // MIN_P_SELECT_ENABLE_V4

#ifdef MIN_P_SELECT_ENABLE_V5
constexpr uint64_t kXtcSecondDrawAbsent = uint64_t(1) << 53;

struct V5Result {
    Result base;
    uint64_t draw53 = 0;
    uint64_t draw53_second = kXtcSecondDrawAbsent;
    size_t draw_count = 0;
    uint32_t activation_draws = 0;
    bool activated = false;
    size_t pre_xtc_count = 0;
    size_t removed_count = 0;
    double xtc_probability = 0.0;
    double xtc_threshold = 0.0;
};

inline V5Result select_v5(const float * logits, size_t vocab,
        const uint8_t * mask, size_t mask_bytes,
        double temperature, double min_p, size_t top_k, double top_p,
        uint64_t draw53, uint64_t draw53_second,
        const std::vector<uint32_t> & history, const std::vector<uint32_t> & breakers,
        const DryConfig & dry, double typical_p, double dyna_range, double dyna_exponent,
        double xtc_probability, double xtc_threshold) {
    if (!finite_binary64(xtc_probability) || xtc_probability < 0.0 || xtc_probability > 1.0)
        throw std::invalid_argument("sampler v5 XTC probability must be finite in [0,1]");
    if (!finite_binary64(xtc_threshold) || xtc_threshold < 0.0 || xtc_threshold > 1.0)
        throw std::invalid_argument("sampler v5 XTC threshold must be finite in [0,1]");
    if (draw53 >= kDraw53Range ||
        (draw53_second >= kDraw53Range && draw53_second != kXtcSecondDrawAbsent))
        throw std::invalid_argument("sampler v5 draw is outside the protocol range");
    if (!finite_binary64(temperature) || temperature <= 0.0)
        throw std::invalid_argument("sampler v5 requires positive base temperature");

    V5Result output;
    output.draw53 = draw53;
    output.draw53_second = draw53_second;
    output.xtc_probability = xtc_probability;
    output.xtc_threshold = xtc_threshold;
    output.base = select_v4(logits, vocab, mask, mask_bytes,
        temperature, min_p, top_k, top_p, draw53, history, breakers, dry,
        typical_p, dyna_range, dyna_exponent, true);

    const xtc_select::PreparedResult xtc = xtc_select::filter_prepared_weights(
        output.base.v5_adjusted_logits,
        std::vector<std::size_t>(output.base.v5_post_top_p_ids.begin(),
                                 output.base.v5_post_top_p_ids.end()),
        output.base.v5_post_top_p_weights,
        output.base.effective_temperature,
        xtc_probability, xtc_threshold, draw53);
    output.activation_draws = xtc.activation_draws;
    output.activated = xtc.activated;
    output.pre_xtc_count = xtc.pre_xtc_count;
    output.removed_count = xtc.removed_count;
    output.base.kept_count = xtc.kept_count;
    output.draw_count = 1 + output.activation_draws;
    if (output.activation_draws != 0 && draw53_second == kXtcSecondDrawAbsent)
        throw std::invalid_argument("sampler v5 requires a second draw when XTC is eligible");

    if (output.base.effective_temperature == 0.0) {
        // v4's positive-base-temperature DynaTemp boundary is deterministic.
        // It still consumes the generation's first host draw, but no XTC draw.
        output.base.kept_count = 1;
        output.base.total_weight = 1.0;
        return output;
    }

    std::vector<double> weights = output.base.v5_post_top_p_weights;
    if (!xtc.removed_ids.empty()) {
        for (std::size_t token : xtc.removed_ids) weights[token] = 0.0;
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t token : xtc.retained_ids)
            maximum = std::max(maximum,
                static_cast<double>(output.base.v5_adjusted_logits[token]));
        for (std::size_t token : xtc.retained_ids) {
            const double weight = std::exp(
                (static_cast<double>(output.base.v5_adjusted_logits[token]) - maximum) /
                output.base.effective_temperature);
            if (!finite_binary64(weight) || weight < 0.0)
                throw std::invalid_argument("sampler v5 recentered weight is invalid");
            weights[token] = weight;
        }
        output.base.total_weight = 0.0;
        for (std::size_t token = 0; token < vocab; ++token)
            output.base.total_weight += weights[token];
        if (!finite_binary64(output.base.total_weight) || output.base.total_weight <= 0.0)
            throw std::invalid_argument("sampler v5 retained support has no positive weight");
    }

    const uint64_t categorical_draw = output.activation_draws == 0 ? draw53 : draw53_second;
    const double target = (static_cast<double>(categorical_draw) /
                           static_cast<double>(kDraw53Range)) * output.base.total_weight;
    double cumulative = 0.0;
    std::size_t last_positive = vocab;
    for (std::size_t token : xtc.retained_ids) {
        const double weight = weights[token];
        if (weight <= 0.0) continue;
        last_positive = token;
        cumulative += weight;
        if (cumulative > target) {
            output.base.token = token;
            return output;
        }
    }
    if (last_positive == vocab)
        throw std::invalid_argument("sampler v5 has no positive categorical weight");
    output.base.token = last_positive;
    return output;
}
#endif // MIN_P_SELECT_ENABLE_V5

} // namespace min_p_select
