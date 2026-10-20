#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xtc_select {

struct Result {
    std::vector<std::size_t> support_ids;
    std::vector<std::size_t> retained_ids;
    std::vector<std::size_t> removed_ids;
    std::vector<std::size_t> eligible_ids;
    std::uint32_t activation_draws = 0;
    bool activated = false;
    std::size_t support_count = 0;
    double total_weight = 0.0;
};

inline Result filter_support(const std::vector<double>& scores,
                             const std::vector<std::size_t>& support,
                             double probability,
                             double threshold,
                             std::uint64_t draw53) {
    constexpr std::uint64_t draw_range = std::uint64_t(1) << 53;
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("XTC probability must be finite and in [0, 1]");
    }
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        throw std::invalid_argument("XTC threshold must be finite and in [0, 1]");
    }
    if (draw53 >= draw_range) {
        throw std::invalid_argument("draw53 is outside [0, 2^53)");
    }
    if (scores.empty() || support.empty()) {
        throw std::invalid_argument("scores and support must be nonempty");
    }
    for (double score : scores) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("scores must be finite");
        }
    }

    Result result;
    result.support_ids = support;
    std::sort(result.support_ids.begin(), result.support_ids.end());
    if (std::adjacent_find(result.support_ids.begin(), result.support_ids.end()) !=
        result.support_ids.end()) {
        throw std::invalid_argument("support contains duplicate token IDs");
    }
    for (std::size_t token_id : result.support_ids) {
        if (token_id >= scores.size()) {
            throw std::invalid_argument("support contains an invalid token ID");
        }
    }
    result.retained_ids = result.support_ids;
    result.support_count = result.support_ids.size();
    if (probability == 0.0 || threshold > 0.5 || result.support_count < 2) {
        return result;
    }

    result.activation_draws = 1;
    const double chance = static_cast<double>(draw53) / static_cast<double>(draw_range);
    if (chance > probability) {
        return result;
    }
    result.activated = true;

    double maximum = scores[result.support_ids.front()];
    for (std::size_t token_id : result.support_ids) {
        maximum = std::max(maximum, scores[token_id]);
    }
    double total_weight = 0.0;
    for (std::size_t token_id : result.support_ids) {
        const double weight = std::exp(scores[token_id] - maximum);
        total_weight += weight;
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::invalid_argument("support has no finite positive softmax weight");
    }
    result.total_weight = total_weight;

    std::vector<std::size_t> ranked_ids = result.support_ids;
    std::sort(ranked_ids.begin(), ranked_ids.end(),
              [&scores](std::size_t left, std::size_t right) {
                  if (scores[left] != scores[right]) {
                      return scores[left] > scores[right];
                  }
                  return left < right;
              });
    std::vector<std::size_t> eligible_ranked;
    for (std::size_t token_id : ranked_ids) {
        const double unnormalized = std::exp(scores[token_id] - maximum);
        const double normalized = unnormalized / total_weight;
        if (normalized < threshold) {
            break;
        }
        eligible_ranked.push_back(token_id);
    }
    result.eligible_ids = eligible_ranked;
    std::sort(result.eligible_ids.begin(), result.eligible_ids.end());
    if (eligible_ranked.size() < 2) {
        return result;
    }

    result.removed_ids.assign(eligible_ranked.begin(), eligible_ranked.end() - 1);
    std::sort(result.removed_ids.begin(), result.removed_ids.end());
    result.retained_ids.clear();
    std::set_difference(result.support_ids.begin(), result.support_ids.end(),
                        result.removed_ids.begin(), result.removed_ids.end(),
                        std::back_inserter(result.retained_ids));
    return result;
}

// Prepared-weight entry for sampler v5. `weights` are the exact unnormalized
// binary64 weights produced after Top-P by the preceding sampler stages.
// Eligibility probabilities are normalized from those weights in token-ID
// order; XTC never reconstructs or rescales logits here.
struct PreparedResult {
    std::vector<std::size_t> retained_ids;
    std::vector<std::size_t> eligible_ids;
    std::vector<std::size_t> removed_ids;
    std::size_t pre_xtc_count = 0;
    std::size_t kept_count = 0;
    std::size_t removed_count = 0;
    std::uint32_t activation_draws = 0;
    bool activated = false;
    double total_weight = 0.0;
};

inline PreparedResult filter_prepared_weights(
        const std::vector<float>& raw_scores,
        const std::vector<std::size_t>& support,
        const std::vector<double>& weights,
        double effective_temperature,
        double probability,
        double threshold,
        std::uint64_t draw53) {
    constexpr std::uint64_t draw_range = std::uint64_t(1) << 53;
    if (!std::isfinite(effective_temperature) || effective_temperature < 0.0)
        throw std::invalid_argument("effective temperature must be finite and nonnegative");
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
        throw std::invalid_argument("XTC probability must be finite and in [0, 1]");
    if (!std::isfinite(threshold) || threshold < 0.0 || threshold > 1.0)
        throw std::invalid_argument("XTC threshold must be finite and in [0, 1]");
    if (draw53 >= draw_range)
        throw std::invalid_argument("XTC activation draw is outside [0, 2^53)");
    if (raw_scores.empty() || raw_scores.size() != weights.size() || support.empty())
        throw std::invalid_argument("XTC prepared score, weight, and support extents are invalid");

    PreparedResult result;
    result.pre_xtc_count = support.size();
    result.retained_ids = support;
    result.kept_count = support.size();
    std::size_t previous = 0;
    bool first = true;
    for (std::size_t token : support) {
        if (token >= raw_scores.size() || (!first && token <= previous))
            throw std::invalid_argument("XTC prepared support must be sorted, unique, and in range");
        if (!std::isfinite(raw_scores[token]))
            throw std::invalid_argument("XTC prepared score is nonfinite");
        if (!std::isfinite(weights[token]) || weights[token] < 0.0)
            throw std::invalid_argument("XTC prepared weight is invalid");
        previous = token;
        first = false;
    }
    double total = 0.0;
    for (std::size_t token : support) total += weights[token];
    if (!std::isfinite(total) || total <= 0.0)
        throw std::invalid_argument("XTC prepared support has no positive finite weight");
    result.total_weight = total;

    if (effective_temperature == 0.0 || probability == 0.0 || threshold > 0.5 || support.size() < 2)
        return result;

    result.activation_draws = 1;
    const double chance = static_cast<double>(draw53) / static_cast<double>(draw_range);
    if (chance > probability) return result;
    result.activated = true;

    std::vector<std::size_t> ranked = support;
    std::sort(ranked.begin(), ranked.end(), [&raw_scores](std::size_t left, std::size_t right) {
        if (raw_scores[left] != raw_scores[right]) return raw_scores[left] > raw_scores[right];
        return left < right;
    });
    std::vector<std::size_t> eligible_ranked;
    for (std::size_t token : ranked) {
        const double normalized = weights[token] / total;
        if (normalized < threshold) break;
        eligible_ranked.push_back(token);
    }
    result.eligible_ids = eligible_ranked;
    std::sort(result.eligible_ids.begin(), result.eligible_ids.end());
    if (eligible_ranked.size() < 2) return result;

    result.removed_ids.assign(eligible_ranked.begin(), eligible_ranked.end() - 1);
    std::sort(result.removed_ids.begin(), result.removed_ids.end());
    result.retained_ids.clear();
    std::set_difference(support.begin(), support.end(),
                        result.removed_ids.begin(), result.removed_ids.end(),
                        std::back_inserter(result.retained_ids));
    result.kept_count = result.retained_ids.size();
    result.removed_count = result.removed_ids.size();
    return result;
}

}  // namespace xtc_select
