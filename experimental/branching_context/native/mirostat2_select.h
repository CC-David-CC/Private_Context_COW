#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mirostat2_select {

struct Result {
    std::uint32_t token = 0;
    double next_mu = 0.0;
    double selected_log_probability = 0.0;
    double selected_probability = 0.0;
    double surprise_bits = 0.0;
    double legal_total_weight = 0.0;
    double total_weight = 0.0;
    std::size_t retention_candidate_visits = 0;
    std::vector<std::uint32_t> legal_ids;
    std::vector<std::uint32_t> kept_ids;
};

inline Result choose(const std::vector<float>& logits,
                     const std::vector<std::uint32_t>& support,
                     double temperature,
                     std::uint64_t draw53,
                     double mu,
                     double tau,
                     double eta) {
    constexpr std::uint64_t draw_range = std::uint64_t{1} << 53;
    if (!std::isfinite(temperature) || temperature <= 0.0 ||
        !std::isfinite(mu) || !std::isfinite(tau) || tau <= 0.0 ||
        !std::isfinite(eta) || eta < 0.0 || draw53 >= draw_range) {
        throw std::invalid_argument("invalid Mirostat parameters or draw");
    }
    if (logits.empty() || support.empty()) {
        throw std::invalid_argument("logits and support must be nonempty");
    }
    for (float score : logits) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("logits must be finite");
        }
    }

    Result result;
    result.legal_ids = support;
    if (!std::is_sorted(result.legal_ids.begin(), result.legal_ids.end())) {
        std::sort(result.legal_ids.begin(), result.legal_ids.end());
    }
    if (std::adjacent_find(result.legal_ids.begin(), result.legal_ids.end()) !=
        result.legal_ids.end()) {
        throw std::invalid_argument("support IDs must be unique");
    }
    for (std::uint32_t id : result.legal_ids) {
        if (id >= logits.size()) {
            throw std::invalid_argument("support ID is outside vocabulary");
        }
    }

    double maximum = -INFINITY;
    for (std::uint32_t id : result.legal_ids) {
        maximum = std::max(maximum, static_cast<double>(logits[id]));
    }
    std::vector<double> shifted(logits.size(), -INFINITY);
    for (std::uint32_t id : result.legal_ids) {
        const double delta = static_cast<double>(logits[id]) - maximum;
        const double score = delta / temperature;
        if (std::isnan(score) || score > 0.0) {
            throw std::invalid_argument("shifted legal logit is invalid");
        }
        shifted[id] = score;
        result.legal_total_weight += std::exp(score);
    }
    if (!std::isfinite(result.legal_total_weight) || result.legal_total_weight <= 0.0) {
        throw std::invalid_argument("legal support has no finite weight");
    }
    const double legal_log_weight = std::log(result.legal_total_weight);

    // The surprisal cutoff selects a set, regardless of probability order.
    // Scan canonical token order so normalization and the categorical CDF
    // keep their original accumulation order without sorting the vocabulary.
    std::uint32_t best = result.legal_ids.front();
    for (std::uint32_t id : result.legal_ids) {
        ++result.retention_candidate_visits;
        if (shifted[id] > shifted[best]) best = id;
        const double surprise = -(shifted[id] - legal_log_weight) / std::log(2.0);
        if (surprise <= mu) result.kept_ids.push_back(id);
    }
    if (result.kept_ids.empty()) result.kept_ids.push_back(best);

    double kept_maximum = -INFINITY;
    for (std::uint32_t id : result.kept_ids) {
        kept_maximum = std::max(kept_maximum, shifted[id]);
    }
    std::vector<double> weights(logits.size(), 0.0);
    for (std::uint32_t id : result.kept_ids) {
        weights[id] = std::exp(shifted[id] - kept_maximum);
        result.total_weight += weights[id];
    }
    if (!std::isfinite(result.total_weight) || result.total_weight <= 0.0) {
        throw std::invalid_argument("retained support has no finite weight");
    }

    const double target = (static_cast<double>(draw53) / static_cast<double>(draw_range)) *
                          result.total_weight;
    double cumulative = 0.0;
    bool selected = false;
    for (std::uint32_t id : result.kept_ids) {
        if (weights[id] <= 0.0) continue;
        cumulative += weights[id];
        if (cumulative > target) {
            result.token = id;
            selected = true;
            break;
        }
    }
    if (!selected) {
        for (auto it = result.kept_ids.rbegin(); it != result.kept_ids.rend(); ++it) {
            if (weights[*it] > 0.0) {
                result.token = *it;
                selected = true;
                break;
            }
        }
    }
    if (!selected) throw std::invalid_argument("no selectable positive-weight token");

    result.selected_log_probability = shifted[result.token] - kept_maximum -
                                     std::log(result.total_weight);
    result.selected_probability = std::exp(result.selected_log_probability);
    result.surprise_bits = -result.selected_log_probability / std::log(2.0);
    // Force the two binary64 controller-feedback operations to round
    // separately even when the surrounding translation unit permits FMA.
    volatile double error = result.surprise_bits - tau;
    volatile double update = eta * error;
    result.next_mu = mu - update;
    if (!std::isfinite(result.selected_log_probability) ||
        result.selected_log_probability > 0.0 ||
        !std::isfinite(result.selected_probability) || result.selected_probability < 0.0 ||
        result.selected_probability > 1.0 ||
        !std::isfinite(result.surprise_bits) || !std::isfinite(update) ||
        !std::isfinite(result.next_mu)) {
        throw std::invalid_argument("Mirostat feedback update is not finite");
    }
    return result;
}

}  // namespace mirostat2_select
