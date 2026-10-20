#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace typical_select {

struct Result {
    std::vector<uint32_t> retained;
    std::vector<uint32_t> ranked;
    size_t support_count = 0;
    size_t kept_count = 0;
    double entropy = 0.0;
    double retained_mass = 0.0;
    bool entropy_computed = false;
};

inline Result retain(const std::vector<float> & scores,
                     const std::vector<uint32_t> & support,
                     double typical_p,
                     size_t min_keep = 1) {
    if (scores.empty() || support.empty())
        throw std::invalid_argument("Typical requires nonempty scores and support");
    if (!std::isfinite(typical_p) || typical_p <= 0.0 || typical_p > 1.0)
        throw std::invalid_argument("Typical p must be finite in (0,1]");
    if (min_keep == 0)
        throw std::invalid_argument("Typical min_keep must be positive");
    for (float score : scores) {
        if (!std::isfinite(score))
            throw std::invalid_argument("Typical scores contain nonfinite values");
    }
    Result result;
    result.support_count = support.size();
    result.ranked = support;
    std::sort(result.ranked.begin(), result.ranked.end());
    for (size_t i = 0; i < result.ranked.size(); ++i) {
        if (static_cast<size_t>(result.ranked[i]) >= scores.size())
            throw std::invalid_argument("Typical support ID is out of range");
        if (i != 0 && result.ranked[i] == result.ranked[i - 1])
            throw std::invalid_argument("Typical support IDs must be unique");
    }

    // p=1 is an exact no-op: validate inputs, but skip softmax and entropy.
    if (typical_p == 1.0) {
        result.retained = result.ranked;
        result.kept_count = result.retained.size();
        return result;
    }

    double maximum = -std::numeric_limits<double>::infinity();
    for (uint32_t token : result.ranked)
        maximum = std::max(maximum, static_cast<double>(scores[token]));
    std::vector<double> weights(result.ranked.size());
    double total = 0.0;
    for (size_t i = 0; i < result.ranked.size(); ++i) {
        weights[i] = std::exp(static_cast<double>(scores[result.ranked[i]]) - maximum);
        total += weights[i];
    }
    if (!std::isfinite(total) || total <= 0.0)
        throw std::invalid_argument("Typical support has no finite positive mass");

    std::vector<double> probabilities(result.ranked.size());
    std::vector<double> log_probabilities(result.ranked.size());
    double entropy = 0.0;
    for (size_t i = 0; i < result.ranked.size(); ++i) {
        probabilities[i] = weights[i] / total;
        log_probabilities[i] = static_cast<double>(scores[result.ranked[i]])
            - maximum - std::log(total);
        if (probabilities[i] > 0.0)
            entropy -= probabilities[i] * log_probabilities[i];
    }
    if (!std::isfinite(entropy))
        throw std::invalid_argument("Typical entropy is not finite");
    result.entropy = entropy;
    result.entropy_computed = true;

    struct Ranked {
        double distance;
        uint32_t token;
        size_t support_index;
    };
    std::vector<Ranked> order;
    order.reserve(result.ranked.size());
    for (size_t i = 0; i < result.ranked.size(); ++i) {
        const double distance = std::abs(-log_probabilities[i] - entropy);
        order.push_back({distance, result.ranked[i], i});
    }
    std::sort(order.begin(), order.end(), [](const Ranked & a, const Ranked & b) {
        if (a.distance != b.distance) return a.distance < b.distance;
        return a.token < b.token;
    });

    double cumulative = 0.0;
    for (const Ranked & item : order) {
        result.retained.push_back(item.token);
        cumulative += probabilities[item.support_index];
        if (result.retained.size() >= min_keep && cumulative > typical_p)
            break;
    }
    std::sort(result.retained.begin(), result.retained.end());
    size_t rank = 0;
    double retained_mass = 0.0;
    for (uint32_t token : result.retained) {
        while (result.ranked[rank] != token) ++rank;
        retained_mass += probabilities[rank];
    }
    result.retained_mass = retained_mass;
    result.kept_count = result.retained.size();
    return result;
}

} // namespace typical_select
