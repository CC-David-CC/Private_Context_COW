#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dynatemp_select {

struct Result {
    double effective_temperature = 0.0;
    double normalized_entropy = 0.0;
    double entropy = 0.0;
    double maximum_entropy = 0.0;
    size_t support_count = 0;
    bool entropy_computed = false;
};

inline Result effective_temperature(const std::vector<float> & scores,
                                    const std::vector<uint32_t> & support,
                                    double temperature,
                                    double temp_range,
                                    double exponent) {
    if (scores.empty() || support.empty())
        throw std::invalid_argument("DynaTemp requires nonempty scores and support");
    if (!std::isfinite(temperature) || temperature < 0.0)
        throw std::invalid_argument("DynaTemp base temperature must be finite and nonnegative");
    if (!std::isfinite(temp_range) || temp_range < 0.0)
        throw std::invalid_argument("DynaTemp range must be finite and nonnegative");
    if (!std::isfinite(exponent) || exponent <= 0.0)
        throw std::invalid_argument("DynaTemp exponent must be finite and positive");
    for (float score : scores) {
        if (!std::isfinite(score))
            throw std::invalid_argument("DynaTemp scores contain nonfinite values");
    }
    Result result;
    result.support_count = support.size();
    std::vector<uint32_t> ids = support;
    std::sort(ids.begin(), ids.end());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (static_cast<size_t>(ids[i]) >= scores.size())
            throw std::invalid_argument("DynaTemp support ID is out of range");
        if (i != 0 && ids[i] == ids[i - 1])
            throw std::invalid_argument("DynaTemp support IDs must be unique");
    }
    const double low = std::max(0.0, temperature - temp_range);
    const double high = temperature + temp_range;
    if (!std::isfinite(high))
        throw std::invalid_argument("DynaTemp range overflows finite values");
    if (temperature == 0.0) {
        result.effective_temperature = 0.0;
        return result;
    }
    if (temp_range == 0.0) {
        result.effective_temperature = temperature;
        return result;
    }

    double normalized = 0.0;
    if (ids.size() > 1) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (uint32_t id : ids) maximum = std::max(maximum, static_cast<double>(scores[id]));
        std::vector<double> weights(ids.size());
        double total = 0.0;
        for (size_t i = 0; i < ids.size(); ++i) {
            weights[i] = std::exp(static_cast<double>(scores[ids[i]]) - maximum);
            total += weights[i];
        }
        if (!std::isfinite(total) || total <= 0.0)
            throw std::invalid_argument("DynaTemp support has no finite positive mass");
        double entropy = 0.0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const double prob = weights[i] / total;
            if (prob > 0.0) entropy -= prob * std::log(prob);
        }
        const double maximum_entropy = std::log(static_cast<double>(ids.size()));
        if (!std::isfinite(entropy) || !std::isfinite(maximum_entropy) || maximum_entropy <= 0.0)
            throw std::invalid_argument("DynaTemp entropy is invalid");
        normalized = std::min(1.0, std::max(0.0, entropy / maximum_entropy));
        result.entropy = entropy;
        result.maximum_entropy = maximum_entropy;
        result.entropy_computed = true;
    } else {
        result.entropy_computed = true;
    }
    result.normalized_entropy = normalized;
    result.effective_temperature = low + (high - low) * std::pow(normalized, exponent);
    if (!std::isfinite(result.effective_temperature) ||
        result.effective_temperature < low || result.effective_temperature > high)
        throw std::invalid_argument("DynaTemp result is outside its finite range");
    return result;
}

} // namespace dynatemp_select
