#ifndef CONTEXT_TIME_TRAVEL_ADAPTIVE_P_SELECT_H
#define CONTEXT_TIME_TRAVEL_ADAPTIVE_P_SELECT_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace adaptive_p_select {

struct State {
    double weighted_sum;
    double total_weight;
};

struct Proposal {
    std::uint64_t token;
    double original_probability;
    std::optional<double> adapted_target;
    double next_weighted_sum;
    double next_total_weight;
    double source_weighted_sum;
    double source_total_weight;
    bool feedback_enabled;
};

struct PreparedProposal {
    std::uint64_t token;
    double original_probability;
    std::optional<double> adapted_target;
    double next_weighted_sum;
    double next_total_weight;
    double source_weighted_sum;
    double source_total_weight;
    double total_weight;
    std::size_t legal_count;
    bool feedback_enabled;
};

inline constexpr std::uint64_t kDrawRange = std::uint64_t{1} << 53;
inline constexpr double kDistributionWidth = 0.3;
inline constexpr double kPeakLogitValue = 5.0;
inline constexpr double kSharpness = 10.0;

inline void validate_policy(double target, double decay) {
    if (!std::isfinite(target) || target > 1.0) {
        throw std::invalid_argument("adaptive target must be finite and at most 1");
    }
    if (!std::isfinite(decay) || decay < 0.0 || decay > 0.99) {
        throw std::invalid_argument("adaptive decay must be finite and in [0, 0.99]");
    }
}

inline State validate_state(const State & state) {
    if (!std::isfinite(state.weighted_sum) || !std::isfinite(state.total_weight)
            || state.total_weight <= 0.0) {
        throw std::invalid_argument("Adaptive-P state must be finite with positive weight");
    }
    return state;
}

inline State initial_state(double target, double decay) {
    validate_policy(target, decay);
    const double denominator = 1.0 - decay;
    const State state{target / denominator, 1.0 / denominator};
    return validate_state(state);
}

// The v7 command accepts the same disabled-DRY convention as v4-v6: when
// multiplier is zero or last_n is zero, callers must provide empty tails.
inline std::size_t expected_dry_history_count(std::uint64_t consumed,
        double dry_multiplier, std::uint64_t dry_last_n) {
    if (!std::isfinite(dry_multiplier) || dry_multiplier < 0.0 ||
        dry_last_n > UINT32_MAX)
        throw std::invalid_argument("Adaptive-P DRY extent arguments are invalid");
    if (dry_multiplier == 0.0 || dry_last_n == 0) return 0;
    return static_cast<std::size_t>(std::min(consumed, dry_last_n));
}

// Preserve the reference's binary64 operation order: (sharpness * distance)
// * distance. The volatile intermediates also prevent contraction/reassociation.
inline double transform_distance(double distance) {
    volatile double scaled_distance = kSharpness * distance;
    volatile double scaled_square = static_cast<double>(scaled_distance) * distance;
    volatile double denominator = 1.0 + distance;
    volatile double penalty = static_cast<double>(scaled_square) /
        static_cast<double>(denominator);
    volatile double transformed = kPeakLogitValue - static_cast<double>(penalty);
    return static_cast<double>(transformed);
}

inline std::vector<double> softmax(const std::vector<double> & scores) {
    if (scores.empty()) {
        throw std::invalid_argument("Adaptive-P distribution cannot be empty");
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    std::vector<double> weights;
    weights.reserve(scores.size());
    double total = 0.0;
    for (const double score : scores) {
        const double weight = std::exp(score - maximum);
        weights.push_back(weight);
        total += weight;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::invalid_argument("Adaptive-P distribution has no finite positive mass");
    }
    for (double & weight : weights) {
        weight /= total;
    }
    return weights;
}

inline std::uint64_t draw(const std::vector<double> & weights, std::uint64_t draw53) {
    if (draw53 >= kDrawRange || weights.empty()) {
        throw std::invalid_argument("draw must be a u53 value for nonempty weights");
    }
    double total = 0.0;
    for (const double weight : weights) {
        total += weight;
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::invalid_argument("Adaptive-P distribution has no finite positive mass");
    }
    const double threshold = (static_cast<double>(draw53) / static_cast<double>(kDrawRange)) * total;
    double cumulative = 0.0;
    std::uint64_t last_positive = 0;
    bool has_positive = false;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        const double weight = weights[index];
        if (weight <= 0.0) {
            continue;
        }
        has_positive = true;
        last_positive = static_cast<std::uint64_t>(index);
        cumulative += weight;
        if (cumulative > threshold) {
            return static_cast<std::uint64_t>(index);
        }
    }
    if (!has_positive) {
        throw std::invalid_argument("Adaptive-P distribution has no positive categorical weight");
    }
    return last_positive;
}

// Select from the post-Top-P v4 handoff. `prepared_weights` is a full-vocab
// vector, and `prepared_ids` is the strictly ascending support. This helper
// deliberately does not reconstruct or re-softmax the original logits.
inline PreparedProposal select_prepared(
        const std::vector<double> & prepared_weights,
        const std::vector<std::uint32_t> & prepared_ids,
        const State & input_state,
        double target,
        double decay,
        std::uint64_t draw53) {
    validate_policy(target, decay);
    const State state = validate_state(input_state);
    if (draw53 >= kDrawRange || prepared_weights.empty() || prepared_ids.empty())
        throw std::invalid_argument("Adaptive-P prepared support or draw is invalid");
    for (double weight : prepared_weights) {
        if (!std::isfinite(weight) || weight < 0.0)
            throw std::invalid_argument("Adaptive-P prepared weights must be finite and nonnegative");
    }
    std::vector<bool> selected(prepared_weights.size(), false);
    std::uint32_t previous = 0;
    bool first = true;
    double original_total = 0.0;
    for (std::uint32_t token : prepared_ids) {
        if (token >= prepared_weights.size() || (!first && token <= previous))
            throw std::invalid_argument("Adaptive-P prepared IDs must be ascending, unique, and in range");
        first = false;
        previous = token;
        selected[token] = true;
        original_total += prepared_weights[token];
    }
    for (std::size_t token = 0; token < prepared_weights.size(); ++token) {
        if (!selected[token] && prepared_weights[token] != 0.0)
            throw std::invalid_argument("Adaptive-P prepared weight exists outside the support");
    }
    if (!std::isfinite(original_total) || original_total <= 0.0)
        throw std::invalid_argument("Adaptive-P prepared support has no positive mass");

    std::vector<double> original;
    original.reserve(prepared_ids.size());
    for (std::uint32_t token : prepared_ids)
        original.push_back(prepared_weights[token] / original_total);

    const bool feedback_enabled = target >= 0.0;
    std::optional<double> adapted_target;
    std::vector<double> selection_weights = original;
    if (feedback_enabled) {
        const double mean_probability = state.weighted_sum / state.total_weight;
        volatile double doubled_target = 2.0 * target;
        volatile double adapted_delta = doubled_target - mean_probability;
        const double adapted = std::min(1.0, std::max(0.0, static_cast<double>(adapted_delta)));
        adapted_target = adapted;
        std::vector<double> transformed;
        transformed.reserve(original.size());
        for (double probability : original) {
            const double distance = std::abs((probability - adapted) / kDistributionWidth);
            transformed.push_back(transform_distance(distance));
        }
        const double maximum = *std::max_element(transformed.begin(), transformed.end());
        selection_weights.clear();
        selection_weights.reserve(transformed.size());
        double transformed_total = 0.0;
        for (double value : transformed) {
            const double weight = std::exp(value - maximum);
            selection_weights.push_back(weight);
            transformed_total += weight;
        }
        if (!std::isfinite(transformed_total) || transformed_total <= 0.0)
            throw std::invalid_argument("Adaptive-P transformed support has no positive mass");
        for (double & weight : selection_weights) weight /= transformed_total;
    }

    double sampling_total = 0.0;
    for (double weight : selection_weights) sampling_total += weight;
    if (!std::isfinite(sampling_total) || sampling_total <= 0.0)
        throw std::invalid_argument("Adaptive-P categorical support has no positive mass");
    const double threshold = (static_cast<double>(draw53) / static_cast<double>(kDrawRange)) * sampling_total;
    double cumulative = 0.0;
    std::size_t selected_index = prepared_ids.size();
    std::size_t last_positive = prepared_ids.size();
    for (std::size_t index = 0; index < selection_weights.size(); ++index) {
        if (selection_weights[index] <= 0.0) continue;
        last_positive = index;
        cumulative += selection_weights[index];
        if (cumulative > threshold) { selected_index = index; break; }
    }
    if (selected_index == prepared_ids.size()) selected_index = last_positive;
    if (selected_index == prepared_ids.size())
        throw std::invalid_argument("Adaptive-P categorical support has no positive weight");

    const double original_probability = original[selected_index];
    double next_weighted_sum = state.weighted_sum;
    double next_total_weight = state.total_weight;
    if (feedback_enabled) {
        volatile double decayed_sum = decay * state.weighted_sum;
        volatile double decayed_total = decay * state.total_weight;
        next_weighted_sum = original_probability + static_cast<double>(decayed_sum);
        next_total_weight = 1.0 + static_cast<double>(decayed_total);
        if (!std::isfinite(next_weighted_sum) || !std::isfinite(next_total_weight) || next_total_weight <= 0.0)
            throw std::invalid_argument("Adaptive-P updated state must be finite");
    }
    return PreparedProposal{prepared_ids[selected_index], original_probability, adapted_target,
        next_weighted_sum, next_total_weight, state.weighted_sum, state.total_weight,
        sampling_total, prepared_ids.size(), feedback_enabled};
}

// `legal_ids` must be strictly ascending. Every sum is accumulated in this
// token-ID order, matching the deterministic binary64 reference contract.
inline Proposal select(const std::vector<double> & logits,
                       const std::vector<std::uint64_t> & legal_ids,
                       const State & input_state,
                       double target,
                       double decay,
                       std::uint64_t draw53) {
    validate_policy(target, decay);
    const State state = validate_state(input_state);
    if (draw53 >= kDrawRange) {
        throw std::invalid_argument("draw53 must be in [0, 2^53)");
    }
    if (logits.empty() || legal_ids.empty()) {
        throw std::invalid_argument("logits and legal support must be nonempty");
    }
    for (const double score : logits) {
        if (!std::isfinite(score)) {
            throw std::invalid_argument("all logits must be finite");
        }
    }
    std::uint64_t previous = 0;
    bool first = true;
    std::vector<double> original_scores;
    original_scores.reserve(legal_ids.size());
    for (const std::uint64_t token : legal_ids) {
        if (token >= logits.size() || (!first && token <= previous)) {
            throw std::invalid_argument("legal IDs must be unique, ascending, and in vocabulary");
        }
        first = false;
        previous = token;
        original_scores.push_back(logits[static_cast<std::size_t>(token)]);
    }

    const std::vector<double> original = softmax(original_scores);
    const bool feedback_enabled = target >= 0.0;
    std::optional<double> adapted_target;
    std::vector<double> selection_weights = original;
    if (feedback_enabled) {
        const double mean_probability = state.weighted_sum / state.total_weight;
        const double adapted = std::min(1.0, std::max(0.0, 2.0 * target - mean_probability));
        adapted_target = adapted;
        std::vector<double> transformed;
        transformed.reserve(original.size());
        for (const double probability : original) {
            const double distance = std::abs((probability - adapted) / kDistributionWidth);
            transformed.push_back(kPeakLogitValue - kSharpness * distance * distance / (1.0 + distance));
        }
        selection_weights = softmax(transformed);
    }

    const std::uint64_t selected_index = draw(selection_weights, draw53);
    const double selected_original_probability = original[static_cast<std::size_t>(selected_index)];
    double next_weighted_sum = state.weighted_sum;
    double next_total_weight = state.total_weight;
    if (feedback_enabled) {
        next_weighted_sum = selected_original_probability + decay * state.weighted_sum;
        next_total_weight = 1.0 + decay * state.total_weight;
        if (!std::isfinite(next_weighted_sum) || !std::isfinite(next_total_weight)
                || next_total_weight <= 0.0) {
            throw std::invalid_argument("updated Adaptive-P state must be finite");
        }
    }
    return Proposal{
        legal_ids[static_cast<std::size_t>(selected_index)],
        selected_original_probability,
        adapted_target,
        next_weighted_sum,
        next_total_weight,
        state.weighted_sum,
        state.total_weight,
        feedback_enabled,
    };
}

inline State accept(const State & input_state, const Proposal & proposal,
                    std::uint64_t consumed_token) {
    const State state = validate_state(input_state);
    if (consumed_token != proposal.token) {
        throw std::invalid_argument("consumed token does not match Adaptive-P proposal");
    }
    if (state.weighted_sum != proposal.source_weighted_sum
            || state.total_weight != proposal.source_total_weight) {
        throw std::invalid_argument("Adaptive-P proposal belongs to a different state");
    }
    if (!std::isfinite(proposal.original_probability)
            || proposal.original_probability < 0.0 || proposal.original_probability > 1.0
            || !std::isfinite(proposal.next_weighted_sum)
            || !std::isfinite(proposal.next_total_weight)
            || proposal.next_total_weight <= 0.0) {
        throw std::invalid_argument("invalid Adaptive-P proposal");
    }
    if (!proposal.feedback_enabled) {
        if (proposal.adapted_target.has_value()
                || proposal.next_weighted_sum != state.weighted_sum
                || proposal.next_total_weight != state.total_weight) {
            throw std::invalid_argument("invalid disabled-feedback proposal");
        }
        return state;
    }
    if (!proposal.adapted_target.has_value()
            || !std::isfinite(*proposal.adapted_target)
            || *proposal.adapted_target < 0.0 || *proposal.adapted_target > 1.0) {
        throw std::invalid_argument("invalid adapted target in proposal");
    }
    return State{proposal.next_weighted_sum, proposal.next_total_weight};
}

} // namespace adaptive_p_select

#endif // CONTEXT_TIME_TRAVEL_ADAPTIVE_P_SELECT_H
