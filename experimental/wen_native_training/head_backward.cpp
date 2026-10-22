// Experimental CPU backward for one native Wen output-head LoRA only.
// The native forward supplies final normalized h and its actual logits.
// Base/recurrent/attention weights are frozen and outside this derivative.
// F32 rounding is treated as identity locally; no quantizer STE is supplied.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

extern "C" double wen_head_backward(int dim, int vocab, int rank,
        const float * h, const float * logits, const float * a, const float * b,
        float scale, int target, float * da, float * db) {
    const double bad = std::numeric_limits<double>::quiet_NaN();
    if (dim <= 0 || vocab <= 1 || rank <= 0 || target < 0 || target >= vocab ||
            !h || !logits || !a || !b || !da || !db || !std::isfinite(scale) || scale <= 0) return bad;
    for (int j = 0; j < dim; ++j) if (!std::isfinite(h[j])) return bad;
    for (int i = 0; i < vocab; ++i) if (!std::isfinite(logits[i])) return bad;
    std::vector<double> low(rank, 0.), dlow(rank, 0.);
    for (int r = 0; r < rank; ++r) {
        for (int j = 0; j < dim; ++j) low[r] += double(a[size_t(r)*dim+j])*h[j];
    }
    const double maximum = *std::max_element(logits, logits+vocab);
    double total = 0.;
    for (int i = 0; i < vocab; ++i) total += std::exp(double(logits[i])-maximum);
    for (int i = 0; i < vocab; ++i) {
        const double dz = std::exp(double(logits[i])-maximum)/total-(i == target ? 1. : 0.);
        for (int r = 0; r < rank; ++r) {
            if (!std::isfinite(b[size_t(i)*rank+r])) return bad;
            db[size_t(i)*rank+r] = float(scale*dz*low[r]);
            dlow[r] += scale*dz*b[size_t(i)*rank+r];
        }
    }
    for (int r = 0; r < rank; ++r) {
        for (int j = 0; j < dim; ++j) da[size_t(r)*dim+j] = float(dlow[r]*h[j]);
    }
    return maximum+std::log(total)-logits[target];
}
