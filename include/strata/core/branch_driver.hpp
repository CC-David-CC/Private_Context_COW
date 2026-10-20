#include <functional>
#include <vector>
#include <string>
#pragma once
#include "strata/core/verify.hpp"
namespace strata::core {
struct ExpertDispatch;
// Private local pipe protocol used by the immutable-history adapter. No network
// listener and no native branch-reference database. Handles name derived state.
int branch_driver(const WeightTable&, const ModelGeometry&, SessionState&, const VerifyHits&,
                  const NativeHead*, PoolMultiFn, void*, ExpertDispatch&, int pcie_mode, ExpertSource* source, const ExpertCache* cache, const int32_t* residency,
                  int64_t chunk, void* stream,
                  const std::function<bool(const float*, float*, std::string&)>& head_rows,
                  const std::vector<int64_t>& initial_tokens, const std::string& output,
                  int64_t decode_steps);
}
