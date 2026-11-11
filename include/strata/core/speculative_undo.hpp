#pragma once
#include "strata/core/session.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
// Ephemeral preimage of the physical KV/indexer rows a single verify window
// can overwrite. The caller must first prepare those writes with BranchSession
// and must commit only the target-accepted prefix before restore_suffix().
class SpeculativeUndo {
public:
    SpeculativeUndo() = default;
    ~SpeculativeUndo();
    SpeculativeUndo(const SpeculativeUndo&) = delete;
    SpeculativeUndo& operator=(const SpeculativeUndo&) = delete;
    bool capture(SessionState& state, const ModelGeometry& geometry, int64_t first, int count, std::string& error);
    bool capture_candidates(std::string& error);
    // The caller must prepare COW writes again before reapplying a window.
    bool reapply_candidates(std::string& error);
    bool restore_suffix(int accepted, std::string& error);
private:
    struct Slot { void* target; size_t offset, bytes; int64_t position; bool pooled; };
    std::vector<Slot> slots_;
    unsigned char* saved_ = nullptr;
    unsigned char* candidates_ = nullptr;
    size_t bytes_ = 0;
    int64_t first_ = 0;
    int count_ = 0;
};
}
