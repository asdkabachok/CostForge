#pragma once

#include "costforge/cost_model.h"
#include "costforge/types.h"
#include <vector>

namespace costforge {

/// DecisionEngine — the brain of CostForge.
///
/// For each optimization opportunity in the IR, it:
/// 1. Computes cost of option A (don't optimize)
/// 2. Computes cost of option B (apply transformation)
/// 3. Picks the cheaper one
///
/// No thresholds. No heuristics. Pure cost comparison.

class DecisionEngine {
public:
    explicit DecisionEngine(const CostModel& model);
    
    /// Should we inline this function at this callsite?
    Decision should_inline(uint32_t callee_instruction_count,
                           double caller_register_pressure,
                           uint64_t callee_working_set,
                           uint32_t call_frequency) const;
    
    /// Should we unroll this loop? If yes, by what factor?
    struct UnrollDecision {
        bool should_unroll;
        uint32_t optimal_factor;  // 1 = don't unroll
        double savings_percent;
        std::vector<Decision> factors_evaluated;
    };
    UnrollDecision should_unroll(const std::vector<std::string>& body_opcodes,
                                  uint64_t trip_count,
                                  uint64_t working_set_bytes,
                                  uint32_t max_factor = 16) const;
    
    /// Should we vectorize this loop? If yes, at what width?
    struct VectorizeDecision {
        bool should_vectorize;
        uint32_t optimal_width;   // in bits: 128, 256, 512
        double savings_percent;
    };
    VectorizeDecision should_vectorize(const std::vector<std::string>& body_opcodes,
                                        uint64_t trip_count,
                                        uint64_t working_set_bytes,
                                        bool has_dependencies) const;
    
    /// Should we convert this branch to a conditional move?
    Decision should_branchless(double taken_probability,
                                uint32_t true_path_cost,
                                uint32_t false_path_cost) const;
    
    /// What is the optimal instruction schedule for this block?
    /// Returns reordered instruction indices
    std::vector<uint32_t> optimal_schedule(
        const std::vector<std::string>& opcodes,
        const std::vector<std::pair<uint32_t, uint32_t>>& dependencies) const;
    
    /// Get all decisions made (for logging/debugging)
    const std::vector<Decision>& decisions() const { return decisions_; }
    
    /// Summary statistics
    struct Stats {
        uint32_t total_decisions;
        uint32_t inlines_accepted;
        uint32_t inlines_rejected;
        uint32_t loops_unrolled;
        uint32_t loops_vectorized;
        uint32_t branches_converted;
        double total_estimated_savings_percent;
    };
    Stats stats() const;

private:
    const CostModel& model_;
    mutable std::vector<Decision> decisions_;
    
    void record_decision(Decision d) const;
};

} // namespace costforge
