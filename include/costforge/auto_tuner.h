#pragma once

#include "costforge/types.h"
#include "costforge/cost_model.h"
#include <cstdint>
#include <string>
#include <vector>

namespace costforge {

/// AutoTuner — model-driven iterative-compilation search (v0.3.2).
///
/// LLVM picks unroll factor and vectorize width with independent
/// heuristics. CostForge already has unrolled_loop_cost() and
/// vectorized_loop_cost(); this tuner searches their *joint* parameter
/// space and returns the configuration that minimizes modeled cost.
///
/// It is a cost-model search (cheap, deterministic) — the real
/// "compile-N-variants-and-benchmark" loop in the v0.4.0 roadmap can
/// later plug measured costs in via the same TuneResult interface.
class AutoTuner {
public:
    explicit AutoTuner(const HWProfile& hw);

    struct Config {
        uint32_t unroll_factor = 1;   // 1,2,4,8,16
        uint32_t vector_width  = 0;    // 0 = scalar, else 128/256/512
    };

    struct TuneResult {
        Config best;
        CostResult best_cost;
        CostResult baseline_cost;      // unroll=1, scalar
        double speedup = 1.0;          // baseline.total() / best.total()
        std::vector<std::pair<Config, double>> grid;  // every point tried
        std::string descriptor() const;               // "u4_v256"
    };

    /// Search unroll x vectorize space for one loop.
    ///   body         : loop body opcodes
    ///   trip_count   : iteration count (0 = unknown)
    ///   working_set  : bytes touched by the loop
    TuneResult tune(const std::vector<std::string>& body,
                    uint64_t trip_count,
                    uint64_t working_set_bytes) const;

private:
    HWProfile hw_;
    CostModel cost_;
};

} // namespace costforge
