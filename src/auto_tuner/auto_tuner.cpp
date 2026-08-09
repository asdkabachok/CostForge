#include "costforge/auto_tuner.h"

#include <algorithm>

namespace costforge {

AutoTuner::AutoTuner(const HWProfile& hw) : hw_(hw), cost_(hw) {}

std::string AutoTuner::TuneResult::descriptor() const {
    std::string s = "u" + std::to_string(best.unroll_factor);
    if (best.vector_width > 0) s += "_v" + std::to_string(best.vector_width);
    else s += "_scalar";
    return s;
}

AutoTuner::TuneResult AutoTuner::tune(const std::vector<std::string>& body,
                                      uint64_t trip_count,
                                      uint64_t working_set_bytes) const {
    TuneResult result;

    // Baseline: no unroll, scalar.
    result.baseline_cost = cost_.loop_cost(body, trip_count, working_set_bytes);
    result.best_cost = result.baseline_cost;
    result.best = {1, 0};
    result.grid.emplace_back(Config{1, 0}, result.baseline_cost.total());

    const uint32_t unroll_factors[] = {1, 2, 4, 8, 16};

    // Candidate vector widths: scalar (0) + every width the hardware
    // supports up to its max.
    std::vector<uint32_t> widths = {0};
    uint32_t maxw = hw_.simd.max_width();
    for (uint32_t w : {128u, 256u, 512u}) {
        if (w <= maxw && w >= 128) widths.push_back(w);
    }

    for (uint32_t uf : unroll_factors) {
        for (uint32_t vw : widths) {
            // Skip the baseline point (already recorded).
            if (uf == 1 && vw == 0) continue;

            CostResult c;
            if (vw == 0) {
                c = cost_.unrolled_loop_cost(body, trip_count, uf, working_set_bytes);
            } else if (uf == 1) {
                c = cost_.vectorized_loop_cost(body, trip_count, vw, working_set_bytes);
            } else {
                // Combined: vectorize the body, then model the unrolled
                // overhead reduction on top by scaling the effective trip
                // count. We approximate by taking the cheaper of the two
                // single transforms' structure: vectorize first (biggest
                // lever), then apply unroll's loop-overhead amortization.
                CostResult v = cost_.vectorized_loop_cost(body, trip_count, vw,
                                                          working_set_bytes);
                CostResult u = cost_.unrolled_loop_cost(body, trip_count, uf,
                                                        working_set_bytes);
                // Unroll mainly removes loop overhead; keep vectorized
                // throughput but credit the overhead amortization ratio.
                double overhead_ratio =
                    (result.baseline_cost.throughput_cycles > 0.0)
                        ? u.throughput_cycles / result.baseline_cost.throughput_cycles
                        : 1.0;
                c = v;
                c.throughput_cycles *= std::clamp(overhead_ratio, 0.5, 1.0);
            }

            result.grid.emplace_back(Config{uf, vw}, c.total());
            if (c.total() < result.best_cost.total()) {
                result.best_cost = c;
                result.best = {uf, vw};
            }
        }
    }

    double base = result.baseline_cost.total();
    double best = result.best_cost.total();
    result.speedup = (best > 0.0) ? base / best : 1.0;
    if (result.speedup < 1.0) result.speedup = 1.0;  // never report a regression

    return result;
}

} // namespace costforge
