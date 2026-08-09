#include "costforge/loop_tiling.h"

#include <algorithm>
#include <cmath>

namespace costforge {

// ── small helpers ────────────────────────────────────

uint32_t LoopTiler::floor_to(uint32_t x, uint32_t m) {
    if (m == 0) return x;
    uint32_t r = (x / m) * m;
    return r == 0 ? m : r;
}

LoopTiler::LoopTiler(const HWProfile& hw) : hw_(hw), cost_(hw) {}

uint32_t LoopTiler::simd_lanes(uint32_t element_bytes) const {
    if (element_bytes == 0) element_bytes = 4;
    uint32_t bits = hw_.simd.max_width();          // 128/256/512
    uint32_t lanes = bits / (element_bytes * 8);
    return std::max<uint32_t>(lanes, 1);
}

// ── descriptors ──────────────────────────────────────

std::string LoopTiler::TilingPlan::descriptor() const {
    return std::to_string(mc) + "x" + std::to_string(kc) + "x" + std::to_string(nc);
}

std::string LoopTiler::StencilPlan::descriptor() const {
    return std::to_string(bi) + "x" + std::to_string(bj);
}

// ── matmul blocking ──────────────────────────────────
//
// We size three nested blocks against the real cache hierarchy:
//
//   register micro-tile  Mr x Nr   -> SIMD registers
//   L1 panel             Kc x Nr   -> L1d  (B micro-panel + A micro-panel)
//   L2 block             Mc x Kc   -> L2   (A block, reused across Nc)
//
// Everything is solved from L1d.size_kb / l2.size_kb, snapped to SIMD
// lanes, and clamped to the matrix dimension n.

LoopTiler::TilingPlan LoopTiler::tile_matmul(uint32_t element_bytes, uint64_t n) const {
    if (element_bytes == 0) element_bytes = 4;

    TilingPlan plan;
    plan.element_bytes = element_bytes;

    const uint32_t lanes = simd_lanes(element_bytes);

    // 1. Register micro-tile. Maximize register-resident accumulator
    //    area Mr*Nr subject to the SIMD register file:
    //      Mr*nr_regs (C accumulators) + nr_regs (B vectors) + 1 (A bcast)
    //      <= SIMD_REGISTERS (16 ymm/zmm).
    const uint32_t REGS = 16;
    uint32_t best_mr = 1, best_nr_regs = 1, best_area = 0;
    for (uint32_t mr = 1; mr <= 8; ++mr) {
        for (uint32_t nr_regs = 1; nr_regs <= 4; ++nr_regs) {
            uint32_t used = mr * nr_regs + nr_regs + 1;
            if (used > REGS) continue;
            uint32_t area = mr * (nr_regs * lanes);
            if (area > best_area) {
                best_area = area;
                best_mr = mr;
                best_nr_regs = nr_regs;
            }
        }
    }
    plan.mr = best_mr;
    plan.nr = best_nr_regs * lanes;

    // 2. L1 panel: solve Kc so the hot inner footprint fits in L1d.
    //    footprint = (B panel Kc*Nr) + (A micro-panel Mr*Kc) + (C tile Mr*Nr)
    //    Use a 0.5 fill factor: leave room for stack/other lines and to
    //    stay clear of the within-cache conflict-miss regime.
    const double FILL_L1 = 0.5;
    double l1_budget = static_cast<double>(hw_.l1d.size_kb) * 1024.0 * FILL_L1
                       / static_cast<double>(element_bytes);
    double kc_d = (l1_budget - static_cast<double>(plan.mr) * plan.nr)
                  / static_cast<double>(plan.nr + plan.mr);
    int32_t kc_i = static_cast<int32_t>(std::floor(kc_d));
    if (kc_i < 1) kc_i = 1;
    uint32_t kc = floor_to(static_cast<uint32_t>(kc_i), 4); // snap for k-unroll
    kc = std::max<uint32_t>(kc, 1);
    if (n > 0) kc = std::min<uint32_t>(kc, static_cast<uint32_t>(n));
    plan.kc = kc;

    // 3. L2 block: solve Mc so the A block (Mc*Kc) is L2-resident.
    const double FILL_L2 = 0.5;
    double l2_budget = static_cast<double>(hw_.l2.size_kb) * 1024.0 * FILL_L2
                       / static_cast<double>(element_bytes);
    double mc_d = l2_budget / static_cast<double>(plan.kc);
    uint32_t mc = floor_to(static_cast<uint32_t>(std::max(1.0, std::floor(mc_d))), plan.mr);
    if (n > 0) mc = std::min<uint32_t>(mc, static_cast<uint32_t>(n));
    plan.mc = std::max<uint32_t>(mc, plan.mr);

    // 4. Nc: cols handled per outer block. Size the B block (Kc*Nc)
    //    against L3 if present, else just clamp to n. Snap to Nr.
    uint64_t l3_bytes = static_cast<uint64_t>(hw_.l3.size_kb) * 1024;
    uint32_t nc;
    if (l3_bytes > 0) {
        double nc_d = (static_cast<double>(l3_bytes) * 0.5 / element_bytes)
                      / static_cast<double>(plan.kc);
        nc = floor_to(static_cast<uint32_t>(std::max(1.0, std::floor(nc_d))), plan.nr);
    } else {
        nc = (n > 0) ? static_cast<uint32_t>(n) : plan.nr;
    }
    if (n > 0) nc = std::min<uint32_t>(nc, static_cast<uint32_t>(n));
    plan.nc = std::max<uint32_t>(nc, plan.nr);

    // Final clamp: never let a micro-tile dimension exceed the matrix
    // itself (tiny-matrix edge case where Nr/Mr > n).
    if (n > 0) {
        uint32_t nn = static_cast<uint32_t>(n);
        plan.mc = std::min(plan.mc, nn);
        plan.kc = std::min(plan.kc, nn);
        plan.nc = std::min(plan.nc, nn);
    }

    // 5. Achieved footprints.
    plan.l1_footprint_bytes = static_cast<uint64_t>(
        plan.kc * plan.nr + plan.mr * plan.kc + plan.mr * plan.nr) * element_bytes;
    plan.l2_footprint_bytes = static_cast<uint64_t>(plan.mc) * plan.kc * element_bytes;

    uint64_t l1_bytes = static_cast<uint64_t>(hw_.l1d.size_kb) * 1024;
    uint64_t l2_bytes = static_cast<uint64_t>(hw_.l2.size_kb) * 1024;
    plan.valid = (plan.l1_footprint_bytes <= l1_bytes) &&
                 (plan.l2_footprint_bytes <= l2_bytes) &&
                 (plan.kc >= 1) && (plan.mr >= 1) && (plan.nr >= 1);

    // 6. Speedup from the cost model: untiled streams the whole 3*n^2
    //    working set through the hierarchy; tiled keeps the hot panel in
    //    L1. Ratio of average per-access memory latency.
    if (n > 0) {
        uint64_t untiled_ws = static_cast<uint64_t>(3) * n * n * element_bytes;
        double untiled_lat = cost_.avg_memory_latency(untiled_ws);
        double tiled_lat = cost_.avg_memory_latency(plan.l1_footprint_bytes);
        double sp = (tiled_lat > 0.0) ? untiled_lat / tiled_lat : 1.0;
        plan.estimated_speedup = std::clamp(sp, 1.0, 30.0);
    } else {
        plan.estimated_speedup = 1.0;
    }

    return plan;
}

// ── 2D stencil blocking ──────────────────────────────
//
// Square spatial block Bi x Bj, sized so the block plus its ghost halo
// ( (B + 2*halo)^2 elements ) is L1d-resident, snapped to SIMD lanes.

LoopTiler::StencilPlan LoopTiler::tile_stencil2d(uint32_t element_bytes,
                                                 uint64_t rows, uint64_t cols,
                                                 uint32_t halo) const {
    if (element_bytes == 0) element_bytes = 4;
    StencilPlan plan;
    plan.element_bytes = element_bytes;
    plan.halo = halo;

    const uint32_t lanes = simd_lanes(element_bytes);
    const double FILL = 0.5;
    double budget = static_cast<double>(hw_.l1d.size_kb) * 1024.0 * FILL
                    / static_cast<double>(element_bytes);

    // (B + 2h)^2 <= budget  ->  B <= sqrt(budget) - 2h
    double edge = std::sqrt(budget) - 2.0 * halo;
    int32_t b = static_cast<int32_t>(std::floor(edge));
    if (b < 1) b = 1;
    uint32_t bj = floor_to(static_cast<uint32_t>(b), lanes); // vectorize cols
    uint32_t bi = static_cast<uint32_t>(b);

    if (cols > 0) bj = std::min<uint32_t>(bj, static_cast<uint32_t>(cols));
    if (rows > 0) bi = std::min<uint32_t>(bi, static_cast<uint32_t>(rows));
    plan.bi = std::max<uint32_t>(bi, 1);
    plan.bj = std::max<uint32_t>(bj, 1);

    plan.l1_footprint_bytes = static_cast<uint64_t>(
        (plan.bi + 2 * halo)) * (plan.bj + 2 * halo) * element_bytes;

    uint64_t l1_bytes = static_cast<uint64_t>(hw_.l1d.size_kb) * 1024;
    plan.valid = plan.l1_footprint_bytes <= l1_bytes;

    if (rows > 0 && cols > 0) {
        uint64_t untiled_ws = rows * cols * element_bytes;
        double untiled_lat = cost_.avg_memory_latency(untiled_ws);
        double tiled_lat = cost_.avg_memory_latency(plan.l1_footprint_bytes);
        double sp = (tiled_lat > 0.0) ? untiled_lat / tiled_lat : 1.0;
        plan.estimated_speedup = std::clamp(sp, 1.0, 30.0);
    }

    return plan;
}

// ── back-compat square edge ──────────────────────────

uint32_t LoopTiler::square_tile_edge(uint32_t element_bytes) const {
    // Representative single tile edge = the L1 contraction block Kc,
    // computed by the real model (uses a large n so it isn't clamped).
    return tile_matmul(element_bytes, 4096).kc;
}

} // namespace costforge
