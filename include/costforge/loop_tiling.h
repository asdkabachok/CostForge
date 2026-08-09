#pragma once

#include "costforge/types.h"
#include "costforge/cost_model.h"
#include <cstdint>
#include <string>

namespace costforge {

/// LoopTiler — cache-blocking tile-size optimizer (v0.3.2).
///
/// Replaces the old single-level square-tile heuristic
/// (T = sqrt(L1 / (3*elem))) with a proper, hardware-grounded
/// cache-blocking model in the GotoBLAS/BLIS spirit:
///
///   - register micro-tile  Mr x Nr  lives in SIMD registers
///   - L1 panel             Kc x Nr  (B-panel) streams through L1d
///   - L2 block             Mc x Kc  (A-block) is resident in L2
///
/// All three dimensions are solved against the *actual* cache sizes
/// from the HWProfile, are type-aware (element width matters), and
/// snap to SIMD-lane multiples. The estimated speedup is derived from
/// the CostModel's own memory-access model — not a magic constant —
/// so it stays consistent with the rest of CostForge.
///
/// Pure math, no LLVM dependency: fully unit-testable standalone.
class LoopTiler {
public:
    explicit LoopTiler(const HWProfile& hw);

    /// A concrete cache-blocking plan for a matmul-shaped loop nest.
    struct TilingPlan {
        // Cache block dimensions, in *elements*.
        uint32_t mc = 0;   // rows of A / C handled per L2 block
        uint32_t kc = 0;   // shared (contraction) dimension per L1 panel
        uint32_t nc = 0;   // cols of B / C handled per outer block

        // Register micro-tile, in *elements*.
        uint32_t mr = 0;   // micro-rows
        uint32_t nr = 0;   // micro-cols (= SIMD lanes, by construction)

        uint32_t element_bytes = 4;

        // Resident footprints actually achieved by the plan.
        uint64_t l1_footprint_bytes = 0;
        uint64_t l2_footprint_bytes = 0;

        bool valid = false;

        /// Speedup of the tiled nest vs the untiled nest, estimated
        /// from the CostModel memory cost (>= 1.0 when tiling helps,
        /// ~1.0 when the problem already fits in cache).
        double estimated_speedup = 1.0;

        /// Compact descriptor for transform strings, e.g. "64x256x512".
        /// Order is Mc x Kc x Nc.
        std::string descriptor() const;
    };

    /// Compute an optimal cache-blocking plan for an n x n matmul
    /// (C[n,n] += A[n,n] * B[n,n]) with the given element size.
    ///   element_bytes : 1/2/4/8 (i8/i16/{i32,f32}/{i64,f64})
    ///   n             : matrix dimension (used for clamping + speedup)
    TilingPlan tile_matmul(uint32_t element_bytes, uint64_t n) const;

    /// Compute a 2D blocking plan for a stencil sweep over an
    /// rows x cols grid (square Bi x Bj blocks sized to L1d).
    struct StencilPlan {
        uint32_t bi = 0;            // block rows
        uint32_t bj = 0;            // block cols
        uint32_t element_bytes = 4;
        uint32_t halo = 1;          // ghost-cell width per side
        uint64_t l1_footprint_bytes = 0;
        bool valid = false;
        double estimated_speedup = 1.0;
        std::string descriptor() const;   // "BixBj"
    };

    StencilPlan tile_stencil2d(uint32_t element_bytes,
                               uint64_t rows, uint64_t cols,
                               uint32_t halo = 1) const;

    /// Back-compat convenience: a single representative square tile
    /// edge (returns the L1 contraction block Kc). Kept so existing
    /// callers of the old optimal_tile_size() keep working.
    uint32_t square_tile_edge(uint32_t element_bytes = 4) const;

private:
    HWProfile hw_;
    CostModel cost_;

    /// SIMD lanes for a given element size on this hardware.
    uint32_t simd_lanes(uint32_t element_bytes) const;

    /// Round x down to a positive multiple of m (never returns 0 if m>0).
    static uint32_t floor_to(uint32_t x, uint32_t m);
};

} // namespace costforge
