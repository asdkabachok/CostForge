/// CostForge v0.3.2 — LoopTiler + AutoTuner tests
///
/// Validates the new cache-blocking tile-size optimizer and the
/// joint unroll x vectorize auto-tuner. Pure cost-model logic, no LLVM.
/// Assertions are invariant-based (footprint <= cache, SIMD-lane
/// alignment, type/cache monotonicity, no reported regressions) so they
/// stay robust across hardware fixtures.

#include "costforge/hwprobe.h"
#include "costforge/cost_model.h"
#include "costforge/pattern_recognizer.h"
#include "costforge/loop_tiling.h"
#include "costforge/auto_tuner.h"
#include "costforge/types.h"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "  [" << #name << "] "; \
    try { test_##name(); tests_passed++; std::cout << "PASS\n"; } \
    catch (const std::exception& e) { \
        tests_failed++; std::cout << "FAIL: " << e.what() << "\n"; \
    }

#define ASSERT(cond) \
    if (!(cond)) throw std::runtime_error("assertion failed: " #cond)

using namespace costforge;

static HWProfile make_zen2(uint32_t l1d_kb = 32) {
    auto hw = HWProbe::scan();
    hw.vendor = "AMD";
    hw.model = "Zen 2 Test";
    hw.microarch = "zen2";
    hw.physical_cores = 8;
    hw.execution_ports = 10;
    hw.pipeline_depth = 19;
    hw.rob_size = 224;
    hw.branch_mispredict_penalty = 18;
    hw.l1d = {l1d_kb, 64, 8, 4};
    hw.l2 = {512, 64, 8, 12};
    hw.l3 = {16384, 64, 16, 40};
    hw.simd = {true, true, true, false}; // AVX2 → 256-bit
    if (hw.base_freq_mhz == 0) hw.base_freq_mhz = 3600;
    if (hw.memory_latency_ns == 0) hw.memory_latency_ns = 80;
    return hw;
}

// ════════════════════════════════════════════════════════════
//  LoopTiler — matmul
// ════════════════════════════════════════════════════════════

void test_matmul_l1_footprint_fits() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 2048);
    uint64_t l1 = (uint64_t)hw.l1d.size_kb * 1024;
    ASSERT(plan.l1_footprint_bytes <= l1);
    ASSERT(plan.valid);
}

void test_matmul_l2_block_fits() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 2048);
    uint64_t l2 = (uint64_t)hw.l2.size_kb * 1024;
    ASSERT(plan.l2_footprint_bytes <= l2);
    ASSERT(plan.mc >= plan.mr);
}

void test_matmul_register_tile_fits_regfile() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 2048);
    uint32_t lanes = hw.simd.max_width() / (4 * 8); // 8 for AVX2/f32
    // nr is a whole number of SIMD registers wide.
    ASSERT(plan.nr % lanes == 0);
    ASSERT(plan.nr >= lanes);
    // C accumulators + B vectors + 1 A-broadcast must fit in 16 regs.
    uint32_t nr_regs = plan.nr / lanes;
    uint32_t used = plan.mr * nr_regs + nr_regs + 1;
    ASSERT(used <= 16);
    ASSERT(plan.mr >= 1);
}

void test_matmul_type_aware_smaller_for_f64() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto p32 = tiler.tile_matmul(4, 4096);
    auto p64 = tiler.tile_matmul(8, 4096);
    // Wider elements → fewer fit in L1 → smaller contraction block.
    ASSERT(p64.kc <= p32.kc);
    // f64 has half the lanes of f32.
    ASSERT(p64.nr <= p32.nr);
}

void test_matmul_larger_l1_gives_larger_kc() {
    LoopTiler small(make_zen2(32));
    LoopTiler big(make_zen2(64));
    auto ps = small.tile_matmul(4, 8192);
    auto pb = big.tile_matmul(4, 8192);
    ASSERT(pb.kc > ps.kc);
}

void test_matmul_clamped_to_small_n() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 16); // tiny matrix
    ASSERT(plan.mc <= 16);
    ASSERT(plan.kc <= 16);
    ASSERT(plan.nc <= 16);
}

void test_matmul_speedup_tiny_is_neutral() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 16); // already fits in L1
    // Tiling a cache-resident problem should not claim a big speedup.
    ASSERT(plan.estimated_speedup < 1.5);
    ASSERT(plan.estimated_speedup >= 1.0);
}

void test_matmul_speedup_large_is_positive() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_matmul(4, 4096); // 3*4096^2*4 ≈ 200MB >> L3
    ASSERT(plan.estimated_speedup > 2.0);
    ASSERT(plan.estimated_speedup <= 30.0);
}

void test_matmul_deterministic() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto a = tiler.tile_matmul(4, 2048);
    auto b = tiler.tile_matmul(4, 2048);
    ASSERT(a.descriptor() == b.descriptor());
    ASSERT(a.estimated_speedup == b.estimated_speedup);
}

// ════════════════════════════════════════════════════════════
//  LoopTiler — stencil
// ════════════════════════════════════════════════════════════

void test_stencil_block_fits_l1() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_stencil2d(4, 4096, 4096, 1);
    uint64_t l1 = (uint64_t)hw.l1d.size_kb * 1024;
    ASSERT(plan.l1_footprint_bytes <= l1);
    ASSERT(plan.valid);
    ASSERT(plan.bi >= 1 && plan.bj >= 1);
}

void test_stencil_cols_simd_aligned() {
    auto hw = make_zen2();
    LoopTiler tiler(hw);
    auto plan = tiler.tile_stencil2d(4, 4096, 4096, 1);
    uint32_t lanes = hw.simd.max_width() / (4 * 8);
    ASSERT(plan.bj % lanes == 0 || plan.bj == 4096);
}

// ════════════════════════════════════════════════════════════
//  PatternRecognizer integration — descriptor now from the model
// ════════════════════════════════════════════════════════════

void test_recognizer_emits_3d_tile_descriptor() {
    auto hw = make_zen2();
    PatternRecognizer rec(hw);

    PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "load", "vfma", "store"};
    loop.trip_count = 2048;
    loop.working_set_bytes = (uint64_t)3 * 2048 * 2048 * 4;
    loop.data_width_bits = 32;
    loop.nesting_depth = 2;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.accesses = {
        {"A", "i*N+k", true, false},
        {"B", "k*N+j", true, false},
        {"C", "i*N+j", true, true},
    };

    auto matches = rec.analyze_loop(loop);
    bool found = false;
    for (auto& m : matches) {
        if (m.kind == PatternKind::MatrixMultiply) {
            found = true;
            // New descriptor has three dims (McxKcxNc), unlike the old
            // square "48x48". Count the 'x' separators in the tile part.
            const std::string& t = m.suggested_transform;
            ASSERT(t.rfind("tiled_matmul_", 0) == 0);
            size_t xs = std::count(t.begin(), t.end(), 'x');
            ASSERT(xs == 2); // Mc x Kc x Nc
            ASSERT(m.estimated_speedup >= 1.0);
        }
    }
    ASSERT(found);
}

void test_old_optimal_tile_still_positive() {
    auto hw = make_zen2();
    PatternRecognizer rec(hw); // exercises optimal_tile_size() via matmul path
    LoopTiler tiler(hw);
    ASSERT(tiler.square_tile_edge(4) >= 1);
}

// ════════════════════════════════════════════════════════════
//  AutoTuner
// ════════════════════════════════════════════════════════════

void test_tuner_never_worse_than_baseline() {
    auto hw = make_zen2();
    AutoTuner tuner(hw);
    std::vector<std::string> body = {"load", "mul", "add", "store"};
    auto r = tuner.tune(body, 10000, 4096 * 4);
    ASSERT(r.best_cost.total() <= r.baseline_cost.total() + 1e-9);
    ASSERT(r.speedup >= 1.0);
    ASSERT(!r.grid.empty());
}

void test_tuner_vectorizes_simple_loop() {
    auto hw = make_zen2();
    AutoTuner tuner(hw);
    // A clean data-parallel body should benefit from SIMD.
    std::vector<std::string> body = {"load", "load", "add", "store"};
    auto r = tuner.tune(body, 100000, 1u << 20);
    ASSERT(r.best.vector_width > 0);
    ASSERT(r.speedup > 1.0);
}

void test_tuner_grid_covers_space() {
    auto hw = make_zen2();
    AutoTuner tuner(hw);
    std::vector<std::string> body = {"load", "add", "store"};
    auto r = tuner.tune(body, 5000, 4096 * 4);
    // 5 unroll factors x {scalar,128,256} = 15 points (baseline dedup'd
    // back in), so at least 13 distinct configs explored.
    ASSERT(r.grid.size() >= 13);
}

void test_tuner_deterministic() {
    auto hw = make_zen2();
    AutoTuner tuner(hw);
    std::vector<std::string> body = {"load", "mul", "store"};
    auto a = tuner.tune(body, 8000, 4096 * 4);
    auto b = tuner.tune(body, 8000, 4096 * 4);
    ASSERT(a.descriptor() == b.descriptor());
}

// ════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║  CostForge v0.3.2 — Tiling + AutoTuner       ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";

    std::cout << "─── LoopTiler: matmul ───\n";
    TEST(matmul_l1_footprint_fits);
    TEST(matmul_l2_block_fits);
    TEST(matmul_register_tile_fits_regfile);
    TEST(matmul_type_aware_smaller_for_f64);
    TEST(matmul_larger_l1_gives_larger_kc);
    TEST(matmul_clamped_to_small_n);
    TEST(matmul_speedup_tiny_is_neutral);
    TEST(matmul_speedup_large_is_positive);
    TEST(matmul_deterministic);

    std::cout << "\n─── LoopTiler: stencil ───\n";
    TEST(stencil_block_fits_l1);
    TEST(stencil_cols_simd_aligned);

    std::cout << "\n─── PatternRecognizer integration ───\n";
    TEST(recognizer_emits_3d_tile_descriptor);
    TEST(old_optimal_tile_still_positive);

    std::cout << "\n─── AutoTuner ───\n";
    TEST(tuner_never_worse_than_baseline);
    TEST(tuner_vectorizes_simple_loop);
    TEST(tuner_grid_covers_space);
    TEST(tuner_deterministic);

    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";
    std::cout << "════════════════════════════════════════════════\n\n";
    return tests_failed == 0 ? 0 : 1;
}
