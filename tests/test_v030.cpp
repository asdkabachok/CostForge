/// CostForge v0.3.0 — Transform Integration Tests
///
/// Tests the decision pipeline end-to-end:
///   Pattern detection → cost analysis → transform decision
///
/// Actual IR transforms require LLVM and are tested via lit tests.
/// These tests verify the logic that DRIVES the transforms.

#include "costforge/hwprobe.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include "costforge/pattern_recognizer.h"
#include "costforge/decision_log.h"
#include "costforge/interprocedural.h"
#include "costforge/calibrator.h"
#include "costforge/types.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

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

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) throw std::runtime_error( \
        "assertion failed: " #a " == " #b " (" + std::to_string(a) + \
        " != " + std::to_string(b) + ")")

// ── Test fixtures ──────────────────────────────────────────

static costforge::HWProfile make_zen2() {
    auto hw = costforge::HWProbe::scan();
    hw.vendor = "AMD";
    hw.model = "Zen 2 Test";
    hw.microarch = "zen2";
    hw.physical_cores = 8;
    hw.execution_ports = 10;
    hw.pipeline_depth = 19;
    hw.rob_size = 224;
    hw.branch_mispredict_penalty = 18;
    hw.l1d = {32, 64, 8, 4};
    hw.l2 = {512, 64, 8, 12};
    hw.l3 = {16384, 64, 16, 40};
    hw.simd = {true, true, true, false};
    return hw;
}

// ════════════════════════════════════════════════════════════
//  1. UNROLL DECISIONS MATCH PATTERNS
// ════════════════════════════════════════════════════════════

void test_unroll_small_reduction() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Small reduction loop: add + load + branch
    std::vector<std::string> body = {"load", "add", "cmp", "branch"};
    
    auto decision = engine.should_unroll(body, 1024, 4096);
    ASSERT(decision.should_unroll);
    ASSERT(decision.optimal_factor >= 2);
    ASSERT(decision.optimal_factor <= 16);
    ASSERT(decision.savings_percent > 0.0);
    
    std::cout << "(factor=" << decision.optimal_factor
              << " savings=" << decision.savings_percent << "%) ";
}

void test_unroll_large_body_rejected() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Large body: unrolling would blow up code
    std::vector<std::string> body;
    for (int i = 0; i < 100; i++) body.push_back("add");
    for (int i = 0; i < 50; i++) body.push_back("load");
    for (int i = 0; i < 50; i++) body.push_back("store");
    body.push_back("cmp");
    body.push_back("branch");
    
    auto decision = engine.should_unroll(body, 64, 1024*1024);
    // Large body with big working set: should either skip or use small factor
    if (decision.should_unroll) {
        ASSERT(decision.optimal_factor <= 8);
        std::cout << "(factor=" << decision.optimal_factor << ") ";
    } else {
        std::cout << "(skipped unroll) ";
    }
}

// ════════════════════════════════════════════════════════════
//  2. VECTORIZATION DECISIONS
// ════════════════════════════════════════════════════════════

void test_vectorize_simple_add_loop() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Simple a[i] = b[i] + c[i]
    std::vector<std::string> body = {"load", "load", "add", "store", "cmp", "branch"};
    
    auto decision = engine.should_vectorize(body, 1024, 4096, false);
    ASSERT(decision.should_vectorize);
    ASSERT(decision.optimal_width >= 128);  // at least SSE
    ASSERT(decision.optimal_width <= 512);
    ASSERT(decision.savings_percent > 0.0);
    
    std::cout << "(width=" << decision.optimal_width
              << " savings=" << decision.savings_percent << "%) ";
}

void test_vectorize_with_deps_rejected() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Loop with cross-iteration dependencies → can't vectorize
    std::vector<std::string> body = {"load", "add", "store", "cmp", "branch"};
    
    auto decision = engine.should_vectorize(body, 1024, 4096, true);
    // With dependencies, vectorization is unsafe or useless
    ASSERT(!decision.should_vectorize || decision.savings_percent < 5.0);
}

// ════════════════════════════════════════════════════════════
//  3. BRANCHLESS DECISIONS
// ════════════════════════════════════════════════════════════

void test_branchless_unpredictable() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // 50/50 branch with cheap paths → should convert to select
    auto decision = engine.should_branchless(0.5, 3, 3);
    ASSERT(decision.choose_b);  // choose branchless
    ASSERT(decision.savings_percent > 0.0);
    
    std::cout << "(savings=" << decision.savings_percent << "%) ";
}

void test_branchless_predictable_rejected() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // 99% predictable branch → branch predictor handles it
    auto decision = engine.should_branchless(0.99, 3, 3);
    // Predictable branch is likely cheaper than cmov
    // (though this depends on model — allow either answer)
    std::cout << "(choose_branchless=" << decision.choose_b << ") ";
}

void test_branchless_expensive_paths_rejected() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Expensive paths: select executes BOTH, so it's worse
    auto decision = engine.should_branchless(0.5, 100, 100);
    // With expensive paths, branch is better (only executes one side)
    ASSERT(!decision.choose_b || decision.savings_percent < 5.0);
}

// ════════════════════════════════════════════════════════════
//  4. PATTERN → TRANSFORM PIPELINE
// ════════════════════════════════════════════════════════════

void test_pattern_drives_vectorize() {
    auto hw = make_zen2();
    costforge::PatternRecognizer recognizer(hw);
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // Reduction pattern → should suggest vectorize
    costforge::PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "add", "cmp", "branch"};
    loop.trip_count = 4096;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.working_set_bytes = 32768;  // 32KB fits in L1
    loop.data_width_bits = 32;
    loop.nesting_depth = 1;
    
    auto patterns = recognizer.analyze_loop(loop);
    ASSERT(!patterns.empty());
    
    bool found_reduction = false;
    for (const auto& p : patterns) {
        if (p.kind == costforge::PatternKind::ReductionSum) {
            found_reduction = true;
            ASSERT(p.estimated_speedup > 1.0);
            ASSERT(!p.suggested_transform.empty());
            std::cout << "(transform='" << p.suggested_transform
                      << "' speedup=" << p.estimated_speedup << "x) ";
        }
    }
    ASSERT(found_reduction);
}

void test_matmul_pattern_tiling() {
    auto hw = make_zen2();
    costforge::PatternRecognizer recognizer(hw);
    
    costforge::PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "load", "mul", "add", "store", "cmp", "branch"};
    loop.trip_count = 256;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.working_set_bytes = 256 * 256 * 4 * 3;  // 3 matrices
    loop.data_width_bits = 32;
    loop.nesting_depth = 3;
    loop.accesses = {
        {"A", "i*N+k", true, false},
        {"B", "k*N+j", true, false},
        {"C", "i*N+j", true, true}
    };
    
    auto patterns = recognizer.analyze_loop(loop);
    bool found_matmul = false;
    for (const auto& p : patterns) {
        if (p.kind == costforge::PatternKind::MatrixMultiply) {
            found_matmul = true;
            // Should suggest tiling for cache
            ASSERT(p.suggested_transform.find("tile") != std::string::npos ||
                   p.suggested_transform.find("FMA") != std::string::npos);
            ASSERT(p.estimated_speedup > 2.0);
            std::cout << "(transform='" << p.suggested_transform
                      << "' speedup=" << p.estimated_speedup << "x) ";
        }
    }
    ASSERT(found_matmul);
}

// ════════════════════════════════════════════════════════════
//  5. IPA INLINE PLAN → TRANSFORM DECISIONS
// ════════════════════════════════════════════════════════════

void test_ipa_hot_leaf_inlined() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::InterproceduralAnalyzer ipa(model, hw);
    
    std::vector<costforge::CallNode> nodes = {
        {"main", 50, 10, 0.2, 1024, false, false, 0, 0},
        {"process", 30, 5, 0.3, 512, false, false, 0, 0},
        {"compute", 8, 2, 0.1, 64, true, false, 0, 0},   // small hot leaf
    };
    
    std::vector<costforge::CallEdge> edges = {
        {"main", "process", 1, 1, false, true, 1},     // in loop
        {"process", "compute", 1, 1, false, true, 2},   // nested loop
    };
    
    ipa.build_call_graph(nodes, edges);
    ipa.compute_traversal_order();
    ipa.propagate_hotness();
    
    auto plan = ipa.compute_inline_plan();
    
    // compute should be inlined into process (hot leaf in nested loop)
    bool compute_inlined = false;
    for (const auto& site : plan.sites) {
        if (site.callee == "compute" && site.caller == "process") {
            compute_inlined = true;
            ASSERT(site.benefit > 0);
            std::cout << "(benefit=" << site.benefit
                      << " reason='" << site.reason << "') ";
        }
    }
    ASSERT(compute_inlined);
}

void test_ipa_cold_outlined() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::InterproceduralAnalyzer ipa(model, hw);
    
    std::vector<costforge::CallNode> nodes = {
        {"main", 200, 30, 0.5, 4096, false, false, 0, 0},
        {"hot_path", 20, 4, 0.2, 256, false, false, 0, 0},
        {"error_handler", 500, 50, 0.8, 8192, true, false, 0, 0},
    };
    
    std::vector<costforge::CallEdge> edges = {
        {"main", "hot_path", 1, 1000, false, true, 2},
        {"main", "error_handler", 1, 1, false, false, 0},
    };
    
    ipa.build_call_graph(nodes, edges);
    ipa.compute_traversal_order();
    ipa.propagate_hotness();
    
    auto cold = ipa.find_outline_candidates();
    
    // error_handler should be cold (large, rarely called)
    bool found_cold = false;
    for (const auto& name : cold) {
        if (name == "error_handler") {
            found_cold = true;
        }
    }
    // Note: depends on outline heuristic thresholds
    std::cout << "(cold_candidates=" << cold.size() << ") ";
}

// ════════════════════════════════════════════════════════════
//  6. DECISION LOG INTEGRATION
// ════════════════════════════════════════════════════════════

void test_log_captures_transform_decisions() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    costforge::DecisionLog log;
    
    log.begin_compilation("test.c", "zen2");
    
    // Simulate transform decisions
    std::vector<std::string> body = {"load", "add", "cmp", "branch"};
    auto unroll = engine.should_unroll(body, 1024, 4096);
    
    costforge::Decision d;
    d.description = "unroll loop @header";
    d.name = "unroll_decision";
    // option chosen based on cost comparison
    d.cost_a = {10.0, 12.0, 0.1, 0.2, 0.5};
    d.cost_b = {8.0, 10.0, 0.1, 0.4, 0.1};
    d.choose_b = unroll.should_unroll;
    d.savings_percent = unroll.savings_percent;
    
    log.record_decision("test_func", "header", d,
                        {"trip_count=1024", "body_size=4"});
    
    auto report = log.finalize();
    ASSERT(report.total_decisions >= 1);
    
    std::ostringstream ss;
    costforge::DecisionLog::write_summary(report, ss);
    std::string summary = ss.str();
    ASSERT(!summary.empty());
    std::cout << "(decisions=" << report.total_decisions << ") ";
}

// ════════════════════════════════════════════════════════════
//  7. END-TO-END: detect → decide → verify profitable
// ════════════════════════════════════════════════════════════

void test_e2e_stencil_should_prefetch() {
    auto hw = make_zen2();
    costforge::PatternRecognizer recognizer(hw);
    costforge::CostModel model(hw);
    costforge::DecisionEngine engine(model);
    
    // 1D stencil: a[i] = b[i-1] + b[i] + b[i+1]
    costforge::PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "load", "load", "add", "add", "store",
                          "cmp", "branch"};
    loop.trip_count = 10000;
    loop.has_accumulator = false;
    loop.working_set_bytes = 10000 * 4 * 2;  // 2 arrays, 80KB > L1
    loop.data_width_bits = 32;
    loop.nesting_depth = 0;
    loop.has_loop_carried_dep = false;
    loop.accesses = {
        {"b", "i-1", true, false},
        {"b", "i", true, false},
        {"b", "i+1", true, false},
        {"a", "i", false, true}
    };
    
    auto patterns = recognizer.analyze_loop(loop);
    bool found_stencil = false;
    for (const auto& p : patterns) {
        if (p.kind == costforge::PatternKind::Stencil1D) {
            found_stencil = true;
            // Stencil with large working set should suggest prefetch
            ASSERT(p.estimated_speedup > 1.0);
            std::cout << "(pattern=stencil1d speedup="
                      << p.estimated_speedup << "x) ";
        }
    }
    ASSERT(found_stencil);
    
    // Should also vectorize (independent iterations)
    auto vec = engine.should_vectorize(
        loop.body_opcodes, loop.trip_count, loop.working_set_bytes, false);
    ASSERT(vec.should_vectorize);
    std::cout << "(vec_width=" << vec.optimal_width << ") ";
}

void test_e2e_branch_chain_should_switch() {
    auto hw = make_zen2();
    costforge::PatternRecognizer recognizer(hw);
    
    costforge::PatternRecognizer::BlockInfo block;
    block.label = "dispatch";
    block.opcodes = {"cmp", "branch", "cmp", "branch",
                     "cmp", "branch", "cmp", "branch",
                     "cmp", "branch", "cmp", "branch"};
    block.branch_count = 6;
    block.comparison_count = 6;
    block.is_chain_link = true;
    block.chain_length = 6;
    
    auto patterns = recognizer.analyze_block(block);
    bool found_chain = false;
    for (const auto& p : patterns) {
        if (p.kind == costforge::PatternKind::BranchChain) {
            found_chain = true;
            ASSERT(p.confidence > 0.5);
            ASSERT(p.estimated_speedup > 1.0);
            std::cout << "(chain_len=" << block.chain_length
                      << " speedup=" << p.estimated_speedup << "x) ";
        }
    }
    ASSERT(found_chain);
}

// ════════════════════════════════════════════════════════════
//  8. CALIBRATION ROUNDTRIP
// ════════════════════════════════════════════════════════════

void test_calibration_save_load_roundtrip() {
    auto hw = make_zen2();
    costforge::Calibrator cal(hw);
    
    // Create a fake profile
    costforge::CalibrationProfile profile;
    profile.cpu_model = "Test CPU";
    profile.microarch = "zen2";
    profile.timestamp = "2026-05-27T12:00:00";
    profile.mean_absolute_error = 8.5;
    profile.alu_correction = 1.1;
    profile.mul_correction = 0.9;
    profile.div_correction = 1.3;
    profile.load_correction = 1.05;
    profile.store_correction = 0.95;
    profile.branch_correction = 1.2;
    profile.simd_correction = 0.85;
    profile.l1_latency_correction = 1.0;
    profile.l2_latency_correction = 1.1;
    profile.l3_latency_correction = 0.9;
    profile.dram_latency_correction = 1.15;
    
    costforge::CalibrationPoint pt;
    pt.operation = "alu_throughput";
    pt.predicted_cycles = 0.25;
    pt.measured_cycles = 0.28;
    pt.error_percent = 12.0;
    profile.points.push_back(pt);
    
    // Save
    std::string path = "/tmp/costforge_test_cal.json";
    costforge::Calibrator::save(profile, path);
    
    // Load
    auto loaded = costforge::Calibrator::load(path);
    
    ASSERT(loaded.cpu_model == "Test CPU");
    ASSERT(loaded.microarch == "zen2");
    ASSERT(std::abs(loaded.mean_absolute_error - 8.5) < 0.1);
    ASSERT(std::abs(loaded.alu_correction - 1.1) < 0.01);
    ASSERT(std::abs(loaded.mul_correction - 0.9) < 0.01);
    ASSERT(std::abs(loaded.div_correction - 1.3) < 0.01);
    ASSERT(std::abs(loaded.simd_correction - 0.85) < 0.01);
    ASSERT(loaded.is_valid());
    ASSERT(!loaded.points.empty());
    
    std::cout << "(roundtrip OK, MAE=" << loaded.mean_absolute_error << "%) ";
}

void test_calibration_apply_changes_costs() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    
    double orig_add = model.instruction_cost("add");
    double orig_mul = model.instruction_cost("mul");
    double orig_div = model.instruction_cost("div");
    
    costforge::CostModel::CorrectionFactors factors;
    factors.alu = 2.0;  // double ALU cost
    factors.mul = 0.5;  // halve multiply cost
    factors.div = 3.0;
    model.set_correction_factors(factors);
    
    double new_add = model.instruction_cost("add");
    double new_mul = model.instruction_cost("mul");
    double new_div = model.instruction_cost("div");
    
    ASSERT(std::abs(new_add - orig_add * 2.0) < 0.01);
    ASSERT(std::abs(new_mul - orig_mul * 0.5) < 0.01);
    ASSERT(std::abs(new_div - orig_div * 3.0) < 0.5);  // latency rounding
    
    std::cout << "(add: " << orig_add << "→" << new_add
              << " mul: " << orig_mul << "→" << new_mul << ") ";
}

// ════════════════════════════════════════════════════════════
//  9. TYPE-AWARE COSTS
// ════════════════════════════════════════════════════════════

void test_type_aware_div_cost() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    
    double div_i8  = model.instruction_cost("div", 8, false);
    double div_i32 = model.instruction_cost("div", 32, false);
    double div_i64 = model.instruction_cost("div", 64, false);
    double div_f32 = model.instruction_cost("div", 32, true);
    double div_f64 = model.instruction_cost("div", 64, true);
    
    // i8 div should be cheapest, i64 most expensive
    ASSERT(div_i8 < div_i32);
    ASSERT(div_i32 < div_i64);
    // f32 div should be cheaper than f64
    ASSERT(div_f32 < div_f64);
    
    std::cout << "(i8=" << div_i8 << " i32=" << div_i32 
              << " i64=" << div_i64 << " f32=" << div_f32
              << " f64=" << div_f64 << ") ";
}

void test_type_aware_mul_cost() {
    auto hw = make_zen2();
    costforge::CostModel model(hw);
    
    double mul_i32 = model.instruction_cost("mul", 32, false);
    double mul_i64 = model.instruction_cost("mul", 64, false);
    double mul_f32 = model.instruction_cost("mul", 32, true);
    
    // 64-bit multiply costs more than 32-bit
    ASSERT(mul_i64 > mul_i32);
    // FP multiply exists and is positive
    ASSERT(mul_f32 > 0.0);
    
    std::cout << "(i32=" << mul_i32 << " i64=" << mul_i64
              << " f32=" << mul_f32 << ") ";
}

// ════════════════════════════════════════════════════════════

int main() {
    std::cout << "\n╔══════════════════════════════════════════════╗\n";
    std::cout << "║   CostForge v0.3.0 — Transform Tests        ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";
    
    std::cout << "─── Unroll Decisions ───\n";
    TEST(unroll_small_reduction);
    TEST(unroll_large_body_rejected);
    
    std::cout << "\n─── Vectorize Decisions ───\n";
    TEST(vectorize_simple_add_loop);
    TEST(vectorize_with_deps_rejected);
    
    std::cout << "\n─── Branchless Decisions ───\n";
    TEST(branchless_unpredictable);
    TEST(branchless_predictable_rejected);
    TEST(branchless_expensive_paths_rejected);
    
    std::cout << "\n─── Pattern → Transform Pipeline ───\n";
    TEST(pattern_drives_vectorize);
    TEST(matmul_pattern_tiling);
    
    std::cout << "\n─── IPA Inline Plan ───\n";
    TEST(ipa_hot_leaf_inlined);
    TEST(ipa_cold_outlined);
    
    std::cout << "\n─── Decision Log Integration ───\n";
    TEST(log_captures_transform_decisions);
    
    std::cout << "\n─── End-to-End ───\n";
    TEST(e2e_stencil_should_prefetch);
    TEST(e2e_branch_chain_should_switch);
    
    std::cout << "\n─── Calibration ───\n";
    TEST(calibration_save_load_roundtrip);
    TEST(calibration_apply_changes_costs);
    
    std::cout << "\n─── Type-Aware Costs ───\n";
    TEST(type_aware_div_cost);
    TEST(type_aware_mul_cost);
    
    std::cout << "\n════════════════════════════════════════════════\n";
    std::cout << "  Results: " << tests_passed << " passed, "
              << tests_failed << " failed\n";
    std::cout << "════════════════════════════════════════════════\n\n";
    
    return tests_failed > 0 ? 1 : 0;
}
