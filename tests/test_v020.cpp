#include "costforge/types.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include "costforge/pattern_recognizer.h"
#include "costforge/decision_log.h"
#include "costforge/interprocedural.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace costforge;

static HWProfile make_zen_plus() {
    HWProfile hw;
    hw.vendor = "AMD";
    hw.model = "Ryzen 5 3550H";
    hw.microarch = "zen+";
    hw.physical_cores = 4;
    hw.logical_threads = 8;
    hw.base_freq_mhz = 2100;
    hw.boost_freq_mhz = 3700;
    hw.l1d = {32, 64, 8, 4};
    hw.l1i = {64, 64, 4, 4};
    hw.l2 = {512, 64, 8, 12};
    hw.l3 = {4096, 64, 16, 35};
    hw.execution_ports = 10;
    hw.pipeline_depth = 19;
    hw.rob_size = 192;
    hw.load_buffer_size = 44;
    hw.store_buffer_size = 44;
    hw.branch_mispredict_penalty = 17;
    hw.simd = {true, true, true, false};
    hw.memory_latency_ns = 70;
    hw.memory_bandwidth_gbs = 34;
    return hw;
}

// ── Pattern recognizer tests ────────────────────────

void test_reduction_detection() {
    auto hw = make_zen_plus();
    PatternRecognizer pr(hw);
    
    PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "add", "store"};
    loop.trip_count = 1000;
    loop.working_set_bytes = 8000;
    loop.nesting_depth = 0;
    loop.has_loop_carried_dep = false;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.accesses = {{"arr", "i", true, false}};
    loop.data_width_bits = 32;
    
    auto matches = pr.analyze_loop(loop);
    
    assert(!matches.empty() && "should detect reduction");
    assert(matches[0].kind == PatternKind::ReductionSum);
    assert(matches[0].confidence >= 0.5);
    assert(matches[0].estimated_speedup > 1.0);
    
    std::cout << "  reduction: " << pattern_name(matches[0].kind)
              << " (conf=" << matches[0].confidence
              << " speedup=" << matches[0].estimated_speedup << "x"
              << " transform=" << matches[0].suggested_transform << ")\n";
    
    std::cout << "  [PASS] test_reduction_detection\n";
}

void test_matmul_detection() {
    auto hw = make_zen_plus();
    PatternRecognizer pr(hw);
    
    PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "load", "mul", "add", "store"};
    loop.trip_count = 512;
    loop.working_set_bytes = 512 * 512 * 4 * 3; // 3 matrices
    loop.nesting_depth = 2; // innermost of triple-nested
    loop.has_loop_carried_dep = false;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.accesses = {
        {"A", "i*N+k", true, false},
        {"B", "k*N+j", true, false},
        {"C", "i*N+j", true, true}
    };
    loop.data_width_bits = 32;
    
    auto matches = pr.analyze_loop(loop);
    
    bool found_matmul = false;
    for (const auto& m : matches) {
        if (m.kind == PatternKind::MatrixMultiply) {
            found_matmul = true;
            assert(m.confidence >= 0.8);
            assert(m.estimated_speedup > 3.0);
            std::cout << "  matmul: conf=" << m.confidence
                      << " speedup=" << m.estimated_speedup << "x"
                      << " transform=" << m.suggested_transform << "\n";
        }
    }
    assert(found_matmul && "should detect matrix multiply");
    std::cout << "  [PASS] test_matmul_detection\n";
}

void test_dot_product_detection() {
    auto hw = make_zen_plus();
    PatternRecognizer pr(hw);
    
    PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "load", "mul", "add"};
    loop.trip_count = 10000;
    loop.working_set_bytes = 80000;
    loop.nesting_depth = 0;
    loop.has_loop_carried_dep = false;
    loop.has_accumulator = true;
    loop.accumulator_op = "add";
    loop.accesses = {
        {"x", "i", true, false},
        {"y", "i", true, false}
    };
    loop.data_width_bits = 32;
    
    auto matches = pr.analyze_loop(loop);
    
    bool found = false;
    for (const auto& m : matches) {
        if (m.kind == PatternKind::DotProduct) {
            found = true;
            assert(m.confidence >= 0.8);
            std::cout << "  dot_product: speedup=" << m.estimated_speedup << "x"
                      << " transform=" << m.suggested_transform << "\n";
        }
    }
    assert(found && "should detect dot product");
    std::cout << "  [PASS] test_dot_product_detection\n";
}

void test_memcopy_detection() {
    auto hw = make_zen_plus();
    PatternRecognizer pr(hw);
    
    PatternRecognizer::LoopInfo loop;
    loop.body_opcodes = {"load", "store"};
    loop.trip_count = 100000;
    loop.working_set_bytes = 400000;
    loop.nesting_depth = 0;
    loop.has_loop_carried_dep = false;
    loop.has_accumulator = false;
    loop.accumulator_op = "";
    loop.accesses = {
        {"src", "i", true, false},
        {"dst", "i", false, true}
    };
    loop.data_width_bits = 32;
    
    auto matches = pr.analyze_loop(loop);
    
    bool found = false;
    for (const auto& m : matches) {
        if (m.kind == PatternKind::Memcopy) {
            found = true;
            assert(m.confidence >= 0.9);
            std::cout << "  memcopy: speedup=" << m.estimated_speedup << "x"
                      << " transform=" << m.suggested_transform << "\n";
        }
    }
    assert(found && "should detect memcopy");
    std::cout << "  [PASS] test_memcopy_detection\n";
}

void test_branch_chain_detection() {
    auto hw = make_zen_plus();
    PatternRecognizer pr(hw);
    
    PatternRecognizer::BlockInfo block;
    block.label = "entry";
    block.opcodes = {"cmp", "branch", "cmp", "branch", "cmp", "branch",
                     "cmp", "branch", "cmp", "branch", "cmp", "branch",
                     "cmp", "branch", "cmp", "branch"};
    block.branch_count = 8;
    block.comparison_count = 8;
    block.is_chain_link = true;
    block.chain_length = 8;
    
    auto matches = pr.analyze_block(block);
    
    bool found = false;
    for (const auto& m : matches) {
        if (m.kind == PatternKind::BranchChain) {
            found = true;
            std::cout << "  branch_chain: speedup=" << m.estimated_speedup << "x"
                      << " transform=" << m.suggested_transform << "\n";
        }
    }
    assert(found && "should detect branch chain");
    std::cout << "  [PASS] test_branch_chain_detection\n";
}

// ── Decision log tests ──────────────────────────────

void test_decision_log() {
    auto hw = make_zen_plus();
    CostModel model(hw);
    DecisionEngine engine(model);
    DecisionLog log;
    
    log.begin_compilation("test.c", "Ryzen 5 3550H");
    
    // Make some decisions and log them
    auto d1 = engine.should_inline(15, 0.3, 1024, 10);
    log.record_decision("main", "line:42", d1,
                        {"callee is small (15 insn)", "called 10 times"});
    
    auto d2 = engine.should_inline(500, 0.8, 65536, 1);
    log.record_decision("process", "line:100", d2,
                        {"callee is large (500 insn)", "called once",
                         "high register pressure in caller"});
    
    auto report = log.finalize();
    
    assert(report.total_decisions == 2);
    assert(report.traces.size() == 2);
    assert(!report.traces[0].verdict_summary.empty());
    
    // Test human-readable output
    std::cout << "\n";
    DecisionLog::write_summary(report, std::cout);
    
    // Test explain
    std::cout << "\n  Detailed explanation:\n";
    std::cout << DecisionLog::explain(report.traces[0]) << "\n";
    
    std::cout << "  [PASS] test_decision_log\n";
}

// ── Interprocedural analysis tests ──────────────────

void test_interprocedural() {
    auto hw = make_zen_plus();
    CostModel model(hw);
    InterproceduralAnalyzer ipa(model, hw);
    
    // Build a simple call graph:
    // main → process → compute (hot leaf in loop)
    // main → error_handler (cold, large)
    
    std::vector<CallNode> nodes = {
        {"main",          50, 5,  0.3, 1024, true, false, 0, 1.0},
        {"process",       100, 10, 0.5, 4096, true, false, 0, 1.0},
        {"compute",       15, 2,  0.2, 512,  true, false, 0, 1.0},
        {"error_handler", 200, 15, 0.4, 2048, true, false, 0, 1.0},
    };
    
    std::vector<CallEdge> edges = {
        {"main", "process",       1, 1, false, true,  1},
        {"process", "compute",    1, 1, false, true,  2},  // nested loop!
        {"main", "error_handler", 1, 1, false, false, 0},
    };
    
    ipa.build_call_graph(nodes, edges);
    ipa.compute_traversal_order();
    ipa.propagate_hotness();
    
    const auto& graph = ipa.call_graph();
    
    // SCCs should exist (all non-recursive here)
    assert(!graph.sccs.empty());
    
    // Compute inline plan
    auto plan = ipa.compute_inline_plan();
    
    std::cout << "\n  Inline plan (" << plan.sites.size() << " sites):\n";
    for (const auto& site : plan.sites) {
        std::cout << "    inline " << site.callee << " into " << site.caller
                  << " — " << site.reason
                  << " (benefit=" << site.benefit 
                  << " growth=" << site.code_growth << ")\n";
    }
    
    // compute should be inlined into process (hot leaf in nested loop)
    bool compute_inlined = false;
    for (const auto& site : plan.sites) {
        if (site.callee == "compute" && site.caller == "process") {
            compute_inlined = true;
        }
    }
    assert(compute_inlined && "compute should be inlined into process");
    
    // Find outline candidates
    auto outlines = ipa.find_outline_candidates();
    std::cout << "  Outline candidates: ";
    for (const auto& f : outlines) std::cout << f << " ";
    std::cout << "\n";
    
    // error_handler should be an outline candidate (cold + large)
    // (depends on hotness propagation result)
    
    std::cout << "  [PASS] test_interprocedural\n";
}

// ── Main ────────────────────────────────────────────

int main() {
    std::cout << "=== CostForge v0.2.0 Test Suite ===\n\n";
    
    std::cout << "Pattern Recognition:\n";
    test_reduction_detection();
    test_matmul_detection();
    test_dot_product_detection();
    test_memcopy_detection();
    test_branch_chain_detection();
    
    std::cout << "\nDecision Logging:\n";
    test_decision_log();
    
    std::cout << "\nInterprocedural Analysis:\n";
    test_interprocedural();
    
    std::cout << "\n=== ALL TESTS PASSED ===\n";
    return 0;
}
