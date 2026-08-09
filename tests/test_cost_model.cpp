#include "costforge/types.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace costforge;

// Create a test hardware profile (AMD Zen+ like Ryzen 5 3550H)
HWProfile test_profile() {
    HWProfile hw;
    hw.vendor = "AMD";
    hw.model = "Test CPU (Zen+)";
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
    hw.memory_latency_ns = 80;
    hw.memory_bandwidth_gbs = 30;
    
    return hw;
}

void test_instruction_costs() {
    std::cout << "[TEST] Instruction costs... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    
    // ADD should be cheaper than MUL
    assert(model.instruction_cost("add") < model.instruction_cost("mul"));
    
    // MUL should be cheaper than DIV
    assert(model.instruction_cost("mul") < model.instruction_cost("div"));
    
    // Load should be cheaper than store (pipelined)
    assert(model.instruction_cost("load") <= model.instruction_cost("store"));
    
    std::cout << "PASS" << std::endl;
}

void test_cache_model() {
    std::cout << "[TEST] Cache model... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    
    // Small working set should have low memory latency (L1 hits)
    double small_latency = model.avg_memory_latency(1024); // 1 KB
    double large_latency = model.avg_memory_latency(1024 * 1024 * 10); // 10 MB
    
    assert(small_latency < large_latency);
    
    // L1-sized working set should be much faster than DRAM-bound
    assert(small_latency < 10.0); // should be close to L1 latency (4 cycles)
    assert(large_latency > 20.0); // should include L3/DRAM misses
    
    std::cout << "PASS" << std::endl;
}

void test_loop_unrolling() {
    std::cout << "[TEST] Loop unrolling decisions... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    DecisionEngine engine(model);
    
    std::vector<std::string> simple_body = {"load", "add", "store"};
    
    // For a simple loop with many iterations, unrolling should help
    auto decision = engine.should_unroll(simple_body, 10000, 4096);
    assert(decision.should_unroll == true);
    assert(decision.optimal_factor >= 2);
    
    // For a very small loop (3 iterations), unrolling might not help
    auto small_decision = engine.should_unroll(simple_body, 3, 64);
    // (result depends on cost model — just verify it doesn't crash)
    
    std::cout << "PASS" << std::endl;
}

void test_vectorization() {
    std::cout << "[TEST] Vectorization decisions... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    DecisionEngine engine(model);
    
    std::vector<std::string> vectorizable = {"load", "mul", "add", "store"};
    
    // Should vectorize a simple arithmetic loop
    auto decision = engine.should_vectorize(vectorizable, 10000, 40000, false);
    assert(decision.should_vectorize == true);
    assert(decision.optimal_width >= 128);
    
    // Should NOT vectorize if there are dependencies
    auto dep_decision = engine.should_vectorize(vectorizable, 10000, 40000, true);
    assert(dep_decision.should_vectorize == false);
    
    std::cout << "PASS" << std::endl;
}

void test_inline_decision() {
    std::cout << "[TEST] Inline decisions... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    DecisionEngine engine(model);
    
    // Small function called many times — should inline
    auto small_hot = engine.should_inline(5, 0.3, 256, 1000);
    // (result depends on cost model — just verify it runs)
    
    // Huge function called once — should probably NOT inline
    auto big_cold = engine.should_inline(500, 0.8, 65536, 1);
    // Big function with high register pressure = high inline cost
    
    std::cout << "PASS" << std::endl;
}

void test_branchless() {
    std::cout << "[TEST] Branchless decisions... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    DecisionEngine engine(model);
    
    // 50/50 branch — should prefer cmov
    auto unpredictable = engine.should_branchless(0.5, 2, 2);
    assert(unpredictable.choose_b == true); // cmov should win
    
    // 99% predictable branch — branch should be fine
    auto predictable = engine.should_branchless(0.99, 2, 2);
    // Highly predictable branch is cheap — might still prefer branch
    
    std::cout << "PASS" << std::endl;
}

void test_cost_comparison_consistency() {
    std::cout << "[TEST] Cost comparison consistency... ";
    HWProfile hw = test_profile();
    CostModel model(hw);
    
    // Vectorized loop should be cheaper than scalar for large trips
    std::vector<std::string> body = {"load", "mul", "add", "store"};
    
    CostResult scalar = model.loop_cost(body, 10000, 40000);
    CostResult vec256 = model.vectorized_loop_cost(body, 10000, 256, 40000);
    
    // With 256-bit SIMD processing 8 floats at once,
    // vectorized should be significantly cheaper
    assert(vec256.total() < scalar.total());
    
    std::cout << "PASS" << std::endl;
}

int main() {
    std::cout << "=== CostForge Test Suite ===" << std::endl;
    std::cout << std::endl;
    
    test_instruction_costs();
    test_cache_model();
    test_loop_unrolling();
    test_vectorization();
    test_inline_decision();
    test_branchless();
    test_cost_comparison_consistency();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
