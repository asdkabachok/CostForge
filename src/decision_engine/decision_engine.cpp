#include "costforge/decision_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace costforge {

DecisionEngine::DecisionEngine(const CostModel& model) : model_(model) {}

void DecisionEngine::record_decision(Decision d) const {
    d.evaluate();
    decisions_.push_back(std::move(d));
}

Decision DecisionEngine::should_inline(uint32_t callee_instruction_count,
                                         double caller_register_pressure,
                                         uint64_t callee_working_set,
                                         uint32_t call_frequency) const {
    Decision d;
    d.name = "inline";
    d.description = "inline " + std::to_string(callee_instruction_count) + 
                    " insn function (called " + std::to_string(call_frequency) + "x)";
    
    // Option A: don't inline (call)
    d.cost_a = model_.call_cost(callee_instruction_count, caller_register_pressure);
    // Scale by call frequency
    d.cost_a.throughput_cycles *= call_frequency;
    d.cost_a.latency_cycles *= call_frequency;
    
    // Option B: inline
    d.cost_b = model_.inline_cost(callee_instruction_count, 
                                    caller_register_pressure,
                                    callee_working_set);
    // Inlined code runs in-place, but only one copy regardless of frequency
    // (if called from different sites, each site gets a copy)
    d.cost_b.throughput_cycles *= call_frequency;
    d.cost_b.latency_cycles *= call_frequency;
    
    d.evaluate();
    record_decision(d);
    return d;
}

DecisionEngine::UnrollDecision DecisionEngine::should_unroll(
        const std::vector<std::string>& body_opcodes,
        uint64_t trip_count,
        uint64_t working_set_bytes,
        uint32_t max_factor) const {
    
    UnrollDecision result;
    result.should_unroll = false;
    result.optimal_factor = 1;
    result.savings_percent = 0.0;
    
    // Baseline: no unrolling
    CostResult baseline = model_.loop_cost(body_opcodes, trip_count, working_set_bytes);
    double best_cost = baseline.total();
    
    // Try each unroll factor: 2, 4, 8, 16
    for (uint32_t factor = 2; factor <= max_factor; factor *= 2) {
        if (trip_count > 0 && factor > trip_count) break;
        
        CostResult unrolled = model_.unrolled_loop_cost(
            body_opcodes, trip_count, factor, working_set_bytes);
        
        Decision d;
        d.name = "unroll_" + std::to_string(factor);
        d.description = "unroll loop by " + std::to_string(factor);
        d.cost_a = baseline;
        d.cost_b = unrolled;
        d.evaluate();
        result.factors_evaluated.push_back(d);
        
        if (unrolled.total() < best_cost) {
            best_cost = unrolled.total();
            result.optimal_factor = factor;
            result.should_unroll = true;
            result.savings_percent = d.savings_percent;
        }
    }
    
    if (result.should_unroll) {
        Decision final_d;
        final_d.name = "unroll";
        final_d.description = "unroll loop by " + std::to_string(result.optimal_factor);
        final_d.cost_a = baseline;
        final_d.cost_b = model_.unrolled_loop_cost(
            body_opcodes, trip_count, result.optimal_factor, working_set_bytes);
        final_d.evaluate();
        record_decision(final_d);
    }
    
    return result;
}

DecisionEngine::VectorizeDecision DecisionEngine::should_vectorize(
        const std::vector<std::string>& body_opcodes,
        uint64_t trip_count,
        uint64_t working_set_bytes,
        bool has_dependencies) const {
    
    VectorizeDecision result;
    result.should_vectorize = false;
    result.optimal_width = 0;
    result.savings_percent = 0.0;
    
    if (has_dependencies) {
        // Loop-carried dependencies prevent vectorization
        return result;
    }
    
    CostResult baseline = model_.loop_cost(body_opcodes, trip_count, working_set_bytes);
    double best_cost = baseline.total();
    
    // Try available SIMD widths
    std::vector<uint32_t> widths;
    if (true) widths.push_back(128);  // SSE always available on x86_64
    // Check actual SIMD capabilities would come from HWProfile
    widths.push_back(256);  // AVX2
    // if (hw has avx512) widths.push_back(512);
    
    for (uint32_t width : widths) {
        CostResult vectorized = model_.vectorized_loop_cost(
            body_opcodes, trip_count, width, working_set_bytes);
        
        if (vectorized.total() < best_cost) {
            best_cost = vectorized.total();
            result.optimal_width = width;
            result.should_vectorize = true;
            result.savings_percent = ((baseline.total() - best_cost) / baseline.total()) * 100.0;
        }
    }
    
    if (result.should_vectorize) {
        Decision d;
        d.name = "vectorize";
        d.description = "vectorize at " + std::to_string(result.optimal_width) + "-bit width";
        d.cost_a = baseline;
        d.cost_b = model_.vectorized_loop_cost(
            body_opcodes, trip_count, result.optimal_width, working_set_bytes);
        d.evaluate();
        record_decision(d);
    }
    
    return result;
}

Decision DecisionEngine::should_branchless(double taken_probability,
                                             uint32_t true_path_cost,
                                             uint32_t false_path_cost) const {
    Decision d;
    d.name = "branchless";
    d.description = "convert branch (p=" + std::to_string(taken_probability) + ") to cmov";
    
    // Option A: branch
    d.cost_a.throughput_cycles = model_.branch_cost(taken_probability);
    d.cost_a.throughput_cycles += taken_probability * true_path_cost 
                                + (1.0 - taken_probability) * false_path_cost;
    d.cost_a.latency_cycles = d.cost_a.throughput_cycles;
    d.cost_a.branch_risk = std::min(taken_probability, 1.0 - taken_probability);
    
    // Option B: cmov (always computes both, no branch)
    d.cost_b.throughput_cycles = model_.cmov_cost() + true_path_cost + false_path_cost;
    d.cost_b.latency_cycles = d.cost_b.throughput_cycles;
    d.cost_b.branch_risk = 0.0;
    
    d.evaluate();
    record_decision(d);
    return d;
}

std::vector<uint32_t> DecisionEngine::optimal_schedule(
        const std::vector<std::string>& opcodes,
        const std::vector<std::pair<uint32_t, uint32_t>>& dependencies) const {
    
    // Simple list scheduling: prioritize high-latency instructions first
    // Real implementation would use a proper DAG scheduler
    
    uint32_t n = opcodes.size();
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    
    // Build dependency depth (longest path from this node to any leaf)
    std::vector<double> depth(n, 0.0);
    
    // Topological sort + depth calculation
    std::vector<std::vector<uint32_t>> adj(n);
    std::vector<uint32_t> in_degree(n, 0);
    
    for (const auto& [from, to] : dependencies) {
        adj[from].push_back(to);
        in_degree[to]++;
    }
    
    // Process in reverse topological order to compute depths
    for (int i = n - 1; i >= 0; i--) {
        double max_child_depth = 0.0;
        for (uint32_t child : adj[i]) {
            max_child_depth = std::max(max_child_depth, depth[child]);
        }
        depth[i] = model_.instruction_cost(opcodes[i]) + max_child_depth;
    }
    
    // Sort by depth (longest path first = highest priority)
    std::sort(order.begin(), order.end(), [&depth](uint32_t a, uint32_t b) {
        return depth[a] > depth[b];
    });
    
    return order;
}

DecisionEngine::Stats DecisionEngine::stats() const {
    Stats s{};
    s.total_decisions = decisions_.size();
    
    for (const auto& d : decisions_) {
        if (d.name == "inline") {
            if (d.choose_b) s.inlines_accepted++;
            else s.inlines_rejected++;
        } else if (d.name == "unroll") {
            if (d.choose_b) s.loops_unrolled++;
        } else if (d.name == "vectorize") {
            if (d.choose_b) s.loops_vectorized++;
        } else if (d.name == "branchless") {
            if (d.choose_b) s.branches_converted++;
        }
        
        s.total_estimated_savings_percent += d.savings_percent;
    }
    
    if (s.total_decisions > 0) {
        s.total_estimated_savings_percent /= s.total_decisions;
    }
    
    return s;
}

} // namespace costforge
