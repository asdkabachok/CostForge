#include "costforge/interprocedural.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stack>

namespace costforge {

InterproceduralAnalyzer::InterproceduralAnalyzer(const CostModel& model,
                                                   const HWProfile& hw)
    : model_(model), hw_(hw) {}

// ── Build call graph ────────────────────────────────

void InterproceduralAnalyzer::build_call_graph(
        const std::vector<CallNode>& nodes,
        const std::vector<CallEdge>& edges) {
    graph_.nodes = nodes;
    graph_.edges = edges;
    
    // Build adjacency lists and node lookup
    adj_.clear();
    rev_adj_.clear();
    node_map_.clear();
    
    for (auto& node : graph_.nodes) {
        node_map_[node.function_name] = &node;
        adj_[node.function_name] = {};
        rev_adj_[node.function_name] = {};
    }
    
    for (const auto& edge : graph_.edges) {
        adj_[edge.caller].push_back(edge.callee);
        rev_adj_[edge.callee].push_back(edge.caller);
        
        // Mark caller as non-leaf
        if (node_map_.count(edge.caller)) {
            node_map_[edge.caller]->is_leaf = false;
        }
    }
}

// ── SCC detection (Tarjan's algorithm) ──────────────

void InterproceduralAnalyzer::detect_sccs() {
    graph_.sccs.clear();
    
    TarjanState state;
    
    for (const auto& node : graph_.nodes) {
        if (state.index.find(node.function_name) == state.index.end()) {
            tarjan_visit(node.function_name, state);
        }
    }
    
    // Mark recursive functions
    for (const auto& scc : graph_.sccs) {
        if (scc.size() > 1) {
            // Mutual recursion group
            for (const auto& fname : scc) {
                if (node_map_.count(fname)) {
                    node_map_[fname]->is_recursive = true;
                }
            }
        } else if (scc.size() == 1) {
            // Check for self-recursion
            const auto& fname = scc[0];
            for (const auto& callee : adj_[fname]) {
                if (callee == fname) {
                    if (node_map_.count(fname)) {
                        node_map_[fname]->is_recursive = true;
                    }
                }
            }
        }
    }
}

void InterproceduralAnalyzer::tarjan_visit(const std::string& v,
                                             TarjanState& state) {
    state.index[v] = state.next_index;
    state.lowlink[v] = state.next_index;
    state.next_index++;
    state.stack.push_back(v);
    state.on_stack[v] = true;
    
    for (const auto& w : adj_[v]) {
        if (state.index.find(w) == state.index.end()) {
            // w not yet visited
            tarjan_visit(w, state);
            state.lowlink[v] = std::min(state.lowlink[v], state.lowlink[w]);
        } else if (state.on_stack[w]) {
            state.lowlink[v] = std::min(state.lowlink[v], state.index[w]);
        }
    }
    
    // If v is a root node of an SCC
    if (state.lowlink[v] == state.index[v]) {
        std::vector<std::string> scc;
        std::string w;
        do {
            w = state.stack.back();
            state.stack.pop_back();
            state.on_stack[w] = false;
            scc.push_back(w);
        } while (w != v);
        
        graph_.sccs.push_back(std::move(scc));
    }
}

// ── Traversal order ─────────────────────────────────

void InterproceduralAnalyzer::compute_traversal_order() {
    detect_sccs();
    
    // Bottom-up: callees before callers (reverse post-order of DAG of SCCs)
    // SCCs from Tarjan are already in reverse topological order
    graph_.bottom_up_order.clear();
    for (const auto& scc : graph_.sccs) {
        for (const auto& fname : scc) {
            graph_.bottom_up_order.push_back(fname);
        }
    }
}

// ── Hotness propagation ─────────────────────────────

void InterproceduralAnalyzer::propagate_hotness() {
    // Initialize: leaf functions called from loops are hot
    for (auto& node : graph_.nodes) {
        node.hotness = 1.0;
    }
    
    // Propagate: callee hotness increases with call frequency and loop depth
    // Process in bottom-up order so callees are computed before callers
    for (const auto& fname : graph_.bottom_up_order) {
        auto* node = node_map_[fname];
        if (!node) continue;
        
        // Accumulate hotness from all callers
        double incoming_heat = 0.0;
        for (const auto& edge : graph_.edges) {
            if (edge.callee != fname) continue;
            
            double caller_hotness = 1.0;
            if (node_map_.count(edge.caller)) {
                caller_hotness = node_map_[edge.caller]->hotness;
            }
            
            // Loop depth multiplier: each nesting level × 10
            double loop_multiplier = 1.0;
            if (edge.is_in_loop) {
                loop_multiplier = std::pow(10.0, edge.loop_depth);
            }
            
            incoming_heat += caller_hotness * edge.estimated_freq * loop_multiplier;
        }
        
        node->hotness = std::max(node->hotness, incoming_heat);
    }
    
    // Compute depth for each node (max call chain length)
    for (auto& node : graph_.nodes) {
        node.depth = 0;
    }
    
    // Simple BFS from entry points
    for (const auto& fname : graph_.bottom_up_order) {
        auto* node = node_map_[fname];
        if (!node) continue;
        
        for (const auto& callee : adj_[fname]) {
            if (node_map_.count(callee)) {
                node_map_[callee]->depth = 
                    std::max(node_map_[callee]->depth, node->depth + 1);
            }
        }
    }
}

// ── Inline benefit/cost calculation ─────────────────

double InterproceduralAnalyzer::inline_benefit(const CallEdge& edge,
                                                 const CallNode& callee,
                                                 const CallNode& caller) const {
    // Benefit = call overhead eliminated × dynamic frequency
    // + potential for further optimization (constant propagation, etc.)
    
    double call_overhead = model_.call_overhead();
    
    // Dynamic frequency estimate
    double freq = edge.estimated_freq;
    if (edge.is_in_loop) {
        freq *= std::pow(10.0, edge.loop_depth);
    }
    
    double base_benefit = call_overhead * freq;
    
    // Bonus for small callees (likely to enable further optimization)
    if (callee.instruction_count < 20) {
        base_benefit *= 1.5; // small functions unlock optimizations in caller
    }
    
    // Bonus for leaf functions (no further call overhead inside)
    if (callee.is_leaf) {
        base_benefit *= 1.2;
    }
    
    // Penalty for high register pressure (inlining will cause spills)
    if (caller.register_pressure + callee.register_pressure > 0.8) {
        double spill_penalty = model_.register_spill_cost(
            static_cast<uint32_t>((caller.register_pressure + callee.register_pressure) * 16));
        base_benefit -= spill_penalty * freq;
    }
    
    return std::max(0.0, base_benefit);
}

double InterproceduralAnalyzer::inline_code_growth(const CallNode& callee,
                                                     uint32_t call_count) const {
    // Each inline site copies the function body
    // Minus one copy (the original function stays unless all sites are inlined)
    return static_cast<double>(callee.instruction_count) * 4.0 * call_count;
    // ~4 bytes per instruction average
}

// ── Compute optimal inline plan ─────────────────────

InlinePlan InterproceduralAnalyzer::compute_inline_plan(
        double max_growth_factor) const {
    InlinePlan plan;
    
    // Collect all candidate inline sites
    struct Candidate {
        const CallEdge* edge;
        const CallNode* callee;
        const CallNode* caller;
        double benefit;
        double growth;
        double ratio;  // benefit/growth — the greedy criterion
    };
    
    std::vector<Candidate> candidates;
    
    for (const auto& edge : graph_.edges) {
        if (edge.is_indirect) continue; // can't inline indirect calls
        
        auto callee_it = node_map_.find(edge.callee);
        auto caller_it = node_map_.find(edge.caller);
        if (callee_it == node_map_.end() || caller_it == node_map_.end()) continue;
        
        const CallNode* callee = callee_it->second;
        const CallNode* caller = caller_it->second;
        
        // Don't inline recursive functions
        if (callee->is_recursive) continue;
        
        // Don't inline very large functions (> 500 instructions)
        if (callee->instruction_count > 500) continue;
        
        double benefit = inline_benefit(edge, *callee, *caller);
        double growth = inline_code_growth(*callee, edge.call_count);
        
        if (benefit <= 0 || growth <= 0) continue;
        
        candidates.push_back({&edge, callee, caller, benefit, growth, benefit / growth});
    }
    
    // Sort by benefit/cost ratio (greedy knapsack)
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.ratio > b.ratio;
              });
    
    // Compute total code size for budget
    double total_code_bytes = 0;
    for (const auto& node : graph_.nodes) {
        total_code_bytes += node.instruction_count * 4.0;
    }
    double budget = total_code_bytes * (max_growth_factor - 1.0);
    
    // Greedily accept best candidates until budget exhausted
    double used_budget = 0;
    
    for (const auto& cand : candidates) {
        if (used_budget + cand.growth > budget) continue;
        
        InlinePlan::Site site;
        site.caller = cand.edge->caller;
        site.callee = cand.edge->callee;
        site.call_site_id = 0; // would come from LLVM IR
        site.benefit = cand.benefit;
        site.code_growth = cand.growth;
        
        // Generate human-readable reason
        std::string reason;
        if (cand.callee->is_leaf && cand.callee->instruction_count < 20) {
            reason = "tiny leaf function";
        } else if (cand.edge->is_in_loop && cand.edge->loop_depth >= 2) {
            reason = "hot call in nested loop (depth " 
                     + std::to_string(cand.edge->loop_depth) + ")";
        } else if (cand.callee->hotness > 100.0) {
            reason = "very hot callee (hotness " 
                     + std::to_string(static_cast<int>(cand.callee->hotness)) + ")";
        } else {
            reason = "positive benefit/cost ratio (" 
                     + std::to_string(static_cast<int>(cand.ratio * 1000)) + "/1000)";
        }
        site.reason = reason;
        
        plan.sites.push_back(std::move(site));
        plan.total_benefit += cand.benefit;
        plan.total_code_growth += cand.growth;
        used_budget += cand.growth;
    }
    
    return plan;
}

// ── Outline candidates ──────────────────────────────

std::vector<std::string> InterproceduralAnalyzer::find_outline_candidates() const {
    std::vector<std::string> result;
    
    for (const auto& node : graph_.nodes) {
        // Cold + large = outline candidate
        // (move to .text.cold section, reduce L1i pressure on hot path)
        if (node.hotness < 2.0 && node.instruction_count > 50) {
            result.push_back(node.function_name);
        }
    }
    
    // Sort by code size (largest cold functions first = most benefit)
    std::sort(result.begin(), result.end(),
              [this](const std::string& a, const std::string& b) {
                  auto ia = node_map_.find(a);
                  auto ib = node_map_.find(b);
                  if (ia == node_map_.end() || ib == node_map_.end()) return false;
                  return ia->second->instruction_count > ib->second->instruction_count;
              });
    
    return result;
}

// ── Devirtualization candidates ─────────────────────

std::vector<InterproceduralAnalyzer::DevirtCandidate> 
InterproceduralAnalyzer::find_devirt_candidates() const {
    std::vector<DevirtCandidate> result;
    
    for (const auto& edge : graph_.edges) {
        if (!edge.is_indirect) continue;
        
        // If we see only one callee for this indirect call site,
        // we can speculatively devirtualize with a guard
        DevirtCandidate cand;
        cand.caller = edge.caller;
        cand.likely_callee = edge.callee;
        cand.probability = 0.9; // heuristic: if only one target seen, high probability
        cand.reason = "single implementation observed in call graph";
        
        result.push_back(std::move(cand));
    }
    
    return result;
}

// ── Impact estimation ───────────────────────────────

double InterproceduralAnalyzer::estimate_inline_impact(
        const std::string& function_name) const {
    double total_savings = 0.0;
    
    auto callee_it = node_map_.find(function_name);
    if (callee_it == node_map_.end()) return 0.0;
    const CallNode* callee = callee_it->second;
    
    for (const auto& edge : graph_.edges) {
        if (edge.callee != function_name) continue;
        
        auto caller_it = node_map_.find(edge.caller);
        if (caller_it == node_map_.end()) continue;
        
        total_savings += inline_benefit(edge, *callee, *caller_it->second);
    }
    
    return total_savings;
}

} // namespace costforge
