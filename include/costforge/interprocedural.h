#pragma once

#include "costforge/types.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include <vector>
#include <string>
#include <set>

namespace costforge {

/// InterproceduralAnalyzer — whole-program intelligence.
///
/// Standard compilers make inline decisions locally: "is this function
/// small enough to inline?" Wrong question. The right questions are:
///
///   1. Is this function HOT? (called from tight loops)
///   2. Will inlining it ENABLE further optimizations in the caller?
///      (constant propagation, dead code elimination, vectorization)
///   3. Will inlining it HURT by blowing out L1i cache?
///   4. What is the GLOBAL optimal set of inlining decisions
///      that maximizes speedup under a code size budget?
///
/// This is an NP-hard knapsack problem, but a greedy benefit/cost
/// ratio heuristic gets within ~90% of optimal.
///
/// The analyzer also detects:
///   - Recursive function groups (SCCs) that can't be inlined
///   - Functions that are better outlined (cold error paths)
///   - Virtual call devirtualization opportunities

class InterproceduralAnalyzer {
public:
    InterproceduralAnalyzer(const CostModel& model, 
                            const HWProfile& hw);
    
    /// Build a call graph from function/edge descriptions
    void build_call_graph(const std::vector<CallNode>& nodes,
                          const std::vector<CallEdge>& edges);
    
    /// Compute bottom-up order (respecting SCCs)
    void compute_traversal_order();
    
    /// Detect strongly connected components (recursive groups)
    void detect_sccs();
    
    /// Compute hotness for every function
    /// Uses loop depth and call frequency propagation
    void propagate_hotness();
    
    /// Generate the optimal inline plan under code growth budget
    InlinePlan compute_inline_plan(double max_growth_factor = 2.0) const;
    
    /// Get the call graph (after build)
    const CallGraph& call_graph() const { return graph_; }
    
    /// Find functions that should be outlined (cold paths)
    /// Returns function names that are called rarely and are large
    std::vector<std::string> find_outline_candidates() const;
    
    /// Find virtual call sites that might be devirtualizable
    /// (single known callee from the call graph)
    struct DevirtCandidate {
        std::string caller;
        std::string likely_callee;
        double probability;    // how sure are we
        std::string reason;    // "only one implementation found"
    };
    std::vector<DevirtCandidate> find_devirt_candidates() const;
    
    /// Estimate the impact of inlining a specific function everywhere
    /// Returns total cycle savings across all call sites
    double estimate_inline_impact(const std::string& function_name) const;
    
private:
    const CostModel& model_;
    HWProfile hw_;
    CallGraph graph_;
    
    // Adjacency lists for graph algorithms
    std::map<std::string, std::vector<std::string>> adj_;      // caller → callees
    std::map<std::string, std::vector<std::string>> rev_adj_;   // callee → callers
    std::map<std::string, CallNode*> node_map_;
    
    // SCC detection (Tarjan's algorithm)
    struct TarjanState {
        std::map<std::string, int> index;
        std::map<std::string, int> lowlink;
        std::map<std::string, bool> on_stack;
        std::vector<std::string> stack;
        int next_index = 0;
    };
    void tarjan_visit(const std::string& v, TarjanState& state);
    
    // Inline benefit calculation
    double inline_benefit(const CallEdge& edge, 
                          const CallNode& callee,
                          const CallNode& caller) const;
    
    double inline_code_growth(const CallNode& callee,
                              uint32_t call_count) const;
};

} // namespace costforge
