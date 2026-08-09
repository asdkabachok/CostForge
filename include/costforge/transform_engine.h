#pragma once
/// CostForge TransformEngine — v0.3.0
///
/// The hands of CostForge. Where DecisionEngine is the brain that decides
/// WHAT to do, TransformEngine is the surgeon that does it.
///
/// Every transform:
///   1. Takes an LLVM IR construct (loop, branch, callsite)
///   2. Applies a specific, cost-model-driven modification
///   3. Returns a TransformResult with what changed and estimated impact
///
/// No heuristic thresholds. Each transform fires only when the
/// DecisionEngine has already proven it's profitable.

#ifdef COSTFORGE_LLVM_PASS

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

#include "costforge/types.h"
#include "costforge/decision_engine.h"
#include "costforge/pattern_recognizer.h"
#include "costforge/decision_log.h"

#include <vector>
#include <string>

namespace costforge {

/// Result of a single transform application
struct TransformResult {
    bool applied               = false;
    std::string transform_name;
    std::string location;        // function::block or function::loop
    std::string description;     // human-readable what we did
    
    double estimated_speedup    = 1.0;  // multiplier
    int32_t code_growth_bytes   = 0;    // positive = grew, negative = shrunk
    int32_t instructions_added  = 0;
    int32_t instructions_removed = 0;
    
    std::string failure_reason;  // if !applied, why
};

/// Summary of all transforms applied to one function
struct FunctionTransformReport {
    std::string function_name;
    uint32_t transforms_applied    = 0;
    uint32_t transforms_skipped    = 0;
    double combined_speedup        = 1.0;
    int32_t total_code_growth      = 0;
    std::vector<TransformResult> results;
};

/// Summary of all transforms applied to the entire module
struct ModuleTransformReport {
    uint32_t functions_modified    = 0;
    uint32_t total_transforms     = 0;
    double estimated_module_speedup = 1.0;
    int32_t total_code_growth     = 0;
    std::vector<FunctionTransformReport> functions;
    
    std::string to_json() const;
    std::string to_summary() const;
};


class TransformEngine {
public:
    TransformEngine(const DecisionEngine& decisions,
                    const PatternRecognizer& patterns,
                    DecisionLog& log,
                    const HWProfile& hw);
    
    // ── Per-function transforms ────────────────────────
    
    /// Full pipeline: run all applicable transforms on a function.
    /// Returns report of what was done.
    FunctionTransformReport transform_function(
        llvm::Function& F,
        llvm::FunctionAnalysisManager& FAM);
    
    // ── Individual transforms ──────────────────────────
    
    /// Convert simple diamond if/else to select instruction.
    /// Profitable when branch is unpredictable and both paths are cheap.
    TransformResult branch_to_select(
        llvm::BranchInst* BI,
        double taken_probability);
    
    /// Convert if/else chain (≥4 comparisons against constants)
    /// to a switch instruction → LLVM lowers to jump table.
    TransformResult branch_chain_to_switch(
        llvm::BasicBlock* entry_block,
        uint32_t chain_length);
    
    /// Set loop unroll metadata based on cost-model-optimal factor.
    /// Doesn't unroll directly — tells LLVM's unroller exactly how much.
    TransformResult set_unroll_hint(
        llvm::Loop* L,
        uint32_t optimal_factor);
    
    /// Set vectorization width + interleave count metadata.
    /// Tells LLVM's LoopVectorizer the exact width we computed.
    TransformResult set_vectorize_hint(
        llvm::Loop* L,
        uint32_t width_bits,
        uint32_t interleave_count);
    
    /// Insert llvm.prefetch intrinsic before streaming loads.
    /// For patterns: Stencil, Memcopy, Gather, StreamStore.
    TransformResult insert_prefetch(
        llvm::Instruction* load_inst,
        uint64_t stride_bytes,
        uint32_t prefetch_distance);
    
    /// Rewrite a canonical i-j-k matmul-shaped reduction loop nest
    /// (scalar accumulator over k, C[i][j] = sum_k A[i][k]*B[k][j]) into
    /// i-k-j order. Only fires when the exact shape is structurally
    /// verified (bails out, no-op, on anything unexpected) — this is a
    /// narrow, pattern-specific rewrite, not a general dependence-based
    /// loop interchange. Purpose: the j-k order makes B[k][j] (and the
    /// C[i][j] store) a large-stride access every iteration; i-k-j makes
    /// every array access in the innermost loop unit-stride, which is what
    /// actually determines performance on real hardware (confirmed via
    /// benchmark, not just modeled) — far more than any vectorize/unroll
    /// hint LLVM's own legality checks accept for this pattern.
    TransformResult interchange_matmul_ikj(
        llvm::Loop* L_i,
        llvm::LoopInfo& LI,
        llvm::ScalarEvolution& SE,
        llvm::DominatorTree& DT);
    
    /// Reorder basic blocks to put hot path first (fall-through).
    /// Uses branch probability from pattern analysis.
    TransformResult reorder_hot_path(
        llvm::Function& F,
        llvm::BranchInst* BI,
        bool true_is_hot);
    
    /// Set branch weight metadata from cost model predictions.
    /// Helps LLVM's block placement and if-conversion.
    TransformResult set_branch_weights(
        llvm::BranchInst* BI,
        uint32_t true_weight,
        uint32_t false_weight);
    
    /// Mark function as cold → __attribute__((cold)) equivalent.
    /// Moves to .text.cold section, disables aggressive inlining.
    TransformResult mark_cold_function(llvm::Function& F);
    
    /// Mark function as hot → __attribute__((hot)) equivalent.
    /// Aggressive inlining, aligned entry, hot section.
    TransformResult mark_hot_function(llvm::Function& F);
    
    /// Apply IPA inline decision: actually inline a callsite.
    TransformResult apply_inline(
        llvm::CallInst* CI,
        const InlinePlan::Site& plan_site);
    
    // ── Module-level transforms ────────────────────────
    
    /// Apply the full inline plan from InterproceduralAnalyzer.
    /// Returns per-site results.
    std::vector<TransformResult> apply_inline_plan(
        llvm::Module& M,
        const InlinePlan& plan);
    
    /// Mark outline candidates as cold.
    std::vector<TransformResult> apply_outline_hints(
        llvm::Module& M,
        const std::vector<std::string>& cold_functions);
    
    // ── Stats ──────────────────────────────────────────
    
    uint32_t total_transforms_applied() const { return transforms_applied_; }
    
private:
    const DecisionEngine& decisions_;
    const PatternRecognizer& patterns_;
    DecisionLog& log_;
    const HWProfile& hw_;
    
    uint32_t transforms_applied_ = 0;
    
    // Helpers
    void record_transform(const TransformResult& result);
    
    /// Attach loop metadata node
    static void set_loop_metadata(llvm::Loop* L,
                                  llvm::StringRef key,
                                  llvm::Constant* value);
    
    /// Check if a diamond pattern is simple enough for select conversion
    static bool is_simple_diamond(llvm::BranchInst* BI,
                                  llvm::BasicBlock*& true_bb,
                                  llvm::BasicBlock*& false_bb,
                                  llvm::BasicBlock*& merge_bb);
    
    /// Detect if-else chain structure and extract comparison values
    struct ChainLink {
        llvm::CmpInst* cmp;
        llvm::ConstantInt* const_val;
        llvm::BasicBlock* target;
        llvm::BasicBlock* next_check;
    };
    static bool extract_chain(llvm::BasicBlock* entry,
                              std::vector<ChainLink>& links,
                              llvm::BasicBlock*& default_target);
};

} // namespace costforge

#endif // COSTFORGE_LLVM_PASS
