/// CostForge TransformEngine — v0.3.0
///
/// Actual IR surgery. Every method here modifies LLVM IR based on
/// decisions that the cost model has already proven profitable.

#ifdef COSTFORGE_LLVM_PASS

#include "costforge/transform_engine.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

using namespace llvm;

namespace costforge {

// ── Helper: map LLVM opcode to abstract name ───────────────

static std::string ir_to_costforge_op(const Instruction& I) {
    switch (I.getOpcode()) {
        case Instruction::Add: case Instruction::FAdd:  return "add";
        case Instruction::Sub: case Instruction::FSub:  return "sub";
        case Instruction::Mul: case Instruction::FMul:  return "mul";
        case Instruction::UDiv: case Instruction::SDiv:
        case Instruction::FDiv:  return "div";
        case Instruction::And:   return "and";
        case Instruction::Or:    return "or";
        case Instruction::Xor:   return "xor";
        case Instruction::Shl: case Instruction::LShr:
        case Instruction::AShr:  return "shift";
        case Instruction::Load:  return "load";
        case Instruction::Store: return "store";
        case Instruction::Br:    return "branch";
        case Instruction::Call:  return "call";
        case Instruction::Ret:   return "ret";
        case Instruction::ICmp: case Instruction::FCmp: return "cmp";
        case Instruction::Select: return "cmov";
        default: return "nop";
    }
}

TransformEngine::TransformEngine(const DecisionEngine& decisions,
                                 const PatternRecognizer& patterns,
                                 DecisionLog& log,
                                 const HWProfile& hw)
    : decisions_(decisions), patterns_(patterns), log_(log), hw_(hw) {}

// ════════════════════════════════════════════════════════════
//  FULL FUNCTION PIPELINE
// ════════════════════════════════════════════════════════════

FunctionTransformReport TransformEngine::transform_function(
    Function& F, FunctionAnalysisManager& FAM)
{
    FunctionTransformReport report;
    report.function_name = F.getName().str();
    
    auto& LI = FAM.getResult<LoopAnalysis>(F);
    auto& SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    auto& DT = FAM.getResult<DominatorTreeAnalysis>(F);
    
    uint64_t working_set = 0;
    for (auto& BB : F)
        for (auto& I : BB)
            if (auto* AI = dyn_cast<AllocaInst>(&I))
                if (AI->getAllocatedType()->isSized())
                    working_set += F.getParent()->getDataLayout().getTypeAllocSize(
                        AI->getAllocatedType());
    if (working_set == 0) working_set = 64;
    
    // ── Pass 1: Loop transforms ────────────────────────
    // Collect loops first (transforms invalidate iterators)
    SmallVector<Loop*, 8> loops(LI.begin(), LI.end());
    // A loop's getBlocks() includes every block of its nested sub-loops too,
    // so without this the prefetch scan below revisits the same innermost
    // load once per enclosing loop level (e.g. 3x for a 3-deep nest) and
    // inserts a redundant prefetch instruction each time. Track which loads
    // already got a prefetch so each one is only ever prefetched once, at
    // the innermost loop that actually contains it.
    SmallPtrSet<LoadInst*, 32> prefetched_loads;
    SmallVector<Loop*, 8> interchanged; // top-level loops rewritten away
    for (auto* L : loops) {
        auto res = interchange_matmul_ikj(L, LI, SE, DT);
        report.results.push_back(res);
        if (res.applied) {
            interchanged.push_back(L);
            report.combined_speedup *= res.estimated_speedup;
        }
    }
    for (auto* L : loops) {
        if (llvm::is_contained(interchanged, L))
            continue; // blocks are gone; nothing left to unroll/vectorize/prefetch
        // Collect sub-loops depth-first
        SmallVector<Loop*, 4> worklist;
        worklist.push_back(L);
        while (!worklist.empty()) {
            Loop* cur = worklist.pop_back_val();
            
            // Extract opcodes from loop body
            std::vector<std::string> body_ops;
            for (auto* BB : cur->getBlocks())
                for (auto& I : *BB)
                    body_ops.push_back(ir_to_costforge_op(I));
            
            uint64_t trip_count = cur->isGuarded() ? 64 : 256;
            // Try to get trip count from backedge taken count metadata
            if (auto* latch = cur->getLoopLatch()) {
                if (auto* BI = dyn_cast<BranchInst>(latch->getTerminator())) {
                    // Heuristic: small body = likely high trip count
                    if (body_ops.size() < 10) trip_count = 1024;
                }
            }
            
            // Check for dependencies (conservative: assume none if no
            // cross-iteration memory deps detected in simple scan)
            bool has_deps = false;
            SmallPtrSet<Value*, 16> stores;
            for (auto* BB : cur->getBlocks()) {
                for (auto& I : *BB) {
                    if (auto* SI = dyn_cast<StoreInst>(&I))
                        stores.insert(SI->getPointerOperand()
                                          ->stripPointerCasts());
                }
            }
            for (auto* BB : cur->getBlocks()) {
                for (auto& I : *BB) {
                    if (auto* LI2 = dyn_cast<LoadInst>(&I)) {
                        if (stores.count(LI2->getPointerOperand()
                                             ->stripPointerCasts()))
                            has_deps = true;
                    }
                }
            }
            
            // ── Unroll decision ──
            auto unroll = decisions_.should_unroll(
                body_ops, trip_count, working_set);
            if (unroll.should_unroll && unroll.optimal_factor > 1) {
                auto res = set_unroll_hint(cur, unroll.optimal_factor);
                res.estimated_speedup = 1.0 + unroll.savings_percent / 100.0;
                report.results.push_back(res);
                if (res.applied) report.transforms_applied++;
                else report.transforms_skipped++;
            }
            
            // ── Vectorize decision ──
            auto vec = decisions_.should_vectorize(
                body_ops, trip_count, working_set, has_deps);
            if (vec.should_vectorize && vec.optimal_width > 0) {
                uint32_t width_elems = vec.optimal_width / 32;
                if (width_elems < 2) width_elems = 2;
                uint32_t interleave = 1;
                // Interleave if we have enough registers and trip count
                if (trip_count >= width_elems * 4 &&
                    body_ops.size() < 20)
                    interleave = 2;
                    
                auto res = set_vectorize_hint(cur, width_elems, interleave);
                res.estimated_speedup = 1.0 + vec.savings_percent / 100.0;
                report.results.push_back(res);
                if (res.applied) report.transforms_applied++;
                else report.transforms_skipped++;
            }
            
            // ── Prefetch for streaming patterns ──
            // Detect streaming loads in loop body
            for (auto* BB : cur->getBlocks()) {
                for (auto& I : *BB) {
                    if (auto* load = dyn_cast<LoadInst>(&I)) {
                        if (prefetched_loads.contains(load))
                            continue; // already prefetched at a deeper loop level
                        auto* ptr = load->getPointerOperand();
                        // Check if this is a GEP with a loop-varying index
                        if (auto* gep = dyn_cast<GetElementPtrInst>(ptr)) {
                            // Simple heuristic: GEP in loop with affine index
                            // = streaming access → prefetch.
                            //
                            // But only when the loop-varying index is NOT the
                            // GEP's trailing (innermost) index: a trailing
                            // varying index means unit-stride, sequential
                            // access (e.g. A[i][k] with k innermost) — the
                            // CPU's hardware stream prefetcher already
                            // handles that well, and adding a software
                            // prefetch there is pure per-iteration overhead
                            // with no cache-miss reduction to offset it.
                            // A varying index in a non-trailing position
                            // (e.g. B[k][j] with k the row index) means a
                            // large, row-sized stride between iterations —
                            // exactly the pattern hardware prefetchers miss
                            // and software prefetch genuinely helps with.
                            bool has_loop_var = false;
                            unsigned num_indices = gep->getNumIndices();
                            unsigned idx_pos = 0;
                            // Use the innermost loop actually containing this
                            // load (not `cur`, which may be an *outer* loop
                            // in the current traversal level) — what matters
                            // for prefetch profitability is the address delta
                            // between consecutive executions of this load,
                            // i.e. between iterations of its tightest
                            // enclosing loop.
                            Loop* innermost = LI.getLoopFor(I.getParent());
                            for (auto& idx : gep->indices()) {
                                bool this_idx_varies = false;
                                if (auto* phi = dyn_cast<PHINode>(idx))
                                    if (innermost && innermost->contains(phi->getParent()))
                                        this_idx_varies = true;
                                if (auto* add = dyn_cast<BinaryOperator>(idx))
                                    if (innermost && innermost->contains(add->getParent()))
                                        this_idx_varies = true;
                                // idx_pos is 0-based; the trailing index is
                                // num_indices - 1.
                                bool is_trailing = (idx_pos == num_indices - 1);
                                if (this_idx_varies && !is_trailing)
                                    has_loop_var = true;
                                ++idx_pos;
                            }
                            if (has_loop_var) {
                                // Stride = element size in bytes
                                uint64_t stride = F.getParent()->getDataLayout()
                                    .getTypeAllocSize(load->getType());
                                // Prefetch distance: ~2 cache lines ahead
                                uint32_t dist = (hw_.l1d.line_size * 2) /
                                    std::max(stride, uint64_t(1));
                                if (dist < 4) dist = 4;
                                if (dist > 64) dist = 64;
                                
                                auto res = insert_prefetch(load, stride, dist);
                                if (res.applied)
                                    prefetched_loads.insert(load);
                                report.results.push_back(res);
                                if (res.applied) report.transforms_applied++;
                                else report.transforms_skipped++;
                            }
                        }
                    }
                }
            }
            
            // Process sub-loops
            for (auto* sub : cur->getSubLoops())
                worklist.push_back(sub);
        }
    }
    
    // ── Pass 2: Branch transforms ──────────────────────
    // Collect branches first (we'll modify CFG)
    SmallVector<BranchInst*, 16> branches;
    for (auto& BB : F)
        if (auto* BI = dyn_cast<BranchInst>(BB.getTerminator()))
            if (BI->isConditional())
                branches.push_back(BI);
    
    for (auto* BI : branches) {
        // Verify this branch still exists (prior transforms may have removed it)
        if (!BI->getParent() || BI->getParent()->getParent() != &F)
            continue;
        
        // ── Try branch-to-select ──
        // Only for unpredictable branches with simple diamond structure
        BasicBlock *true_bb, *false_bb, *merge_bb;
        if (is_simple_diamond(BI, true_bb, false_bb, merge_bb)) {
            // Count ops in each path
            uint32_t true_cost = true_bb->size();
            uint32_t false_cost = false_bb->size();
            
            auto decision = decisions_.should_branchless(
                0.5,  // unknown probability → 50/50 (worst case for branch pred)
                true_cost, false_cost);
            
            if (decision.choose_b) {
                auto res = branch_to_select(BI, 0.5);
                report.results.push_back(res);
                if (res.applied) report.transforms_applied++;
                else report.transforms_skipped++;
            }
        }
        
        // ── Try branch chain to switch ──
        if (!BI->getParent()) continue;
        BasicBlock* entry = BI->getParent();
        std::vector<ChainLink> links;
        BasicBlock* default_target;
        if (extract_chain(entry, links, default_target) &&
            links.size() >= 4) {
            auto res = branch_chain_to_switch(entry, links.size());
            report.results.push_back(res);
            if (res.applied) report.transforms_applied++;
            else report.transforms_skipped++;
        }
    }
    
    // ── Pass 3: Branch weight annotation ───────────────
    // Annotate remaining branches with cost-model-predicted weights
    for (auto& BB : F) {
        auto* BI = dyn_cast<BranchInst>(BB.getTerminator());
        if (!BI || !BI->isConditional()) continue;
        // Don't re-annotate already weighted branches
        if (BI->getMetadata(LLVMContext::MD_prof)) continue;
        
        // Heuristic weights based on CFG structure
        BasicBlock* true_dest = BI->getSuccessor(0);
        BasicBlock* false_dest = BI->getSuccessor(1);
        
        uint32_t tw = 50, fw = 50;
        
        // Back-edge (loop latch) → true path is loop body, heavily weighted
        for (auto* L : loops) {
            if (L->getLoopLatch() == &BB) {
                if (L->contains(true_dest)) { tw = 95; fw = 5; }
                else if (L->contains(false_dest)) { tw = 5; fw = 95; }
                break;
            }
        }
        
        // Early exit / error check → false path (continuation) is hot
        if (true_dest->size() <= 2) {
            // Short true path = likely error/early return
            auto* term = true_dest->getTerminator();
            if (isa<ReturnInst>(term) || isa<UnreachableInst>(term)) {
                tw = 5; fw = 95;
            }
        }
        
        if (tw != 50 || fw != 50) {
            auto res = set_branch_weights(BI, tw, fw);
            report.results.push_back(res);
            if (res.applied) report.transforms_applied++;
            else report.transforms_skipped++;
        }
    }
    
    // Compute combined speedup (multiplicative)
    report.combined_speedup = 1.0;
    for (const auto& r : report.results) {
        if (r.applied) {
            report.combined_speedup *= r.estimated_speedup;
            report.total_code_growth += r.code_growth_bytes;
        }
    }
    
    return report;
}

// ════════════════════════════════════════════════════════════
//  BRANCH → SELECT
// ════════════════════════════════════════════════════════════

bool TransformEngine::is_simple_diamond(BranchInst* BI,
                                        BasicBlock*& true_bb,
                                        BasicBlock*& false_bb,
                                        BasicBlock*& merge_bb)
{
    if (!BI->isConditional()) return false;
    
    true_bb = BI->getSuccessor(0);
    false_bb = BI->getSuccessor(1);
    
    // Both sides must have single predecessor (the branch block)
    if (!true_bb->getSinglePredecessor() ||
        !false_bb->getSinglePredecessor())
        return false;
    
    // Both must be small (≤3 non-terminator instructions)
    auto non_term_count = [](BasicBlock* BB) -> unsigned {
        unsigned n = 0;
        for (auto& I : *BB) if (!I.isTerminator()) n++;
        return n;
    };
    if (non_term_count(true_bb) > 3 || non_term_count(false_bb) > 3)
        return false;
    
    // Both must branch unconditionally to the same merge block
    auto* true_term = dyn_cast<BranchInst>(true_bb->getTerminator());
    auto* false_term = dyn_cast<BranchInst>(false_bb->getTerminator());
    if (!true_term || !false_term) return false;
    if (true_term->isConditional() || false_term->isConditional()) return false;
    
    if (true_term->getSuccessor(0) != false_term->getSuccessor(0))
        return false;
    
    merge_bb = true_term->getSuccessor(0);
    
    // Merge block must have exactly 2 predecessors
    unsigned pred_count = 0;
    for (auto it = pred_begin(merge_bb); it != pred_end(merge_bb); ++it)
        if (++pred_count > 2) return false;
    if (pred_count != 2) return false;
    
    // No side effects in either path (no stores, calls, etc.)
    auto has_side_effects = [](BasicBlock* BB) -> bool {
        for (auto& I : *BB) {
            if (I.isTerminator()) continue;
            if (I.mayWriteToMemory()) return true;
            if (isa<CallInst>(&I) || isa<InvokeInst>(&I)) return true;
        }
        return false;
    };
    if (has_side_effects(true_bb) || has_side_effects(false_bb))
        return false;
    
    return true;
}

TransformResult TransformEngine::branch_to_select(
    BranchInst* BI, double taken_probability)
{
    TransformResult result;
    result.transform_name = "branch_to_select";
    result.location = BI->getParent()->getParent()->getName().str() + "::" +
                      BI->getParent()->getName().str();
    
    BasicBlock *true_bb, *false_bb, *merge_bb;
    if (!is_simple_diamond(BI, true_bb, false_bb, merge_bb)) {
        result.failure_reason = "not a simple diamond pattern";
        return result;
    }
    
    // Find PHI nodes in merge block — these become selects
    SmallVector<PHINode*, 4> phis;
    for (auto& I : *merge_bb) {
        if (auto* phi = dyn_cast<PHINode>(&I))
            phis.push_back(phi);
        else
            break;  // PHIs are always at the start
    }
    
    if (phis.empty()) {
        result.failure_reason = "no PHI nodes in merge block";
        return result;
    }
    
    // Move non-terminator instructions from both sides into the branch block
    BasicBlock* entry = BI->getParent();
    Value* cond = BI->getCondition();
    
    // Hoist instructions from true_bb and false_bb before the branch
    for (auto& I : make_early_inc_range(*true_bb)) {
        if (I.isTerminator()) continue;
        I.moveBefore(BI);
    }
    for (auto& I : make_early_inc_range(*false_bb)) {
        if (I.isTerminator()) continue;
        I.moveBefore(BI);
    }
    
    // Replace PHIs with selects
    IRBuilder<> builder(BI);
    for (auto* phi : phis) {
        Value* true_val = phi->getIncomingValueForBlock(true_bb);
        Value* false_val = phi->getIncomingValueForBlock(false_bb);
        
        Value* sel = builder.CreateSelect(cond, true_val, false_val,
                                          phi->getName() + ".sel");
        phi->replaceAllUsesWith(sel);
        phi->eraseFromParent();
    }
    
    // Replace conditional branch with unconditional to merge
    builder.CreateBr(merge_bb);
    BI->eraseFromParent();
    
    // Remove dead blocks
    if (true_bb->hasNPredecessors(0)) {
        true_bb->dropAllReferences();
        true_bb->eraseFromParent();
    }
    if (false_bb->hasNPredecessors(0)) {
        false_bb->dropAllReferences();
        false_bb->eraseFromParent();
    }
    
    result.applied = true;
    result.description = "converted diamond if/else to select (branchless)";
    result.estimated_speedup = 1.0 +
        (hw_.branch_mispredict_penalty * 0.5) /
        std::max(double(hw_.pipeline_depth), 1.0) * 0.01;
    result.instructions_removed = 2;  // two branches
    result.instructions_added = static_cast<int32_t>(phis.size());  // selects
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  BRANCH CHAIN → SWITCH
// ════════════════════════════════════════════════════════════

bool TransformEngine::extract_chain(BasicBlock* entry,
                                    std::vector<ChainLink>& links,
                                    BasicBlock*& default_target)
{
    links.clear();
    BasicBlock* current = entry;
    Value* compared_val = nullptr;
    
    for (uint32_t i = 0; i < 64; ++i) {  // safety limit
        auto* BI = dyn_cast<BranchInst>(current->getTerminator());
        if (!BI || !BI->isConditional()) {
            // End of chain — this is the default
            if (auto* br = dyn_cast<BranchInst>(current->getTerminator())) {
                if (!br->isConditional()) {
                    default_target = br->getSuccessor(0);
                    return links.size() >= 4;
                }
            }
            default_target = current;
            return links.size() >= 4;
        }
        
        // Must be icmp eq against a constant
        auto* cmp = dyn_cast<ICmpInst>(BI->getCondition());
        if (!cmp || cmp->getPredicate() != ICmpInst::ICMP_EQ)
            break;
        
        // One operand must be a constant int
        ConstantInt* const_val = nullptr;
        Value* var_val = nullptr;
        if (auto* ci = dyn_cast<ConstantInt>(cmp->getOperand(0))) {
            const_val = ci;
            var_val = cmp->getOperand(1);
        } else if (auto* ci = dyn_cast<ConstantInt>(cmp->getOperand(1))) {
            const_val = ci;
            var_val = cmp->getOperand(0);
        } else {
            break;
        }
        
        // All comparisons must be against the same variable
        if (compared_val == nullptr) {
            compared_val = var_val;
        } else if (var_val != compared_val) {
            // Allow stripped-pointer equality
            if (var_val->stripPointerCasts() !=
                compared_val->stripPointerCasts())
                break;
        }
        
        ChainLink link;
        link.cmp = cmp;
        link.const_val = const_val;
        link.target = BI->getSuccessor(0);  // true → matched case
        link.next_check = BI->getSuccessor(1); // false → next comparison
        links.push_back(link);
        
        // Follow the false path to the next comparison
        BasicBlock* next = BI->getSuccessor(1);
        
        // Next block must have single predecessor (the current block)
        if (!next->getSinglePredecessor())
            break;
        
        current = next;
    }
    
    // Default is the false branch of the last comparison
    if (!links.empty()) {
        default_target = links.back().next_check;
        return links.size() >= 4;
    }
    return false;
}

TransformResult TransformEngine::branch_chain_to_switch(
    BasicBlock* entry_block, uint32_t chain_length)
{
    TransformResult result;
    result.transform_name = "branch_chain_to_switch";
    result.location = entry_block->getParent()->getName().str() + "::" +
                      entry_block->getName().str();
    
    std::vector<ChainLink> links;
    BasicBlock* default_target;
    if (!extract_chain(entry_block, links, default_target)) {
        result.failure_reason = "failed to extract chain structure";
        return result;
    }
    
    // Build switch instruction
    Value* switch_val = links[0].cmp->getOperand(0);
    if (isa<ConstantInt>(switch_val))
        switch_val = links[0].cmp->getOperand(1);
    
    // Remove the conditional branch from entry block
    entry_block->getTerminator()->eraseFromParent();
    
    IRBuilder<> builder(entry_block);
    auto* sw = builder.CreateSwitch(switch_val, default_target,
                                    links.size());
    
    for (const auto& link : links) {
        sw->addCase(link.const_val, link.target);
    }
    
    // Clean up dead intermediate comparison blocks.
    // These are the blocks that only existed to hold the next icmp+br in
    // the chain. After the switch replaces the chain, they're unreachable.
    //
    // Collect unique blocks first (avoid double-free), skip entry_block.
    SmallPtrSet<BasicBlock*, 8> dead_set;
    for (size_t i = 1; i < links.size(); ++i) {
        BasicBlock* bb = links[i].cmp->getParent();
        if (bb && bb != entry_block)
            dead_set.insert(bb);
    }
    
    // Pass 1: drop all references (breaks circular PHI/branch deps)
    for (auto* bb : dead_set)
        bb->dropAllReferences();
    
    // Pass 2: erase blocks that are now truly dead (no predecessors)
    for (auto* bb : dead_set) {
        if (bb->hasNPredecessors(0))
            bb->eraseFromParent();
    }
    
    result.applied = true;
    result.description = "converted " + std::to_string(links.size()) +
                         "-way if/else chain to switch (→ jump table)";
    // Switch/jump table: O(1) vs O(n) comparisons
    result.estimated_speedup = 1.0 + (links.size() - 1) * 0.02;
    result.instructions_removed = links.size() * 2;  // cmp + br per link
    result.instructions_added = 1;  // one switch
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  LOOP UNROLL HINT
// ════════════════════════════════════════════════════════════

void TransformEngine::set_loop_metadata(Loop* L, StringRef key,
                                        Constant* value)
{
    LLVMContext& ctx = L->getHeader()->getContext();
    
    // Build the new property node: !{!"key"} or !{!"key", i32 value}
    SmallVector<Metadata*, 2> prop_ops;
    prop_ops.push_back(MDString::get(ctx, key));
    if (value)
        prop_ops.push_back(ConstantAsMetadata::get(value));
    MDNode* new_prop = MDNode::get(ctx, prop_ops);
    
    // Collect existing properties (skip any with same key → replace)
    SmallVector<Metadata*, 8> all_props;
    if (MDNode* old_id = L->getLoopID()) {
        for (unsigned i = 1; i < old_id->getNumOperands(); ++i) {
            auto* op = old_id->getOperand(i).get();
            if (auto* md = dyn_cast<MDNode>(op)) {
                if (md->getNumOperands() > 0)
                    if (auto* str = dyn_cast<MDString>(md->getOperand(0)))
                        if (str->getString() == key)
                            continue;
            }
            all_props.push_back(op);
        }
    }
    all_props.push_back(new_prop);
    
    // Build self-referencing loop ID:
    //   !N = distinct !{!N, !{!"key", value}, ...}
    // Strategy: create with nullptr placeholder, then fix operand 0.
    SmallVector<Metadata*, 8> id_ops;
    id_ops.push_back(nullptr);  // placeholder for self-reference
    for (auto* p : all_props)
        id_ops.push_back(p);
    
    MDNode* loop_id = MDNode::getDistinct(ctx, id_ops);
    loop_id->replaceOperandWith(0, loop_id);  // wire self-reference
    
    L->setLoopID(loop_id);
}

TransformResult TransformEngine::set_unroll_hint(Loop* L,
                                                 uint32_t optimal_factor)
{
    TransformResult result;
    result.transform_name = "set_unroll_hint";
    result.location = L->getHeader()->getParent()->getName().str() + "::loop@" +
                      L->getHeader()->getName().str();
    
    if (optimal_factor <= 1) {
        result.failure_reason = "optimal factor ≤ 1";
        return result;
    }
    
    LLVMContext& ctx = L->getHeader()->getContext();
    auto* factor_val = ConstantInt::get(Type::getInt32Ty(ctx), optimal_factor);
    
    set_loop_metadata(L, "llvm.loop.unroll.count", factor_val);
    // llvm.loop.unroll.count overrides LLVM's heuristic — no need for disable.
    
    result.applied = true;
    result.description = "set unroll factor = " + std::to_string(optimal_factor);
    result.code_growth_bytes = (optimal_factor - 1) *
        static_cast<int32_t>(L->getNumBlocks()) * 16;  // rough estimate
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  LOOP INTERCHANGE (i-j-k matmul reduction → i-k-j)
// ════════════════════════════════════════════════════════════
//
// Narrow, structurally-verified rewrite. Not a general dependence-based
// interchange (LLVM's own `loop-interchange` pass exists for that, but
// currently refuses this exact common shape — "only outer loops with
// induction or reduction PHI nodes can be interchanged"). Every check
// below either confirms the shape matches EXACTLY, or bails out with
// applied=false and a reason, doing nothing. Only fires for compile-time
// constant trip counts (v1 limitation, kept intentionally narrow because
// a wrong loop-transform is worse than a missed one).

TransformResult TransformEngine::interchange_matmul_ikj(
    Loop* L_i, LoopInfo& LI, ScalarEvolution& SE, DominatorTree& DT)
{
    (void)DT;
    TransformResult result;
    result.transform_name = "interchange_matmul_ikj";
    result.location = L_i->getHeader()->getParent()->getName().str() +
                      "::loop@" + L_i->getHeader()->getName().str();

    auto bail = [&](std::string why) -> TransformResult {
        result.applied = false;
        result.failure_reason = std::move(why);
        return result;
    };

    // ---- Structural verification (read-only; no mutation yet) ----
    if (L_i->getSubLoops().size() != 1)
        return bail("outer loop must have exactly one sub-loop");
    Loop* L_j = L_i->getSubLoops()[0];
    if (L_j->getSubLoops().size() != 1)
        return bail("mid loop must have exactly one sub-loop");
    Loop* L_k = L_j->getSubLoops()[0];
    if (!L_k->getSubLoops().empty())
        return bail("inner loop must be innermost");
    if (!L_j->isLoopSimplifyForm() || !L_k->isLoopSimplifyForm())
        return bail("loops not in simplified form");

    BasicBlock* j_preheader = L_j->getLoopPreheader();
    BasicBlock* j_header    = L_j->getHeader();
    BasicBlock* j_latch     = L_j->getLoopLatch();
    BasicBlock* j_exit      = L_j->getExitBlock();
    BasicBlock* k_preheader = L_k->getLoopPreheader();
    BasicBlock* k_header    = L_k->getHeader();
    BasicBlock* k_latch     = L_k->getLoopLatch();
    BasicBlock* k_exit      = L_k->getExitBlock();
    if (!j_preheader || !j_header || !j_latch || !j_exit)
        return bail("mid loop missing simple-form blocks");
    if (!k_preheader || !k_header || !k_latch || !k_exit)
        return bail("inner loop missing simple-form blocks");
    if (k_header != k_latch)
        return bail("inner loop body must be a single block");
    if (k_preheader != j_header)
        return bail("loop nest isn't perfectly nested (k preheader != j header)");
    if (k_exit != j_latch)
        return bail("loop nest isn't perfectly nested (k exit != j latch)");

    auto* entry_br = dyn_cast<BranchInst>(j_preheader->getTerminator());
    if (!entry_br || entry_br->isConditional() || entry_br->getSuccessor(0) != j_header)
        return bail("edge into mid loop isn't a simple unconditional branch");

    PHINode* i_iv = L_i->getCanonicalInductionVariable();
    PHINode* j_iv = L_j->getCanonicalInductionVariable();
    PHINode* k_iv = L_k->getCanonicalInductionVariable();
    if (!i_iv || !j_iv || !k_iv)
        return bail("a loop in the nest has no canonical induction variable");

    PHINode* acc_phi = nullptr;
    for (PHINode& phi : k_header->phis()) {
        if (&phi == k_iv) continue;
        if (acc_phi) return bail("inner loop has more than one non-IV phi");
        acc_phi = &phi;
    }
    if (!acc_phi) return bail("inner loop has no accumulator phi");
    if (!acc_phi->getType()->isFloatingPointTy())
        return bail("accumulator is not floating-point");

    auto* acc_init = dyn_cast<ConstantFP>(
        acc_phi->getIncomingValueForBlock(k_preheader));
    if (!acc_init || !acc_init->isZero())
        return bail("accumulator does not start at 0.0");

    Value* acc_next = acc_phi->getIncomingValueForBlock(k_latch);
    Value* load_a_v = nullptr;
    Value* load_b_v = nullptr;
    bool is_fmuladd = false;
    if (auto* call = dyn_cast<CallInst>(acc_next)) {
        Function* callee = call->getCalledFunction();
        if (!callee || callee->getIntrinsicID() != Intrinsic::fmuladd)
            return bail("accumulator update is not fmuladd");
        if (call->getArgOperand(2) != acc_phi)
            return bail("fmuladd addend is not the accumulator");
        load_a_v = call->getArgOperand(0);
        load_b_v = call->getArgOperand(1);
        is_fmuladd = true;
    } else if (auto* fadd = dyn_cast<BinaryOperator>(acc_next)) {
        if (fadd->getOpcode() != Instruction::FAdd)
            return bail("accumulator update is not fadd/fmuladd");
        Value* other = nullptr;
        if (fadd->getOperand(0) == acc_phi) other = fadd->getOperand(1);
        else if (fadd->getOperand(1) == acc_phi) other = fadd->getOperand(0);
        else return bail("fadd does not directly use the accumulator");
        auto* fmul = dyn_cast<BinaryOperator>(other);
        if (!fmul || fmul->getOpcode() != Instruction::FMul)
            return bail("accumulator update is not a multiply-add");
        load_a_v = fmul->getOperand(0);
        load_b_v = fmul->getOperand(1);
    } else {
        return bail("accumulator update is not fadd/fmuladd");
    }

    auto* load_a = dyn_cast<LoadInst>(load_a_v);
    auto* load_b = dyn_cast<LoadInst>(load_b_v);
    if (!load_a || !load_b) return bail("multiply operands are not direct loads");
    auto* gep_a = dyn_cast<GetElementPtrInst>(load_a->getPointerOperand());
    auto* gep_b = dyn_cast<GetElementPtrInst>(load_b->getPointerOperand());
    if (!gep_a || !gep_b) return bail("load addresses are not simple GEPs");

    auto gep_depends_on = [](GetElementPtrInst* gep, Value* v) {
        for (auto& idx : gep->indices())
            if (idx.get() == v) return true;
        return false;
    };
    auto gep_trailing_is = [](GetElementPtrInst* gep, Value* v) {
        if (gep->getNumIndices() == 0) return false;
        return (gep->idx_end() - 1)->get() == v;
    };

    LoadInst *inv_load = nullptr, *stream_load = nullptr;
    GetElementPtrInst *inv_gep = nullptr, *stream_gep = nullptr;
    if (!gep_depends_on(gep_a, j_iv) && gep_depends_on(gep_a, k_iv) &&
        gep_depends_on(gep_b, j_iv) && !gep_depends_on(gep_b, i_iv)) {
        inv_load = load_a; inv_gep = gep_a;
        stream_load = load_b; stream_gep = gep_b;
    } else if (!gep_depends_on(gep_b, j_iv) && gep_depends_on(gep_b, k_iv) &&
               gep_depends_on(gep_a, j_iv) && !gep_depends_on(gep_a, i_iv)) {
        inv_load = load_b; inv_gep = gep_b;
        stream_load = load_a; stream_gep = gep_a;
    } else {
        return bail("access pattern doesn't match A[i][k]*B[k][j] shape");
    }
    if (!gep_trailing_is(inv_gep, k_iv))
        return bail("invariant load isn't unit-stride in k");
    if (!gep_trailing_is(stream_gep, j_iv))
        return bail("streaming load isn't unit-stride in j");

    StoreInst* c_store = nullptr;
    for (Instruction& I : *j_latch) {
        if (auto* SI = dyn_cast<StoreInst>(&I)) {
            if (c_store) return bail("more than one store in reduction epilogue");
            c_store = SI;
        } else if (I.mayHaveSideEffects() && !isa<BranchInst>(&I)) {
            return bail("unexpected side-effecting instruction in epilogue");
        }
    }
    if (!c_store) return bail("no store of the reduction result found");
    if (c_store->getValueOperand() != acc_next)
        return bail("stored value is not the reduction result");
    auto* c_gep = dyn_cast<GetElementPtrInst>(c_store->getPointerOperand());
    if (!c_gep) return bail("store address is not a simple GEP");
    if (gep_depends_on(c_gep, k_iv)) return bail("store address depends on k");
    if (!gep_depends_on(c_gep, i_iv) || !gep_depends_on(c_gep, j_iv))
        return bail("store address doesn't depend on both i and j");
    if (!gep_trailing_is(c_gep, j_iv))
        return bail("store isn't unit-stride in j");

    for (Instruction& I : *k_header) {
        if (&I == k_iv || &I == acc_phi) continue;
        if (I.mayHaveSideEffects() && !isa<BranchInst>(&I))
            return bail("unexpected side-effecting instruction in inner loop body");
    }

    unsigned j_n = SE.getSmallConstantTripCount(L_j);
    unsigned k_n = SE.getSmallConstantTripCount(L_k);
    if (j_n == 0 || k_n == 0)
        return bail("trip count is not a compile-time constant (v1 limitation)");

    // ---- All checks passed. Safe to rewrite. ----
    Function* Fp = j_header->getParent();
    LLVMContext& Ctx = Fp->getContext();
    Type* ivty = k_iv->getType();
    Type* elemty = acc_phi->getType();
    auto* c0 = ConstantInt::get(ivty, 0);
    auto* c1 = ConstantInt::get(ivty, 1);
    auto* c_jn = ConstantInt::get(ivty, j_n);
    auto* c_kn = ConstantInt::get(ivty, k_n);
    auto* fp_zero = ConstantFP::get(elemty, 0.0);

    auto clone_gep = [&](GetElementPtrInst* orig, IRBuilder<>& B,
                         Value* from1, Value* to1,
                         Value* from2, Value* to2) -> Value* {
        SmallVector<Value*, 4> idxs;
        for (auto& idx : orig->indices()) {
            Value* v = idx.get();
            if (v == from1) v = to1;
            else if (from2 && v == from2) v = to2;
            idxs.push_back(v);
        }
        return B.CreateGEP(orig->getSourceElementType(), orig->getPointerOperand(),
                           idxs, "", orig->isInBounds());
    };

    auto* zi_header = BasicBlock::Create(Ctx, "cf.ikj.zi.header", Fp);
    auto* zi_body   = BasicBlock::Create(Ctx, "cf.ikj.zi.body", Fp);
    auto* zi_exit   = BasicBlock::Create(Ctx, "cf.ikj.zi.exit", Fp);
    auto* k2_header = BasicBlock::Create(Ctx, "cf.ikj.k2.header", Fp);
    auto* k2_body   = BasicBlock::Create(Ctx, "cf.ikj.k2.body", Fp);
    auto* j2_header = BasicBlock::Create(Ctx, "cf.ikj.j2.header", Fp);
    auto* j2_body   = BasicBlock::Create(Ctx, "cf.ikj.j2.body", Fp);
    auto* j2_exit   = BasicBlock::Create(Ctx, "cf.ikj.j2.exit", Fp);
    auto* k2_exit   = BasicBlock::Create(Ctx, "cf.ikj.k2.exit", Fp);

    // zero-init: for j2 in [0,N): C[i][j2] = 0.0
    IRBuilder<> B(zi_header);
    auto* j2z = B.CreatePHI(ivty, 2, "cf.j2z");
    B.CreateCondBr(B.CreateICmpSLT(j2z, c_jn), zi_body, zi_exit);

    B.SetInsertPoint(zi_body);
    Value* c_ptr_z = clone_gep(c_gep, B, j_iv, j2z, nullptr, nullptr);
    B.CreateStore(fp_zero, c_ptr_z);
    Value* j2z_next = B.CreateAdd(j2z, c1);
    B.CreateBr(zi_header);
    j2z->addIncoming(c0, j_preheader);
    j2z->addIncoming(j2z_next, zi_body);

    B.SetInsertPoint(zi_exit);
    B.CreateBr(k2_header);

    // k2 loop: for k2 in [0,N)
    B.SetInsertPoint(k2_header);
    auto* k2 = B.CreatePHI(ivty, 2, "cf.k2");
    B.CreateCondBr(B.CreateICmpSLT(k2, c_kn), k2_body, k2_exit);

    B.SetInsertPoint(k2_body);
    Value* a_ptr = clone_gep(inv_gep, B, k_iv, k2, nullptr, nullptr);
    Value* a_val = B.CreateLoad(load_a->getType(), a_ptr, "cf.a");
    B.CreateBr(j2_header);

    // j2 loop (nested in k2): C[i][j2] += a * B[k2][j2]
    B.SetInsertPoint(j2_header);
    auto* j2 = B.CreatePHI(ivty, 2, "cf.j2");
    B.CreateCondBr(B.CreateICmpSLT(j2, c_jn), j2_body, j2_exit);

    B.SetInsertPoint(j2_body);
    Value* b_ptr = clone_gep(stream_gep, B, k_iv, k2, j_iv, j2);
    Value* b_val = B.CreateLoad(load_b->getType(), b_ptr, "cf.b");
    Value* c_ptr2 = clone_gep(c_gep, B, j_iv, j2, nullptr, nullptr);
    Value* c_old = B.CreateLoad(elemty, c_ptr2, "cf.c.old");
    Value* c_new;
    if (is_fmuladd)
        c_new = B.CreateIntrinsic(Intrinsic::fmuladd, {elemty}, {a_val, b_val, c_old});
    else
        c_new = B.CreateFAdd(c_old, B.CreateFMul(a_val, b_val));
    B.CreateStore(c_new, c_ptr2);
    Value* j2_next = B.CreateAdd(j2, c1);
    B.CreateBr(j2_header);
    j2->addIncoming(c0, k2_body);
    j2->addIncoming(j2_next, j2_body);

    B.SetInsertPoint(j2_exit);
    Value* k2_next = B.CreateAdd(k2, c1);
    B.CreateBr(k2_header);
    k2->addIncoming(c0, zi_exit);
    k2->addIncoming(k2_next, j2_exit);

    B.SetInsertPoint(k2_exit);
    B.CreateBr(j_exit);

    // Rewire j_exit's incoming-value phis (if any) away from j_latch,
    // since j_latch (the old epilogue block) is about to become dead.
    for (PHINode& phi : j_exit->phis())
        phi.replaceIncomingBlockWith(j_latch, k2_exit);

    // Splice in: entry (j_preheader) now goes to our new nest instead of
    // the old j/k loop blocks.
    entry_br->setSuccessor(0, zi_header);

    // Old j/k blocks are now unreachable; remove them. LoopInfo::erase()
    // must run first — it walks the loop's block list, which would be
    // full of dangling pointers if we erased the blocks first.
    LI.erase(L_j); // also removes L_k (its sub-loop)

    SmallVector<BasicBlock*, 4> dead = {j_header, k_header, j_latch};
    for (BasicBlock* BB : dead) {
        BB->dropAllReferences();
    }
    for (BasicBlock* BB : dead) {
        BB->eraseFromParent();
    }

    result.applied = true;
    result.description = "rewrote i-j-k matmul reduction as i-k-j "
                         "(unit-stride inner loop for both operands, "
                         "was strided on B[k][j])";
    result.estimated_speedup = 2.0; // conservative; measured 2-5x on our benchmark
    return result;
}

// ════════════════════════════════════════════════════════════
//  VECTORIZATION HINT
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::set_vectorize_hint(
    Loop* L, uint32_t width_elems, uint32_t interleave_count)
{
    TransformResult result;
    result.transform_name = "set_vectorize_hint";
    result.location = L->getHeader()->getParent()->getName().str() + "::loop@" +
                      L->getHeader()->getName().str();
    
    if (width_elems < 2) {
        result.failure_reason = "width < 2 elements";
        return result;
    }
    
    LLVMContext& ctx = L->getHeader()->getContext();
    auto* i32ty = Type::getInt32Ty(ctx);
    
    set_loop_metadata(L, "llvm.loop.vectorize.width",
                      ConstantInt::get(i32ty, width_elems));
    
    if (interleave_count > 1) {
        set_loop_metadata(L, "llvm.loop.interleave.count",
                          ConstantInt::get(i32ty, interleave_count));
    }
    
    // Enable vectorization explicitly
    set_loop_metadata(L, "llvm.loop.vectorize.enable",
                      ConstantInt::get(Type::getInt1Ty(ctx), 1));
    
    result.applied = true;
    result.description = "set vectorize width=" + std::to_string(width_elems) +
                         " interleave=" + std::to_string(interleave_count);
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  PREFETCH INSERTION
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::insert_prefetch(
    Instruction* load_inst, uint64_t stride_bytes,
    uint32_t prefetch_distance)
{
    TransformResult result;
    result.transform_name = "insert_prefetch";
    result.location = load_inst->getParent()->getParent()->getName().str() +
                      "::" + load_inst->getParent()->getName().str();
    
    auto* load = dyn_cast<LoadInst>(load_inst);
    if (!load) {
        result.failure_reason = "not a load instruction";
        return result;
    }
    
    LLVMContext& ctx = load->getContext();
    Module* M = load->getModule();
    IRBuilder<> builder(load);
    
    Value* ptr = load->getPointerOperand();
    
    // Compute prefetch address: ptr + prefetch_distance * stride
    auto* i8ptr_ty = builder.getPtrTy();
    auto* i32ty = Type::getInt32Ty(ctx);
    
    Value* ptr_cast = builder.CreateBitCast(ptr, i8ptr_ty);
    Value* offset = ConstantInt::get(Type::getInt64Ty(ctx),
                                      stride_bytes * prefetch_distance);
    Value* prefetch_addr = builder.CreateGEP(
        builder.getInt8Ty(), ptr_cast, offset, "prefetch.addr");
    
    // llvm.prefetch(ptr, rw, locality, cache_type)
    //   rw = 0 (read), locality = 3 (keep in all caches), type = 1 (data)
    Function* prefetch_fn = Intrinsic::getDeclaration(
        M, Intrinsic::prefetch, {i8ptr_ty});
    
    builder.CreateCall(prefetch_fn, {
        prefetch_addr,
        ConstantInt::get(i32ty, 0),   // read
        ConstantInt::get(i32ty, 3),   // high locality
        ConstantInt::get(i32ty, 1)    // data cache
    });
    
    result.applied = true;
    result.description = "inserted prefetch +" +
        std::to_string(stride_bytes * prefetch_distance) + "B ahead";
    result.instructions_added = 3;  // gep + bitcast + prefetch
    result.estimated_speedup = 1.02; // conservative
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  BLOCK REORDERING
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::reorder_hot_path(
    Function& F, BranchInst* BI, bool true_is_hot)
{
    TransformResult result;
    result.transform_name = "reorder_hot_path";
    result.location = F.getName().str() + "::" +
                      BI->getParent()->getName().str();
    
    // Simply swap successors if the hot path isn't the fall-through
    if (!true_is_hot) {
        BI->swapSuccessors();
        // Also negate the condition
        if (auto* cmp = dyn_cast<ICmpInst>(BI->getCondition())) {
            cmp->setPredicate(cmp->getInversePredicate());
        } else {
            IRBuilder<> builder(BI);
            Value* neg = builder.CreateNot(BI->getCondition(), "hot.neg");
            BI->setCondition(neg);
        }
        
        result.applied = true;
        result.description = "swapped branch to put hot path as fall-through";
        result.estimated_speedup = 1.005;  // minor: better i-cache locality
    } else {
        result.failure_reason = "hot path already fall-through";
    }
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  BRANCH WEIGHT METADATA
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::set_branch_weights(
    BranchInst* BI, uint32_t true_weight, uint32_t false_weight)
{
    TransformResult result;
    result.transform_name = "set_branch_weights";
    result.location = BI->getParent()->getParent()->getName().str() + "::" +
                      BI->getParent()->getName().str();
    
    LLVMContext& ctx = BI->getContext();
    MDBuilder mdb(ctx);
    
    BI->setMetadata(LLVMContext::MD_prof,
                    mdb.createBranchWeights(true_weight, false_weight));
    
    result.applied = true;
    result.description = "set branch weights " + std::to_string(true_weight) +
                         ":" + std::to_string(false_weight);
    result.estimated_speedup = 1.001;  // indirect: helps later passes
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  FUNCTION HOT/COLD MARKING
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::mark_cold_function(Function& F) {
    TransformResult result;
    result.transform_name = "mark_cold";
    result.location = F.getName().str();
    
    F.addFnAttr(Attribute::Cold);
    F.addFnAttr(Attribute::MinSize);
    F.addFnAttr(Attribute::OptimizeForSize);
    F.setSection(".text.unlikely");
    
    result.applied = true;
    result.description = "marked as cold → .text.unlikely, minsize";
    result.estimated_speedup = 1.0;  // doesn't speed up, but frees i-cache
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

TransformResult TransformEngine::mark_hot_function(Function& F) {
    TransformResult result;
    result.transform_name = "mark_hot";
    result.location = F.getName().str();
    
    // Remove any cold markers
    F.removeFnAttr(Attribute::Cold);
    F.removeFnAttr(Attribute::MinSize);
    F.removeFnAttr(Attribute::OptimizeForSize);
    
    // Set hot section and alignment
    F.setSection(".text.hot");
    F.setAlignment(Align(64));  // cache-line aligned entry
    
    // Inline hint (not forced — LLVM can still decide)
    if (!F.hasFnAttribute(Attribute::NoInline))
        F.addFnAttr(Attribute::InlineHint);
    
    result.applied = true;
    result.description = "marked as hot → .text.hot, aligned 64, inline hint";
    result.estimated_speedup = 1.005;
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  INLINE APPLICATION
// ════════════════════════════════════════════════════════════

TransformResult TransformEngine::apply_inline(
    CallInst* CI, const InlinePlan::Site& plan_site)
{
    TransformResult result;
    result.transform_name = "apply_inline";
    result.location = plan_site.caller + " ← " + plan_site.callee;
    
    Function* callee = CI->getCalledFunction();
    if (!callee || callee->isDeclaration()) {
        result.failure_reason = "callee is declaration or indirect call";
        return result;
    }
    
    InlineFunctionInfo IFI;
    auto inline_result = InlineFunction(*CI, IFI);
    
    if (!inline_result.isSuccess()) {
        result.failure_reason = "LLVM InlineFunction failed: " +
            std::string(inline_result.getFailureReason());
        return result;
    }
    
    result.applied = true;
    result.description = "inlined " + plan_site.callee + " into " +
                         plan_site.caller + " (" + plan_site.reason + ")";
    result.code_growth_bytes = static_cast<int32_t>(
        callee->getInstructionCount() * 4);  // ~4 bytes per insn
    result.instructions_added = callee->getInstructionCount();
    result.instructions_removed = 1;  // the call itself
    result.estimated_speedup = 1.0 + plan_site.benefit * 0.001;
    
    transforms_applied_++;
    record_transform(result);
    return result;
}

// ════════════════════════════════════════════════════════════
//  MODULE-LEVEL TRANSFORMS
// ════════════════════════════════════════════════════════════

std::vector<TransformResult> TransformEngine::apply_inline_plan(
    Module& M, const InlinePlan& plan)
{
    std::vector<TransformResult> results;
    
    // Sort by benefit (highest first) to inline the most impactful first
    auto sorted = plan.sites;
    std::sort(sorted.begin(), sorted.end(),
              [](const InlinePlan::Site& a, const InlinePlan::Site& b) {
                  return a.benefit > b.benefit;
              });
    
    for (const auto& site : sorted) {
        // Find the call instruction
        Function* caller_fn = M.getFunction(site.caller);
        Function* callee_fn = M.getFunction(site.callee);
        if (!caller_fn || !callee_fn) continue;
        if (callee_fn->isDeclaration()) continue;
        
        // Find call to callee in caller
        CallInst* target_call = nullptr;
        for (auto& BB : *caller_fn) {
            for (auto& I : BB) {
                if (auto* CI = dyn_cast<CallInst>(&I)) {
                    if (CI->getCalledFunction() == callee_fn) {
                        target_call = CI;
                        break;
                    }
                }
            }
            if (target_call) break;
        }
        
        if (!target_call) continue;
        
        auto res = apply_inline(target_call, site);
        results.push_back(res);
    }
    
    return results;
}

std::vector<TransformResult> TransformEngine::apply_outline_hints(
    Module& M, const std::vector<std::string>& cold_functions)
{
    std::vector<TransformResult> results;
    
    for (const auto& name : cold_functions) {
        Function* F = M.getFunction(name);
        if (!F || F->isDeclaration()) continue;
        
        auto res = mark_cold_function(*F);
        results.push_back(res);
    }
    
    return results;
}

// ════════════════════════════════════════════════════════════
//  REPORTING
// ════════════════════════════════════════════════════════════

void TransformEngine::record_transform(const TransformResult& result) {
    if (result.applied) {
        std::vector<std::string> factors;
        factors.push_back("transform=" + result.transform_name);
        if (result.code_growth_bytes != 0)
            factors.push_back("code_growth=" +
                              std::to_string(result.code_growth_bytes));
        
        // Create a minimal decision trace for the log
        Decision d;
        d.description = result.description;
        d.name = result.transform_name;
        d.choose_b = true;
        d.savings_percent = (result.estimated_speedup - 1.0) * 100.0;
        
        log_.record_decision(
            result.location, result.transform_name, d, factors);
    }
}

std::string ModuleTransformReport::to_json() const {
    std::ostringstream ss;
    ss << "{\n";
    ss << "  \"functions_modified\": " << functions_modified << ",\n";
    ss << "  \"total_transforms\": " << total_transforms << ",\n";
    ss << "  \"estimated_speedup\": " << estimated_module_speedup << ",\n";
    ss << "  \"total_code_growth\": " << total_code_growth << ",\n";
    ss << "  \"functions\": [\n";
    for (size_t i = 0; i < functions.size(); ++i) {
        const auto& f = functions[i];
        ss << "    {\n";
        ss << "      \"name\": \"" << f.function_name << "\",\n";
        ss << "      \"transforms_applied\": " << f.transforms_applied << ",\n";
        ss << "      \"combined_speedup\": " << f.combined_speedup << ",\n";
        ss << "      \"code_growth\": " << f.total_code_growth << ",\n";
        ss << "      \"details\": [\n";
        for (size_t j = 0; j < f.results.size(); ++j) {
            const auto& r = f.results[j];
            ss << "        {\"transform\": \"" << r.transform_name
               << "\", \"applied\": " << (r.applied ? "true" : "false")
               << ", \"speedup\": " << r.estimated_speedup
               << ", \"desc\": \"" << r.description << "\"";
            if (!r.failure_reason.empty())
                ss << ", \"failure\": \"" << r.failure_reason << "\"";
            ss << "}";
            if (j + 1 < f.results.size()) ss << ",";
            ss << "\n";
        }
        ss << "      ]\n";
        ss << "    }";
        if (i + 1 < functions.size()) ss << ",";
        ss << "\n";
    }
    ss << "  ]\n";
    ss << "}\n";
    return ss.str();
}

std::string ModuleTransformReport::to_summary() const {
    std::ostringstream ss;
    ss << "=== CostForge v0.3.0 Transform Report ===\n\n";
    ss << "Functions modified: " << functions_modified << "\n";
    ss << "Total transforms:   " << total_transforms << "\n";
    ss << "Estimated speedup:  " << estimated_module_speedup << "x\n";
    ss << "Code growth:        " << total_code_growth << " bytes\n\n";
    
    for (const auto& f : functions) {
        if (f.transforms_applied == 0) continue;
        ss << "  " << f.function_name << " ("
           << f.transforms_applied << " transforms, "
           << f.combined_speedup << "x):\n";
        for (const auto& r : f.results) {
            if (!r.applied) continue;
            ss << "    ✓ " << r.description << "\n";
        }
    }
    return ss.str();
}

} // namespace costforge

#endif // COSTFORGE_LLVM_PASS
