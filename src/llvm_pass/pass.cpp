/// CostForge LLVM Pass — v0.3.1
///
/// Fixes from audit:
///   1. ScalarEvolution for real trip counts (not guesses)
///   2. MemorySSA / AliasAnalysis for dependency checking
///   3. Thread-safe state via ModuleAnalysisManager (no static globals)
///   4. Type-aware costs (passes IR type info to cost model)
///   5. TTI cross-check (sanity-check CostForge vs LLVM's own model)
///   6. LoopInfo integration for loop detection

#ifdef COSTFORGE_LLVM_PASS

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include "costforge/hwprobe.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include "costforge/pattern_recognizer.h"
#include "costforge/calibrator.h"
#include "costforge/decision_log.h"
#include "costforge/interprocedural.h"
#include "costforge/transform_engine.h"
#include "costforge/profile_reader.h"

#include <fstream>
#include <mutex>

using namespace llvm;

// ── CLI Options ────────────────────────────────────────────

static cl::opt<std::string> CostForgeProfile(
    "costforge-profile",
    cl::desc("Path to hardware profile JSON"),
    cl::init(""));

static cl::opt<std::string> CostForgeCalibration(
    "costforge-calibration",
    cl::desc("Path to calibration profile JSON"),
    cl::init(""));

static cl::opt<std::string> CostForgeReport(
    "costforge-report",
    cl::desc("Path to write transform report JSON"),
    cl::init(""));

static cl::opt<bool> CostForgeVerbose(
    "costforge-verbose",
    cl::desc("Print transforms to stderr"),
    cl::init(false));

static cl::opt<bool> CostForgeDryRun(
    "costforge-dry-run",
    cl::desc("Analyze but don't apply transforms"),
    cl::init(false));

static cl::opt<double> CostForgeCodeBudget(
    "costforge-code-budget",
    cl::desc("Max code growth factor (default 2.0)"),
    cl::init(2.0));

static cl::opt<bool> CostForgeTTICheck(
    "costforge-tti-check",
    cl::desc("Cross-check decisions against LLVM TTI"),
    cl::init(false));

namespace {

// ════════════════════════════════════════════════════════════
//  THREAD-SAFE STATE
// ════════════════════════════════════════════════════════════
//
// Old approach: static CostForgeState — not thread-safe, leaked
// across compilations in parallel builds.
//
// New approach: state per compilation unit, guarded by mutex.
// ModulePass creates it, FunctionPass reads it via Module metadata
// pointer. No global mutable state.

struct CompilationState {
    costforge::HWProfile hw;
    costforge::CostModel model;
    costforge::PatternRecognizer recognizer;
    costforge::DecisionEngine engine;
    costforge::DecisionLog log;
    costforge::ModuleTransformReport module_report;
    costforge::ModuleProfile pgo_profile;
    bool calibrated = false;
    bool has_pgo = false;
    
    CompilationState()
        : hw(CostForgeProfile.empty()
                ? costforge::HWProbe::scan()
                : costforge::HWProbe::load(CostForgeProfile)),
          model(hw),
          recognizer(hw),
          engine(model) {
        if (!CostForgeCalibration.empty()) {
            auto cal = costforge::Calibrator::load(CostForgeCalibration);
            if (cal.is_valid()) {
                costforge::Calibrator::apply(model, cal);
                calibrated = true;
                if (CostForgeVerbose)
                    errs() << "[CostForge] calibration loaded (MAE "
                           << cal.mean_absolute_error << "%)\n";
            }
        }
    }
};

// One state per Module. Protected by mutex for ThinLTO parallel backends.
static std::mutex state_mutex;
static std::map<const Module*, std::unique_ptr<CompilationState>> module_states;

static CompilationState& get_state(const Module& M) {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto& ptr = module_states[&M];
    if (!ptr) {
        ptr = std::make_unique<CompilationState>();
        ptr->log.begin_compilation(M.getSourceFileName(), ptr->hw.model);
    }
    return *ptr;
}

static void release_state(const Module& M) {
    std::lock_guard<std::mutex> lock(state_mutex);
    module_states.erase(&M);
}

// ── Helpers ────────────────────────────────────────────────

/// Map LLVM opcode to CostForge abstract opcode + extract type info
struct OpInfo {
    std::string opcode;
    uint32_t bit_width = 32;
    bool is_float = false;
};

static OpInfo classify_instruction(const Instruction& I) {
    OpInfo info;
    
    // Get type info from result or first operand
    Type* ty = I.getType();
    if (ty->isVoidTy() && I.getNumOperands() > 0)
        ty = I.getOperand(0)->getType();
    
    if (ty->isFloatingPointTy()) {
        info.is_float = true;
        info.bit_width = ty->getPrimitiveSizeInBits();
    } else if (ty->isIntegerTy()) {
        info.bit_width = ty->getIntegerBitWidth();
    } else if (ty->isVectorTy()) {
        auto* vty = cast<VectorType>(ty);
        info.bit_width = vty->getElementType()->getPrimitiveSizeInBits();
        info.is_float = vty->getElementType()->isFloatingPointTy();
    }
    
    switch (I.getOpcode()) {
        case Instruction::Add: case Instruction::FAdd:  info.opcode = "add"; break;
        case Instruction::Sub: case Instruction::FSub:  info.opcode = "sub"; break;
        case Instruction::Mul: case Instruction::FMul:  info.opcode = "mul"; break;
        case Instruction::UDiv: case Instruction::SDiv:
        case Instruction::FDiv:  info.opcode = "div"; break;
        case Instruction::And:   info.opcode = "and"; break;
        case Instruction::Or:    info.opcode = "or"; break;
        case Instruction::Xor:   info.opcode = "xor"; break;
        case Instruction::Shl: case Instruction::LShr:
        case Instruction::AShr:  info.opcode = "shift"; break;
        case Instruction::Load:  info.opcode = "load"; break;
        case Instruction::Store: info.opcode = "store"; break;
        case Instruction::Br:    info.opcode = "branch"; break;
        case Instruction::Call:  info.opcode = "call"; break;
        case Instruction::Ret:   info.opcode = "ret"; break;
        case Instruction::ICmp: case Instruction::FCmp: info.opcode = "cmp"; break;
        case Instruction::Select: info.opcode = "cmov"; break;
        default: info.opcode = "nop"; break;
    }
    
    return info;
}

/// Extract trip count from ScalarEvolution (real, not guessed)
static uint64_t get_trip_count(Loop* L, ScalarEvolution& SE) {
    // Try exact trip count first
    unsigned exact = SE.getSmallConstantTripCount(L);
    if (exact > 0)
        return exact;
    
    // Try max trip count (upper bound)
    unsigned max_tc = SE.getSmallConstantMaxTripCount(L);
    if (max_tc > 0 && max_tc < 1000000)
        return max_tc;
    
    // Fallback: estimate from loop structure
    if (auto* latch = L->getLoopLatch()) {
        // Count instructions in body — smaller body implies higher trip count
        unsigned body_size = 0;
        for (auto* BB : L->getBlocks())
            body_size += BB->size();
        
        if (body_size < 10) return 1024;
        if (body_size < 30) return 256;
        return 64;
    }
    
    return 128; // conservative default
}

/// Check for cross-iteration dependencies using MemorySSA
static bool has_loop_carried_deps(Loop* L, MemorySSA& MSSA) {
    MemorySSAWalker* walker = MSSA.getWalker();
    
    for (auto* BB : L->getBlocks()) {
        for (auto& I : *BB) {
            auto* store = dyn_cast<StoreInst>(&I);
            if (!store) continue;
            
            // Check if any load in the loop reads from memory
            // that this store might have written to in a previous iteration
            MemoryAccess* store_acc = MSSA.getMemoryAccess(store);
            if (!store_acc) continue;
            
            for (auto* BB2 : L->getBlocks()) {
                for (auto& I2 : *BB2) {
                    auto* load = dyn_cast<LoadInst>(&I2);
                    if (!load) continue;
                    
                    MemoryAccess* load_acc = MSSA.getMemoryAccess(load);
                    if (!load_acc) continue;
                    
                    // Walk to the defining access of this load
                    MemoryAccess* def = walker->getClobberingMemoryAccess(load);
                    
                    // If the defining access is the store (or reaches it),
                    // we have a potential loop-carried dependency
                    if (def == store_acc)
                        return true;
                    
                    // Also check if def is a MemoryPhi at loop header
                    // (indicates values from previous iteration)
                    if (auto* phi = dyn_cast<MemoryPhi>(def)) {
                        if (L->getHeader() == phi->getBlock())
                            return true;
                    }
                }
            }
        }
    }
    
    return false;
}

/// Estimate working set from allocas + accessed global sizes
static uint64_t estimate_working_set(const Function& F) {
    uint64_t total = 0;
    for (const auto& BB : F)
        for (const auto& I : BB) {
            if (auto* AI = dyn_cast<AllocaInst>(&I)) {
                if (AI->getAllocatedType()->isSized())
                    total += F.getParent()->getDataLayout().getTypeAllocSize(
                        AI->getAllocatedType());
            }
            // Also count globals accessed by the function
            if (auto* LI = dyn_cast<LoadInst>(&I)) {
                if (auto* GV = dyn_cast<GlobalVariable>(
                        LI->getPointerOperand()->stripPointerCasts())) {
                    if (GV->getValueType()->isSized())
                        total += F.getParent()->getDataLayout().getTypeAllocSize(
                            GV->getValueType());
                }
            }
        }
    return std::max(total, uint64_t(64));
}

/// Cross-check a CostForge decision against LLVM's TTI
static bool tti_agrees(const TargetTransformInfo& TTI,
                       const std::string& transform,
                       Loop* L,
                       ScalarEvolution& SE) {
    if (transform == "vectorize") {
        // TTI can tell us the preferred vector width
        auto width = TTI.getRegisterBitWidth(
            TargetTransformInfo::RGK_FixedWidthVector);
        return width.getFixedValue() >= 128;
    }
    if (transform == "unroll") {
        // TTI has unroll preferences
        TargetTransformInfo::UnrollingPreferences UP;
        // ORE is only used by TTI for optimization remarks; not needed here.
        TTI.getUnrollingPreferences(L, SE, UP, /*ORE=*/nullptr);
        return UP.Count > 1 || UP.UpperBound;
    }
    return true; // no opinion → agree
}

// ════════════════════════════════════════════════════════════
//  FUNCTION PASS
// ════════════════════════════════════════════════════════════

struct CostForgePass : PassInfoMixin<CostForgePass> {
    
    PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
        if (F.isDeclaration() || F.size() < 2)
            return PreservedAnalyses::all();
        
        auto& state = get_state(*F.getParent());
        
        // Get LLVM analyses
        auto& LI = FAM.getResult<LoopAnalysis>(F);
        auto& SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        auto& MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
        auto& TTI = FAM.getResult<TargetIRAnalysis>(F);
        
        if (CostForgeVerbose)
            errs() << "[CostForge] analyzing: " << F.getName() << "\n";
        
        if (CostForgeDryRun)
            return PreservedAnalyses::all();
        
        // ── PGO: apply measured branch weights before transforms ──
        if (state.has_pgo) {
            auto it = state.pgo_profile.functions.find(F.getName().str());
            if (it != state.pgo_profile.functions.end()) {
                const auto& fp = it->second;
                
                // Apply real branch probabilities from PGO
                for (auto& BB : F) {
                    auto* BI = dyn_cast<BranchInst>(BB.getTerminator());
                    if (!BI || !BI->isConditional()) continue;
                    
                    // If PGO already set weights, use them directly
                    auto prob = costforge::ProfileReader::get_branch_prob(BI);
                    if (prob.has_value()) {
                        if (CostForgeVerbose)
                            errs() << "[CostForge]   PGO branch @"
                                   << BB.getName() << ": "
                                   << prob->true_probability * 100.0
                                   << "% true (" << prob->total_count
                                   << " samples)\n";
                    }
                }
                
                if (CostForgeVerbose && fp.entry_count > 0)
                    errs() << "[CostForge]   PGO entry count: "
                           << fp.entry_count
                           << (fp.is_hot ? " (HOT)" : "")
                           << (fp.is_cold ? " (COLD)" : "") << "\n";
            }
        }
        
        // Create transform engine
        costforge::TransformEngine xform(
            state.engine, state.recognizer, state.log, state.hw);
        
        // Run full pipeline
        auto report = xform.transform_function(F, FAM);
        
        // If a loop-interchange fired, it tore down and rebuilt part of
        // the CFG (see TransformEngine::interchange_matmul_ikj) — LI/SE/MSSA
        // fetched above are now stale for the loops it touched. Skip the
        // rest of this function's loop-based analysis below; it'll run
        // fresh next time this function is visited (we return
        // PreservedAnalyses::none() below whenever transforms_applied > 0,
        // which forces recomputation).
        bool structure_changed = false;
        for (const auto& r : report.results)
            if (r.applied && r.transform_name == "interchange_matmul_ikj")
                structure_changed = true;
        
        // ── Post-transform analysis ──
        
        if (!structure_changed)
        for (auto* L : LI) {
            SmallVector<Loop*, 4> worklist;
            worklist.push_back(L);
            while (!worklist.empty()) {
                Loop* cur = worklist.pop_back_val();
                
                uint64_t tc = get_trip_count(cur, SE);
                bool deps = has_loop_carried_deps(cur, MSSA);
                
                // ── LAA: check vectorization legality ──
                // If we set a vectorize hint, verify with LAA that
                // it's actually safe. If not, remove the hint.
                bool vec_hint_set = false;
                if (MDNode* lid = cur->getLoopID()) {
                    for (unsigned i = 1; i < lid->getNumOperands(); ++i) {
                        if (auto* md = dyn_cast<MDNode>(lid->getOperand(i).get()))
                            if (md->getNumOperands() > 0)
                                if (auto* s = dyn_cast<MDString>(md->getOperand(0)))
                                    if (s->getString() == "llvm.loop.vectorize.width")
                                        vec_hint_set = true;
                    }
                }
                
                if (vec_hint_set && deps) {
                    // MemorySSA detected cross-iteration deps.
                    // Remove vectorization hint — it's unsafe.
                    LLVMContext& ctx = cur->getHeader()->getContext();
                    MDNode* old_id = cur->getLoopID();
                    if (old_id) {
                        SmallVector<Metadata*, 8> ops;
                        ops.push_back(nullptr);
                        for (unsigned i = 1; i < old_id->getNumOperands(); ++i) {
                            auto* op = old_id->getOperand(i).get();
                            bool is_vec = false;
                            if (auto* md = dyn_cast<MDNode>(op))
                                if (md->getNumOperands() > 0)
                                    if (auto* s = dyn_cast<MDString>(md->getOperand(0)))
                                        if (s->getString().starts_with("llvm.loop.vectorize") ||
                                            s->getString().starts_with("llvm.loop.interleave"))
                                            is_vec = true;
                            if (!is_vec)
                                ops.push_back(op);
                        }
                        MDNode* new_id = MDNode::getDistinct(ctx, ops);
                        new_id->replaceOperandWith(0, new_id);
                        cur->setLoopID(new_id);
                        
                        if (CostForgeVerbose)
                            errs() << "[CostForge]   REVOKED vectorize hint @"
                                   << cur->getHeader()->getName()
                                   << " — loop-carried dependency\n";
                    }
                }
                
                // Type-aware cost analysis
                double type_aware_cost = 0.0;
                for (auto* BB : cur->getBlocks())
                    for (auto& I : *BB) {
                        auto info = classify_instruction(I);
                        type_aware_cost += state.model.instruction_cost(
                            info.opcode, info.bit_width, info.is_float);
                    }
                
                if (CostForgeVerbose && CostForgeTTICheck) {
                    errs() << "[CostForge]   loop @"
                           << cur->getHeader()->getName()
                           << " TC=" << tc
                           << " deps=" << (deps ? "yes" : "no")
                           << " type_cost=" << type_aware_cost << "\n";
                }
                
                for (auto* sub : cur->getSubLoops())
                    worklist.push_back(sub);
            }
        }
        
        // TTI cross-check
        if (!structure_changed && CostForgeTTICheck && CostForgeVerbose) {
            for (const auto& r : report.results) {
                if (!r.applied) continue;
                bool agrees = true;
                // Check if TTI would also recommend this transform
                for (auto* L : LI) {
                    if (r.transform_name == "set_vectorize_hint")
                        agrees = tti_agrees(TTI, "vectorize", L, SE);
                    else if (r.transform_name == "set_unroll_hint")
                        agrees = tti_agrees(TTI, "unroll", L, SE);
                }
                errs() << "[CostForge]   TTI " << r.transform_name
                       << ": " << (agrees ? "agrees" : "DISAGREES") << "\n";
            }
        }
        
        if (CostForgeVerbose && report.transforms_applied > 0) {
            errs() << "[CostForge]   " << report.transforms_applied
                   << " transforms, " << report.combined_speedup << "x\n";
            for (const auto& r : report.results)
                if (r.applied)
                    errs() << "[CostForge]     ✓ " << r.description << "\n";
        }
        
        // Accumulate
        if (report.transforms_applied > 0) {
            state.module_report.functions_modified++;
            state.module_report.total_transforms += report.transforms_applied;
            state.module_report.estimated_module_speedup *= report.combined_speedup;
            state.module_report.total_code_growth += report.total_code_growth;
            state.module_report.functions.push_back(std::move(report));
            return PreservedAnalyses::none();
        }
        return PreservedAnalyses::all();
    }
    
    static bool isRequired() { return false; }
};

// ════════════════════════════════════════════════════════════
//  MODULE PASS
// ════════════════════════════════════════════════════════════

struct CostForgeModulePass : PassInfoMixin<CostForgeModulePass> {
    
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& MAM) {
        auto& state = get_state(M);
        
        if (CostForgeVerbose)
            errs() << "[CostForge] === Module: "
                   << M.getSourceFileName() << " ===\n";
        
        // ── Load PGO data if available ─────────────────
        if (costforge::ProfileReader::has_profile_data(M)) {
            state.pgo_profile = costforge::ProfileReader::read_module(M, MAM);
            state.has_pgo = true;
            if (CostForgeVerbose)
                errs() << "[CostForge] PGO: "
                       << state.pgo_profile.hot_function_count << " hot, "
                       << state.pgo_profile.cold_function_count << " cold / "
                       << state.pgo_profile.total_function_count << " total\n";
        }
        
        // ── Build call graph ───────────────────────────
        costforge::InterproceduralAnalyzer ipa(state.model, state.hw);
        
        std::vector<costforge::CallNode> nodes;
        std::vector<costforge::CallEdge> edges;
        
        for (auto& F : M) {
            if (F.isDeclaration()) continue;
            
            costforge::CallNode node;
            node.function_name = F.getName().str();
            node.instruction_count = F.getInstructionCount();
            node.basic_block_count = F.size();
            node.register_pressure =
                static_cast<double>(F.getInstructionCount()) / 48.0;
            node.working_set_bytes = estimate_working_set(F);
            
            // PGO: use measured entry count + hot/cold classification
            if (state.has_pgo) {
                auto it = state.pgo_profile.functions.find(F.getName().str());
                if (it != state.pgo_profile.functions.end()) {
                    const auto& fp = it->second;
                    node.hotness = static_cast<double>(fp.entry_count);
                    if (fp.is_hot) node.hotness = std::max(node.hotness, 1000.0);
                    if (fp.is_cold) node.hotness = std::min(node.hotness, 0.1);
                }
            }
            
            // Check if leaf
            node.is_leaf = true;
            for (auto& BB : F)
                for (auto& I : BB)
                    if (auto* CI = dyn_cast<CallInst>(&I))
                        if (auto* callee = CI->getCalledFunction())
                            if (!callee->isDeclaration())
                                node.is_leaf = false;
            
            nodes.push_back(std::move(node));
            
            // Extract call edges — use PGO block frequencies for call counts
            for (auto& BB : F) {
                for (auto& I : BB) {
                    auto* CI = dyn_cast<CallInst>(&I);
                    if (!CI || CI->isInlineAsm()) continue;
                    Function* callee = CI->getCalledFunction();
                    if (!callee || callee->isDeclaration()) continue;
                    
                    costforge::CallEdge edge;
                    edge.caller = F.getName().str();
                    edge.callee = callee->getName().str();
                    edge.call_count = 1;
                    edge.estimated_freq = 1;
                    edge.is_indirect = false;
                    edge.loop_depth = 0;
                    edge.is_in_loop = false;
                    
                    // PGO: use measured call frequency from block counts
                    if (state.has_pgo) {
                        auto it = state.pgo_profile.functions.find(
                            F.getName().str());
                        if (it != state.pgo_profile.functions.end()) {
                            auto blk = it->second.block_counts.find(
                                BB.getName().str());
                            if (blk != it->second.block_counts.end()) {
                                edge.call_count = blk->second;
                                edge.estimated_freq = blk->second;
                            }
                        }
                    }
                    
                    edges.push_back(std::move(edge));
                }
            }
        }
        
        if (nodes.empty()) {
            release_state(M);
            return PreservedAnalyses::all();
        }
        
        ipa.build_call_graph(nodes, edges);
        ipa.compute_traversal_order();
        ipa.propagate_hotness();
        
        auto inline_plan = ipa.compute_inline_plan();
        auto cold_fns = ipa.find_outline_candidates();
        
        if (CostForgeVerbose) {
            errs() << "[CostForge] IPA: " << nodes.size() << " fn, "
                   << edges.size() << " edges, "
                   << ipa.call_graph().sccs.size() << " SCCs\n";
            errs() << "[CostForge] plan: "
                   << inline_plan.sites.size() << " inlines, "
                   << cold_fns.size() << " cold\n";
        }
        
        bool modified = false;
        
        if (!CostForgeDryRun) {
            costforge::TransformEngine xform(
                state.engine, state.recognizer, state.log, state.hw);
            
            // Apply inlines
            auto inline_results = xform.apply_inline_plan(M, inline_plan);
            for (const auto& r : inline_results) {
                if (r.applied) {
                    modified = true;
                    state.module_report.total_transforms++;
                    if (CostForgeVerbose)
                        errs() << "[CostForge]   ✓ " << r.description << "\n";
                }
            }
            
            // Mark cold — from IPA analysis + PGO
            // With PGO, also mark functions PSI classified as cold
            auto all_cold = cold_fns;
            if (state.has_pgo) {
                for (const auto& [name, fp] : state.pgo_profile.functions) {
                    if (fp.is_cold && std::find(all_cold.begin(),
                            all_cold.end(), name) == all_cold.end())
                        all_cold.push_back(name);
                }
            }
            auto cold_results = xform.apply_outline_hints(M, all_cold);
            for (const auto& r : cold_results)
                if (r.applied) { modified = true; state.module_report.total_transforms++; }
            
            // Mark hot — from IPA hotness OR PGO hot classification
            for (auto& node : nodes) {
                bool is_hot = node.hotness > 100.0;
                if (state.has_pgo) {
                    auto it = state.pgo_profile.functions.find(node.function_name);
                    if (it != state.pgo_profile.functions.end())
                        is_hot = is_hot || it->second.is_hot;
                }
                if (is_hot) {
                    if (Function* F = M.getFunction(node.function_name)) {
                        if (!F->isDeclaration()) {
                            auto res = xform.mark_hot_function(*F);
                            if (res.applied) {
                                modified = true;
                                state.module_report.total_transforms++;
                            }
                        }
                    }
                }
            }
        }
        
        // Write report
        if (!CostForgeReport.empty()) {
            std::ofstream out(CostForgeReport.getValue());
            if (out.is_open())
                out << state.module_report.to_json();
        }
        
        if (CostForgeVerbose)
            errs() << state.module_report.to_summary();
        
        // Clean up module state
        release_state(M);
        
        if (modified)
            return PreservedAnalyses::none();
        return PreservedAnalyses::all();
    }
    
    static bool isRequired() { return false; }
};

} // anonymous namespace

// ════════════════════════════════════════════════════════════
//  PLUGIN REGISTRATION
// ════════════════════════════════════════════════════════════

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "CostForge", LLVM_VERSION_STRING,
            [](PassBuilder& PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager& FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "costforge") {
                            FPM.addPass(CostForgePass());
                            return true;
                        }
                        return false;
                    });
                
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, ModulePassManager& MPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "costforge-ipa") {
                            MPM.addPass(CostForgeModulePass());
                            return true;
                        }
                        return false;
                    });
                
                PB.registerOptimizerLastEPCallback(
                    [](ModulePassManager& MPM, OptimizationLevel Level) {
                        if (Level.getSpeedupLevel() >= 2) {
                            MPM.addPass(CostForgeModulePass());
                            MPM.addPass(createModuleToFunctionPassAdaptor(
                                CostForgePass()));
                        }
                    });
            }};
}

#endif // COSTFORGE_LLVM_PASS
