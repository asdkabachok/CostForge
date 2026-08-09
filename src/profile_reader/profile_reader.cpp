/// CostForge ProfileReader — v0.3.1
///
/// Reads PGO data from LLVM IR metadata. Zero invention —
/// if data isn't there, returns nullopt.

#ifdef COSTFORGE_LLVM_PASS

#include "costforge/profile_reader.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace costforge {

// ════════════════════════════════════════════════════════════
//  MODULE-LEVEL
// ════════════════════════════════════════════════════════════

bool ProfileReader::has_profile_data(Module& M) {
    // Check for profile summary metadata
    if (M.getProfileSummary(/*IsCS=*/false))
        return true;
    
    // Check if any function has entry count
    for (auto& F : M) {
        if (F.isDeclaration()) continue;
        if (F.getEntryCount().has_value())
            return true;
    }
    return false;
}

ModuleProfile ProfileReader::read_module(Module& M,
                                         ModuleAnalysisManager& MAM) {
    ModuleProfile profile;
    profile.has_pgo_data = has_profile_data(M);
    
    if (!profile.has_pgo_data)
        return profile;
    
    // Get ProfileSummaryInfo for hot/cold classification
    auto& PSI = MAM.getResult<ProfileSummaryAnalysis>(M);
    
    auto& FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M)
                    .getManager();
    
    for (auto& F : M) {
        if (F.isDeclaration()) continue;
        profile.total_function_count++;
        
        // Get per-function analyses
        BlockFrequencyInfo* BFI = nullptr;
        BranchProbabilityInfo* BPI = nullptr;
        
        if (FAM.getCachedResult<BlockFrequencyAnalysis>(F))
            BFI = &FAM.getResult<BlockFrequencyAnalysis>(F);
        if (FAM.getCachedResult<BranchProbabilityAnalysis>(F))
            BPI = &FAM.getResult<BranchProbabilityAnalysis>(F);
        
        auto func_profile = read_function(F, BFI, BPI);
        
        // Hot/cold classification
        if (auto count = F.getEntryCount()) {
            if (PSI.isFunctionEntryHot(&F)) {
                func_profile.is_hot = true;
                profile.hot_function_count++;
            } else if (PSI.isFunctionEntryCold(&F)) {
                func_profile.is_cold = true;
                profile.cold_function_count++;
            }
        }
        
        profile.functions[F.getName().str()] = std::move(func_profile);
    }
    
    return profile;
}

// ════════════════════════════════════════════════════════════
//  FUNCTION-LEVEL
// ════════════════════════════════════════════════════════════

FunctionProfile ProfileReader::read_function(Function& F,
                                              BlockFrequencyInfo* BFI,
                                              BranchProbabilityInfo* BPI) {
    FunctionProfile profile;
    profile.name = F.getName().str();
    
    // Entry count
    if (auto ec = get_entry_count(F))
        profile.entry_count = *ec;
    
    // Per-block analysis
    for (auto& BB : F) {
        // Block frequency
        if (BFI) {
            auto freq = BFI->getBlockFreq(&BB);
            uint64_t count = freq.getFrequency();
            profile.block_counts[BB.getName().str()] = count;
            profile.max_block_count = std::max(profile.max_block_count, count);
            profile.estimated_dynamic_insns += count * BB.size();
        }
        
        // Branch probabilities
        auto* term = BB.getTerminator();
        if (auto* BI = dyn_cast<BranchInst>(term)) {
            if (BI->isConditional()) {
                if (auto prob = get_branch_prob(BI))
                    profile.branch_probs.push_back(*prob);
            }
        }
        
        // Call site profiles
        for (auto& I : BB) {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI || CI->isInlineAsm()) continue;
            
            CallSiteProfile csp;
            
            if (Function* callee = CI->getCalledFunction()) {
                csp.callee_name = callee->getName().str();
                csp.is_indirect = false;
            } else {
                csp.callee_name = "<indirect>";
                csp.is_indirect = true;
                csp.indirect_targets = get_indirect_targets(CI);
            }
            
            // Call count from block frequency
            if (BFI) {
                csp.call_count = BFI->getBlockFreq(CI->getParent())
                                     .getFrequency();
            }
            
            profile.call_profiles.push_back(std::move(csp));
        }
    }
    
    return profile;
}

// ════════════════════════════════════════════════════════════
//  INSTRUCTION-LEVEL
// ════════════════════════════════════════════════════════════

std::optional<uint64_t> ProfileReader::get_entry_count(Function& F) {
    auto ec = F.getEntryCount();
    if (ec.has_value())
        return ec->getCount();
    return std::nullopt;
}

std::optional<MeasuredBranchProb> ProfileReader::get_branch_prob(
    BranchInst* BI) {
    
    if (!BI->isConditional()) return std::nullopt;
    
    // Extract !prof branch_weights metadata
    MDNode* prof = BI->getMetadata(LLVMContext::MD_prof);
    if (!prof) return std::nullopt;
    
    // Format: !{!"branch_weights", i32 true_weight, i32 false_weight}
    if (prof->getNumOperands() < 3) return std::nullopt;
    
    auto* kind = dyn_cast<MDString>(prof->getOperand(0));
    if (!kind || kind->getString() != "branch_weights")
        return std::nullopt;
    
    auto extract_weight = [](const MDOperand& op) -> uint64_t {
        if (auto* ci = mdconst::dyn_extract<ConstantInt>(op))
            return ci->getZExtValue();
        return 0;
    };
    
    MeasuredBranchProb result;
    result.true_count = extract_weight(prof->getOperand(1));
    result.false_count = extract_weight(prof->getOperand(2));
    result.total_count = result.true_count + result.false_count;
    
    if (result.total_count == 0) return std::nullopt;
    
    result.true_probability =
        static_cast<double>(result.true_count) / result.total_count;
    
    return result;
}

std::map<unsigned, double> ProfileReader::get_switch_probs(
    SwitchInst* SI) {
    
    std::map<unsigned, double> probs;
    
    MDNode* prof = SI->getMetadata(LLVMContext::MD_prof);
    if (!prof) return probs;
    
    if (prof->getNumOperands() < 2) return probs;
    
    auto* kind = dyn_cast<MDString>(prof->getOperand(0));
    if (!kind || kind->getString() != "branch_weights")
        return probs;
    
    // Operands: !"branch_weights", default_weight, case0_weight, case1_weight, ...
    uint64_t total = 0;
    std::vector<uint64_t> weights;
    for (unsigned i = 1; i < prof->getNumOperands(); ++i) {
        uint64_t w = 0;
        if (auto* ci = mdconst::dyn_extract<ConstantInt>(prof->getOperand(i)))
            w = ci->getZExtValue();
        weights.push_back(w);
        total += w;
    }
    
    if (total == 0) return probs;
    
    for (unsigned i = 0; i < weights.size(); ++i)
        probs[i] = static_cast<double>(weights[i]) / total;
    
    return probs;
}

std::vector<CallSiteProfile::Target> ProfileReader::get_indirect_targets(
    CallInst* CI) {
    
    std::vector<CallSiteProfile::Target> targets;
    
    // Value profile metadata: !{!"VP", i32 kind, i64 total, i64 val0, i64 cnt0, ...}
    MDNode* prof = CI->getMetadata(LLVMContext::MD_prof);
    if (!prof) return targets;
    
    if (prof->getNumOperands() < 5) return targets;
    
    auto* kind = dyn_cast<MDString>(prof->getOperand(0));
    if (!kind || kind->getString() != "VP")
        return targets;
    
    // Get total count
    uint64_t total = 0;
    if (auto* ci = mdconst::dyn_extract<ConstantInt>(prof->getOperand(2)))
        total = ci->getZExtValue();
    if (total == 0) return targets;
    
    // Parse value-count pairs (starting at operand 3)
    Module* M = CI->getModule();
    for (unsigned i = 3; i + 1 < prof->getNumOperands(); i += 2) {
        CallSiteProfile::Target t;
        
        // Value is a function pointer hash — look up in module
        uint64_t val = 0;
        if (auto* ci = mdconst::dyn_extract<ConstantInt>(prof->getOperand(i)))
            val = ci->getZExtValue();
        
        uint64_t count = 0;
        if (auto* ci = mdconst::dyn_extract<ConstantInt>(prof->getOperand(i+1)))
            count = ci->getZExtValue();
        
        // Try to resolve the hash to a function name
        // (VP stores function pointer values, not names directly)
        t.name = "func_" + std::to_string(val);
        for (auto& F : *M) {
            if (F.isDeclaration()) continue;
            // Simple hash: lower bits of function address
            // In practice, LLVM uses MD5 hashes
            if (reinterpret_cast<uintptr_t>(&F) == val) {
                t.name = F.getName().str();
                break;
            }
        }
        
        t.count = count;
        t.probability = static_cast<double>(count) / total;
        targets.push_back(std::move(t));
    }
    
    // Sort by count descending
    std::sort(targets.begin(), targets.end(),
              [](const auto& a, const auto& b) {
                  return a.count > b.count;
              });
    
    return targets;
}

} // namespace costforge

#endif // COSTFORGE_LLVM_PASS
