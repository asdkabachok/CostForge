#pragma once
/// CostForge ProfileReader — v0.3.1
///
/// Extracts profiling data from LLVM IR metadata set by PGO
/// (-fprofile-use / -fprofile-instr-use). Replaces heuristic
/// guesses with measured reality.
///
/// Data sources:
///   - !prof branch_weights on BranchInst/SwitchInst
///   - !prof VP (value profile) on indirect CallInst
///   - function entry counts from !prof metadata on Function
///   - block frequency info from BlockFrequencyAnalysis
///
/// When PGO data is absent, returns nullopt — callers fall back
/// to heuristics. Never invents data.

#ifdef COSTFORGE_LLVM_PASS

#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"

#include "costforge/types.h"

#include <optional>
#include <string>
#include <vector>
#include <map>

namespace costforge {

/// Branch probability from PGO (not guessed)
struct MeasuredBranchProb {
    double true_probability;   // 0.0 – 1.0
    uint64_t true_count;
    uint64_t false_count;
    uint64_t total_count;
};

/// Call site profile data
struct CallSiteProfile {
    std::string callee_name;
    uint64_t call_count;
    bool is_indirect;
    /// For indirect calls: most likely targets from VP metadata
    struct Target {
        std::string name;
        uint64_t count;
        double probability;
    };
    std::vector<Target> indirect_targets;
};

/// Function-level profile summary
struct FunctionProfile {
    std::string name;
    uint64_t entry_count = 0;
    bool is_hot = false;
    bool is_cold = false;
    
    std::map<std::string, uint64_t> block_counts;  // BB name → exec count
    std::vector<MeasuredBranchProb> branch_probs;
    std::vector<CallSiteProfile> call_profiles;
    
    /// Hottest basic block execution count
    uint64_t max_block_count = 0;
    /// Total dynamic instruction count estimate
    uint64_t estimated_dynamic_insns = 0;
};

/// Module-level profile summary
struct ModuleProfile {
    bool has_pgo_data = false;
    uint64_t total_function_count = 0;
    uint64_t hot_function_count = 0;
    uint64_t cold_function_count = 0;
    
    std::map<std::string, FunctionProfile> functions;
};


class ProfileReader {
public:
    /// Read all PGO data from a module.
    /// Returns ModuleProfile with has_pgo_data=false if no profiling info.
    static ModuleProfile read_module(
        llvm::Module& M,
        llvm::ModuleAnalysisManager& MAM);
    
    /// Read PGO data for a single function.
    /// Requires BlockFrequencyInfo and BranchProbabilityInfo.
    static FunctionProfile read_function(
        llvm::Function& F,
        llvm::BlockFrequencyInfo* BFI,
        llvm::BranchProbabilityInfo* BPI);
    
    /// Extract branch weights from a branch instruction.
    /// Returns nullopt if no !prof metadata.
    static std::optional<MeasuredBranchProb> get_branch_prob(
        llvm::BranchInst* BI);
    
    /// Extract branch weights from a switch instruction.
    /// Returns case_index → probability map.
    static std::map<unsigned, double> get_switch_probs(
        llvm::SwitchInst* SI);
    
    /// Get function entry count from !prof metadata.
    /// Returns nullopt if not instrumented.
    static std::optional<uint64_t> get_entry_count(
        llvm::Function& F);
    
    /// Get value profile targets for an indirect call.
    /// Returns empty if no VP metadata.
    static std::vector<CallSiteProfile::Target> get_indirect_targets(
        llvm::CallInst* CI);
    
    /// Check if a module has any PGO data at all
    static bool has_profile_data(llvm::Module& M);
};

} // namespace costforge

#endif // COSTFORGE_LLVM_PASS
