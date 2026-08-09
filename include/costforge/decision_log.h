#pragma once

#include "costforge/types.h"
#include <string>
#include <vector>
#include <ostream>

namespace costforge {

/// DecisionLog — the compiler that explains itself.
///
/// Every time CostForge makes an optimization decision, the log
/// records not just WHAT it chose, but WHY — including the cost
/// breakdown, the hardware factors that mattered, and the
/// confidence level.
///
/// Output: JSON report that a human can read to understand
/// exactly why this binary is faster (or slower) than expected.
///
/// Usage:
///   clang -fplugin=./CostForge.so -O2
///         -mllvm -costforge-report=report.json main.c
///
/// Then:
///   costforge-explain report.json
///
/// This is something NO production compiler does — GCC's -fopt-info
/// tells you "loop vectorized" but never "because your L1 is 32KB
/// and the working set is 24KB and SIMD throughput on Zen+ is 2
/// vadd/cycle so vectorization saves 4.3 cycles per iteration."

class DecisionLog {
public:
    DecisionLog();
    
    /// Start a new compilation unit
    void begin_compilation(const std::string& source_file,
                           const std::string& target_cpu);
    
    /// Record a decision with full reasoning trace
    void record(const std::string& function_name,
                const std::string& location,
                const std::string& decision_type,
                const std::vector<DecisionTrace::Option>& options,
                uint32_t chosen_index,
                const std::vector<std::string>& factors,
                std::optional<PatternKind> pattern = std::nullopt);
    
    /// Convenience: record from a Decision struct + extra context
    void record_decision(const std::string& function_name,
                         const std::string& location,
                         const Decision& decision,
                         const std::vector<std::string>& factors);
    
    /// Finalize and get the full report
    CompilationReport finalize();
    
    /// Write report to JSON file
    static void write_json(const CompilationReport& report,
                           const std::string& path);
    
    /// Write human-readable summary to stream
    static void write_summary(const CompilationReport& report,
                              std::ostream& out);
    
    /// Format a single decision trace as human-readable text
    static std::string explain(const DecisionTrace& trace);
    
    /// Generate a diff: what changed between two compilations?
    /// (useful for seeing impact of code changes or CPU upgrades)
    static std::string diff_reports(const CompilationReport& before,
                                    const CompilationReport& after);

private:
    CompilationReport report_;
    uint32_t next_id_ = 0;
    
    std::string format_cost(const CostResult& cost) const;
    std::string generate_verdict(const std::vector<DecisionTrace::Option>& options,
                                 uint32_t chosen) const;
    double compute_confidence(const std::vector<DecisionTrace::Option>& options,
                              uint32_t chosen) const;
};

} // namespace costforge
