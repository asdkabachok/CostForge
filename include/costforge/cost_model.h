#pragma once

#include "costforge/types.h"
#include <vector>

namespace costforge {

/// CostModel — the mathematical core of CostForge.
///
/// Given an IR block and a hardware profile, computes the estimated
/// execution cost in CPU cycles. No heuristics, no thresholds —
/// pure math based on instruction latencies, cache behavior,
/// and pipeline characteristics.

class CostModel {
public:
    explicit CostModel(const HWProfile& hw);
    
    // ── Instruction-level costs ──────────────────────────
    
    /// Cost of a single instruction on this CPU
    double instruction_cost(const std::string& opcode) const;
    
    /// Type-aware instruction cost: same opcode costs differently
    /// for i8/i32/i64/f32/f64. bit_width=0 means "use default".
    double instruction_cost(const std::string& opcode,
                            uint32_t bit_width,
                            bool is_float) const;
    
    /// Throughput cost of a sequence of instructions
    /// (accounts for port contention and ILP)
    double sequence_throughput(const std::vector<std::string>& opcodes) const;
    
    /// Latency of a dependency chain
    double chain_latency(const std::vector<std::string>& chain) const;
    
    // ── Memory access costs ──────────────────────────────
    
    /// Cost of accessing data of given size, considering cache hierarchy
    /// working_set_bytes: total data footprint of the loop/function
    /// access_stride: stride between accesses in bytes (0 = sequential)
    double memory_access_cost(uint64_t working_set_bytes, 
                               uint32_t access_stride = 0) const;
    
    /// Probability that data of given size fits in each cache level
    struct CacheDistribution {
        double l1_hit_rate;
        double l2_hit_rate;
        double l3_hit_rate;
        double dram_rate;
    };
    CacheDistribution cache_distribution(uint64_t working_set_bytes) const;
    
    /// Average memory latency given a working set size
    double avg_memory_latency(uint64_t working_set_bytes) const;
    
    // ── Loop costs ───────────────────────────────────────
    
    /// Cost of a loop body executed N times
    /// body_opcodes: instructions in the loop body
    /// trip_count: number of iterations (0 = unknown/runtime)
    /// working_set: total data touched by the loop
    CostResult loop_cost(const std::vector<std::string>& body_opcodes,
                         uint64_t trip_count,
                         uint64_t working_set_bytes) const;
    
    /// Cost of the same loop after unrolling by given factor
    CostResult unrolled_loop_cost(const std::vector<std::string>& body_opcodes,
                                   uint64_t trip_count,
                                   uint32_t unroll_factor,
                                   uint64_t working_set_bytes) const;
    
    /// Cost of a vectorized loop (SIMD width = vector_width bits)
    CostResult vectorized_loop_cost(const std::vector<std::string>& body_opcodes,
                                     uint64_t trip_count,
                                     uint32_t vector_width,
                                     uint64_t working_set_bytes) const;
    
    // ── Function call costs ──────────────────────────────
    
    /// Cost of a function call (call + ret + arg passing + stack frame)
    double call_overhead() const;
    
    /// Cost of inlining a function (body copied to caller)
    /// instruction_count: number of instructions in the function
    /// caller_register_pressure: how many registers caller is using
    CostResult inline_cost(uint32_t instruction_count,
                           double caller_register_pressure,
                           uint64_t callee_working_set) const;
    
    /// Cost of NOT inlining (call + return + possible cache disruption)
    CostResult call_cost(uint32_t instruction_count,
                         double caller_register_pressure) const;
    
    // ── Branch costs ─────────────────────────────────────
    
    /// Cost of a branch given estimated taken probability
    /// taken_probability: 0.0 = never taken, 1.0 = always taken
    /// 0.5 = worst case for branch predictor
    double branch_cost(double taken_probability) const;
    
    /// Cost of a branch converted to conditional move (branchless)
    double cmov_cost() const;
    
    // ── Register pressure ────────────────────────────────
    
    /// Estimate register pressure given number of live variables
    /// Returns spill cost if pressure exceeds available registers
    double register_spill_cost(uint32_t live_variables) const;

    // ── Calibration correction ────────────────────────
    
    struct CorrectionFactors {
        double alu   = 1.0;
        double mul   = 1.0;
        double div   = 1.0;
        double load  = 1.0;
        double store = 1.0;
        double branch = 1.0;
        double simd  = 1.0;
        double l1_latency   = 1.0;
        double l2_latency   = 1.0;
        double l3_latency   = 1.0;
        double dram_latency = 1.0;
    };
    
    /// Apply calibration-measured correction factors to all costs.
    /// Scales internal cost table entries and cache latencies.
    void set_correction_factors(const CorrectionFactors& factors);
    
    /// Get current correction factors (for diagnostics)
    const CorrectionFactors& correction_factors() const { return corrections_; }

private:
    HWProfile hw_;
    CostTable cost_table_;
    CorrectionFactors corrections_;
    
    /// Available general-purpose registers (platform-dependent)
    static constexpr uint32_t GP_REGISTERS = 16; // x86_64
    static constexpr uint32_t SIMD_REGISTERS = 16; // AVX: ymm0-ymm15
    
    /// Model cache miss probability using simple set-associative model
    double cache_miss_probability(uint64_t working_set_bytes, 
                                   const CacheLevel& cache) const;
};

} // namespace costforge
