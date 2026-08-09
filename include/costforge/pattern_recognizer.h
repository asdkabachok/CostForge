#pragma once

#include "costforge/types.h"
#include "costforge/cost_model.h"
#include <vector>
#include <functional>

namespace costforge {

/// PatternRecognizer — detects known computational idioms in IR.
///
/// Unlike dumb compilers that only see instructions, CostForge
/// recognizes WHAT the code is trying to do and applies
/// domain-specific optimizations:
///
///   - Reduction loop? → use horizontal SIMD reduction
///   - Matrix multiply? → tile for L1/L2, use FMA
///   - Stencil? → prefetch neighbors, vectorize across dimension
///   - Gather/scatter? → consider permute instructions
///   - Branch chain? → convert to jump table
///
/// The recognizer works on an abstract representation of loops
/// and basic blocks (not raw LLVM IR, so it's testable standalone).

class PatternRecognizer {
public:
    explicit PatternRecognizer(const HWProfile& hw);
    
    /// Abstract loop representation for pattern matching
    struct LoopInfo {
        std::vector<std::string> body_opcodes;
        uint64_t trip_count;
        uint64_t working_set_bytes;
        
        // Access pattern analysis
        struct ArrayAccess {
            std::string array_name;
            std::string index_expr;    // "i", "j", "i*N+j", "idx[i]"
            bool is_read;
            bool is_write;
        };
        std::vector<ArrayAccess> accesses;
        
        // Loop structure
        uint32_t nesting_depth;        // 0 = innermost
        uint32_t data_width_bits = 32; // element width: 8/16/32/64
        bool has_loop_carried_dep;
        
        // Accumulator detection
        bool has_accumulator;          // sum += ..., max = max(...)
        std::string accumulator_op;    // "add", "mul", "min", "max"
    };
    
    /// Abstract basic block for branch pattern matching
    struct BlockInfo {
        std::string label;
        std::vector<std::string> opcodes;
        uint32_t branch_count;
        uint32_t comparison_count;
        
        // Branch targets
        std::vector<std::string> successors;
        
        // If/else chain detection
        bool is_chain_link;            // part of if/else if chain
        uint32_t chain_length;         // total length of chain
    };
    
    // ── Main entry points ────────────────────────────
    
    /// Analyze a loop and return all matching patterns
    std::vector<PatternMatch> analyze_loop(const LoopInfo& loop) const;
    
    /// Analyze a basic block for branch patterns
    std::vector<PatternMatch> analyze_block(const BlockInfo& block) const;
    
    /// Get the recommended transform for a pattern on THIS hardware
    std::string recommend_transform(const PatternMatch& match) const;
    
    /// Estimate speedup of applying the recommended transform
    double estimate_speedup(const PatternMatch& match) const;
    
private:
    HWProfile hw_;
    
    // Individual pattern detectors
    PatternMatch detect_reduction(const LoopInfo& loop) const;
    PatternMatch detect_matrix_multiply(const LoopInfo& loop) const;
    PatternMatch detect_dot_product(const LoopInfo& loop) const;
    PatternMatch detect_axpy(const LoopInfo& loop) const;
    PatternMatch detect_stencil(const LoopInfo& loop) const;
    PatternMatch detect_memcopy(const LoopInfo& loop) const;
    PatternMatch detect_gather_scatter(const LoopInfo& loop) const;
    PatternMatch detect_histogram(const LoopInfo& loop) const;
    PatternMatch detect_prefix_scan(const LoopInfo& loop) const;
    PatternMatch detect_branch_chain(const BlockInfo& block) const;
    PatternMatch detect_predicated_loop(const LoopInfo& loop) const;
    
    // Hardware-specific transform selection
    uint32_t optimal_tile_size() const;
    bool has_fma() const;
    bool should_use_streaming_stores(uint64_t working_set) const;
};

} // namespace costforge
