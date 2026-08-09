#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <optional>

namespace costforge {

// ============================================================
// Hardware Profile Types (unchanged from v0.1.0)
// ============================================================

struct CacheLevel {
    uint32_t size_kb;
    uint32_t line_size;
    uint32_t associativity;
    uint32_t latency_cycles;
    
    uint32_t num_sets() const {
        return (size_kb * 1024) / (line_size * associativity);
    }
};

struct SIMDCapabilities {
    bool sse42  = false;
    bool avx    = false;
    bool avx2   = false;
    bool avx512 = false;
    
    uint32_t max_width() const {
        if (avx512) return 512;
        if (avx2 || avx) return 256;
        if (sse42) return 128;
        return 64;
    }
};

struct HWProfile {
    std::string vendor;
    std::string model;
    std::string microarch;
    
    uint32_t physical_cores;
    uint32_t logical_threads;
    uint32_t base_freq_mhz;
    uint32_t boost_freq_mhz;
    
    CacheLevel l1d;
    CacheLevel l1i;
    CacheLevel l2;
    CacheLevel l3;
    
    uint32_t execution_ports;
    uint32_t pipeline_depth;
    uint32_t rob_size;
    uint32_t load_buffer_size;
    uint32_t store_buffer_size;
    
    uint32_t branch_mispredict_penalty;
    
    SIMDCapabilities simd;
    
    uint32_t memory_latency_ns;
    uint32_t memory_bandwidth_gbs;
};

// ============================================================
// Cost Model Types (unchanged from v0.1.0)
// ============================================================

struct CostResult {
    double throughput_cycles  = 0.0;
    double latency_cycles    = 0.0;
    double cache_pressure    = 0.0;
    double register_pressure = 0.0;
    double branch_risk       = 0.0;
    
    double total() const {
        // Cache and register pressure are relative costs —
        // scale them by throughput so they matter proportionally.
        // A 0.5 cache_pressure on a 1000-cycle loop = 500 extra cycles
        // from cache misses, not a fixed 25.
        double cache_penalty = cache_pressure * throughput_cycles * 0.3;
        double reg_penalty = register_pressure * throughput_cycles * 0.05;
        double branch_penalty = branch_risk * 17.0;
        return throughput_cycles + cache_penalty + reg_penalty + branch_penalty;
    }
    
    bool operator<(const CostResult& other) const {
        return total() < other.total();
    }
};

struct Decision {
    std::string name;
    std::string description;
    CostResult cost_a;
    CostResult cost_b;
    bool choose_b;
    double savings_percent;
    
    void evaluate() {
        choose_b = cost_b < cost_a;
        double chosen = choose_b ? cost_b.total() : cost_a.total();
        double rejected = choose_b ? cost_a.total() : cost_b.total();
        savings_percent = (rejected > 0.0) 
            ? ((rejected - chosen) / rejected) * 100.0
            : 0.0;
    }
};

struct InstructionCost {
    std::string mnemonic;
    double throughput;
    double latency;
    uint32_t port_mask;
    uint32_t uops;
};

using CostTable = std::map<std::string, InstructionCost>;

// ============================================================
// v0.2.0: Pattern Recognition
// ============================================================

enum class PatternKind {
    None,
    
    // Reductions
    ReductionSum,
    ReductionMin,
    ReductionMax,
    ReductionProduct,
    
    // Linear algebra
    MatrixMultiply,
    MatrixTranspose,
    DotProduct,
    AXPY,
    
    // Stencils
    Stencil1D,
    Stencil2D,
    
    // Data movement
    Memcopy,
    Gather,
    Scatter,
    StreamStore,
    
    // Combinatorial
    HistogramUpdate,
    PrefixScan,
    
    // Branch patterns
    BranchChain,
    PredicatedLoop,
};

const char* pattern_name(PatternKind kind);

struct PatternMatch {
    PatternKind kind           = PatternKind::None;
    double confidence          = 0.0;
    std::string location;
    
    uint32_t trip_count        = 0;
    uint32_t data_width_bits   = 32;
    uint64_t working_set       = 0;
    uint32_t dimensions        = 1;
    
    std::string suggested_transform;
    double estimated_speedup   = 1.0;
};

// ============================================================
// v0.2.0: Calibration
// ============================================================

struct CalibrationPoint {
    std::string operation;
    double predicted_cycles;
    double measured_cycles;
    double error_percent;
};

struct CalibrationProfile {
    std::string cpu_model;
    std::string microarch;
    std::string timestamp;
    
    double alu_correction        = 1.0;
    double mul_correction        = 1.0;
    double div_correction        = 1.0;
    double load_correction       = 1.0;
    double store_correction      = 1.0;
    double branch_correction     = 1.0;
    double simd_correction       = 1.0;
    
    double l1_latency_correction  = 1.0;
    double l2_latency_correction  = 1.0;
    double l3_latency_correction  = 1.0;
    double dram_latency_correction = 1.0;
    
    double mean_absolute_error   = 0.0;
    
    std::vector<CalibrationPoint> points;
    
    bool is_valid() const { return mean_absolute_error < 50.0; }
};

// ============================================================
// v0.2.0: Decision Logging
// ============================================================

struct DecisionTrace {
    uint32_t id;
    std::string timestamp;
    
    std::string function_name;
    std::string location;
    std::string decision_type;
    
    struct Option {
        std::string name;
        CostResult cost;
        std::string reasoning;
    };
    std::vector<Option> options;
    
    uint32_t chosen_index;
    double confidence;
    std::string verdict_summary;
    
    std::vector<std::string> factors;
    std::optional<PatternKind> detected_pattern;
};

struct CompilationReport {
    std::string source_file;
    std::string target_cpu;
    std::string timestamp;
    
    uint32_t total_functions      = 0;
    uint32_t total_decisions      = 0;
    uint32_t transforms_applied   = 0;
    double estimated_speedup_pct  = 0.0;
    
    std::vector<DecisionTrace> traces;
    
    struct CategoryStats {
        uint32_t considered = 0;
        uint32_t applied    = 0;
        double avg_savings  = 0.0;
    };
    std::map<std::string, CategoryStats> by_category;
};

// ============================================================
// v0.2.0: Interprocedural Analysis
// ============================================================

struct CallNode {
    std::string function_name;
    uint32_t instruction_count  = 0;
    uint32_t basic_block_count  = 0;
    double register_pressure    = 0.0;
    uint64_t working_set_bytes  = 0;
    bool is_leaf                = true;
    bool is_recursive           = false;
    uint32_t depth              = 0;
    double hotness              = 0.0;
};

struct CallEdge {
    std::string caller;
    std::string callee;
    uint32_t call_count      = 1;
    uint32_t estimated_freq  = 1;
    bool is_indirect         = false;
    bool is_in_loop          = false;
    uint32_t loop_depth      = 0;
};

struct CallGraph {
    std::vector<CallNode> nodes;
    std::vector<CallEdge> edges;
    std::vector<std::string> bottom_up_order;
    std::vector<std::vector<std::string>> sccs;
};

struct InlinePlan {
    struct Site {
        std::string caller;
        std::string callee;
        uint32_t call_site_id;
        double benefit;
        double code_growth;
        std::string reason;
    };
    
    std::vector<Site> sites;
    double total_benefit     = 0.0;
    double total_code_growth = 0.0;
    
    static constexpr double MAX_CODE_GROWTH_FACTOR = 2.0;
};

} // namespace costforge
