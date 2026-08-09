#include "costforge/cost_model.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace costforge {

CostModel::CostModel(const HWProfile& hw) : hw_(hw) {
    // Load instruction cost table for this microarchitecture
    // TODO: load from Agner Fog tables / uops.info database
    // For now, use generic x86_64 costs
    
    // ALU operations
    cost_table_["add"]   = {"add",   0.25, 1, 0b1111, 1};  // 4 per cycle, 1 cycle latency
    cost_table_["sub"]   = {"sub",   0.25, 1, 0b1111, 1};
    cost_table_["and"]   = {"and",   0.25, 1, 0b1111, 1};
    cost_table_["or"]    = {"or",    0.25, 1, 0b1111, 1};
    cost_table_["xor"]   = {"xor",   0.25, 1, 0b1111, 1};
    cost_table_["shift"] = {"shift", 0.5,  1, 0b0011, 1};
    
    // Multiply / Divide
    cost_table_["mul"]   = {"mul",   1.0,  3, 0b0001, 1};  // 1 per cycle, 3 cycle latency
    cost_table_["imul"]  = {"imul",  1.0,  3, 0b0001, 1};
    cost_table_["div"]   = {"div",   6.0, 20, 0b0001, 1};  // 1 per 6 cycles, 20 cycle latency
    
    // Memory
    cost_table_["load"]  = {"load",  0.5,  4, 0b0010, 1};  // 2 per cycle, L1 latency
    cost_table_["store"] = {"store", 1.0,  4, 0b0100, 1};  // 1 per cycle
    
    // Branch
    cost_table_["branch"]   = {"branch",   0.5, 1, 0b1000, 1};
    cost_table_["call"]     = {"call",     1.0, 3, 0b1000, 3};
    cost_table_["ret"]      = {"ret",      1.0, 3, 0b1000, 2};
    
    // SIMD
    cost_table_["vadd"]     = {"vadd",     0.5, 1, 0b0011, 1};   // AVX add
    cost_table_["vmul"]     = {"vmul",     1.0, 4, 0b0001, 1};   // AVX multiply
    cost_table_["vfma"]     = {"vfma",     0.5, 4, 0b0011, 1};   // FMA
    cost_table_["vload"]    = {"vload",    0.5, 5, 0b0010, 1};   // AVX load (256-bit)
    cost_table_["vstore"]   = {"vstore",   1.0, 5, 0b0100, 1};   // AVX store
    
    // Comparison / conditional
    cost_table_["cmp"]   = {"cmp",   0.25, 1, 0b1111, 1};
    cost_table_["cmov"]  = {"cmov",  0.5,  1, 0b0011, 1};
    cost_table_["nop"]   = {"nop",   0.25, 0, 0b1111, 0};
}

// ── Instruction-level costs ──────────────────────────

double CostModel::instruction_cost(const std::string& opcode) const {
    auto it = cost_table_.find(opcode);
    if (it != cost_table_.end()) {
        return it->second.throughput;
    }
    return 1.0; // unknown instruction = assume 1 cycle
}

double CostModel::instruction_cost(const std::string& opcode,
                                    uint32_t bit_width,
                                    bool is_float) const {
    double base = instruction_cost(opcode);
    
    if (bit_width == 0) return base;
    
    // Type-dependent scaling
    if (is_float) {
        if (opcode == "add" || opcode == "sub") {
            // FP add: typically 3-4 cycle latency, 0.5-1.0 throughput
            base = (bit_width == 64) ? 1.0 : 0.5;
        } else if (opcode == "mul") {
            // FP mul: 4-5 cycle latency
            base = (bit_width == 64) ? 1.0 : 0.5;
        } else if (opcode == "div") {
            // FP div: 10-20 cycles for f32, 15-25 for f64
            base = (bit_width == 64) ? 15.0 : 10.0;
        }
    } else {
        // Integer type scaling
        if (opcode == "mul") {
            if (bit_width == 8) base *= 0.8;
            else if (bit_width == 64) base *= 1.2; // imul r64 is slower
        } else if (opcode == "div") {
            // Integer division scales roughly with bit width
            if (bit_width <= 8) base *= 0.3;
            else if (bit_width == 16) base *= 0.5;
            else if (bit_width == 32) base *= 1.0;
            else base *= 1.5; // 64-bit div is expensive
        }
    }
    
    return base;
}

// ── Calibration correction ───────────────────────────

void CostModel::set_correction_factors(const CorrectionFactors& factors) {
    corrections_ = factors;
    
    // Scale ALU ops
    for (auto& name : {"add", "sub", "and", "or", "xor", "shift", "cmp", "cmov", "nop"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end()) {
            it->second.throughput *= factors.alu;
            it->second.latency = static_cast<uint32_t>(
                it->second.latency * factors.alu + 0.5);
        }
    }
    
    // Scale multiply
    for (auto& name : {"mul", "imul", "vmul"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end()) {
            it->second.throughput *= factors.mul;
            it->second.latency = static_cast<uint32_t>(
                it->second.latency * factors.mul + 0.5);
        }
    }
    
    // Scale divide
    {
        auto it = cost_table_.find("div");
        if (it != cost_table_.end()) {
            it->second.throughput *= factors.div;
            it->second.latency = static_cast<uint32_t>(
                it->second.latency * factors.div + 0.5);
        }
    }
    
    // Scale loads
    for (auto& name : {"load", "vload"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end())
            it->second.throughput *= factors.load;
    }
    
    // Scale stores
    for (auto& name : {"store", "vstore"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end())
            it->second.throughput *= factors.store;
    }
    
    // Scale branch
    for (auto& name : {"branch", "call", "ret"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end())
            it->second.throughput *= factors.branch;
    }
    
    // Scale SIMD
    for (auto& name : {"vadd", "vfma"}) {
        auto it = cost_table_.find(name);
        if (it != cost_table_.end()) {
            it->second.throughput *= factors.simd;
            it->second.latency = static_cast<uint32_t>(
                it->second.latency * factors.simd + 0.5);
        }
    }
    
    // Scale cache latencies in HWProfile
    hw_.l1d.latency_cycles = static_cast<uint32_t>(
        hw_.l1d.latency_cycles * factors.l1_latency + 0.5);
    hw_.l2.latency_cycles = static_cast<uint32_t>(
        hw_.l2.latency_cycles * factors.l2_latency + 0.5);
    hw_.l3.latency_cycles = static_cast<uint32_t>(
        hw_.l3.latency_cycles * factors.l3_latency + 0.5);
    // DRAM latency is in ns, scale it too
    hw_.memory_latency_ns = static_cast<uint32_t>(
        hw_.memory_latency_ns * factors.dram_latency + 0.5);
}

double CostModel::sequence_throughput(const std::vector<std::string>& opcodes) const {
    if (opcodes.empty()) return 0.0;
    
    // Model port contention:
    // Count how many uops go to each port, find the bottleneck port
    std::vector<double> port_pressure(hw_.execution_ports, 0.0);
    
    for (const auto& op : opcodes) {
        auto it = cost_table_.find(op);
        if (it == cost_table_.end()) {
            // Unknown op — distribute evenly across all ports
            for (auto& p : port_pressure) {
                p += 1.0 / hw_.execution_ports;
            }
            continue;
        }
        
        const auto& cost = it->second;
        // Count available ports for this instruction
        uint32_t available_ports = 0;
        for (uint32_t i = 0; i < hw_.execution_ports; i++) {
            if (cost.port_mask & (1 << i)) available_ports++;
        }
        
        // Distribute pressure evenly across available ports
        if (available_ports > 0) {
            double pressure_per_port = cost.throughput / available_ports;
            for (uint32_t i = 0; i < hw_.execution_ports; i++) {
                if (cost.port_mask & (1 << i)) {
                    port_pressure[i] += pressure_per_port;
                }
            }
        }
    }
    
    // Throughput = bottleneck port
    return *std::max_element(port_pressure.begin(), port_pressure.end());
}

double CostModel::chain_latency(const std::vector<std::string>& chain) const {
    double total = 0.0;
    for (const auto& op : chain) {
        auto it = cost_table_.find(op);
        if (it != cost_table_.end()) {
            total += it->second.latency;
        } else {
            total += 1.0;
        }
    }
    return total;
}

// ── Memory access costs ──────────────────────────────

double CostModel::cache_miss_probability(uint64_t working_set_bytes, 
                                           const CacheLevel& cache) const {
    // Simple model: if working set fits in cache, hit rate ≈ 1.0
    // If working set > cache, hit rate degrades proportionally
    // More sophisticated: account for associativity conflicts
    
    uint64_t cache_bytes = static_cast<uint64_t>(cache.size_kb) * 1024;
    
    if (working_set_bytes <= cache_bytes) {
        // Fits entirely — small conflict miss probability
        // Conflict misses depend on associativity
        double conflict_rate = 1.0 / (cache.associativity * 2.0);
        return conflict_rate * (static_cast<double>(working_set_bytes) / cache_bytes);
    }
    
    // Doesn't fit — miss rate proportional to overflow
    double overflow_ratio = static_cast<double>(working_set_bytes) / cache_bytes;
    return std::min(1.0, 1.0 - (1.0 / overflow_ratio));
}

CostModel::CacheDistribution CostModel::cache_distribution(uint64_t working_set_bytes) const {
    CacheDistribution dist;
    
    double l1_miss = cache_miss_probability(working_set_bytes, hw_.l1d);
    double l2_miss = cache_miss_probability(working_set_bytes, hw_.l2);
    double l3_miss = cache_miss_probability(working_set_bytes, hw_.l3);
    
    dist.l1_hit_rate = 1.0 - l1_miss;
    dist.l2_hit_rate = l1_miss * (1.0 - l2_miss);
    dist.l3_hit_rate = l1_miss * l2_miss * (1.0 - l3_miss);
    dist.dram_rate   = l1_miss * l2_miss * l3_miss;
    
    return dist;
}

double CostModel::avg_memory_latency(uint64_t working_set_bytes) const {
    auto dist = cache_distribution(working_set_bytes);
    
    return dist.l1_hit_rate * hw_.l1d.latency_cycles
         + dist.l2_hit_rate * hw_.l2.latency_cycles
         + dist.l3_hit_rate * hw_.l3.latency_cycles
         + dist.dram_rate   * (hw_.memory_latency_ns * hw_.base_freq_mhz / 1000.0);
}

double CostModel::memory_access_cost(uint64_t working_set_bytes, 
                                       uint32_t access_stride) const {
    double base_latency = avg_memory_latency(working_set_bytes);
    
    // Stride penalty: non-sequential access defeats hardware prefetcher
    if (access_stride > hw_.l1d.line_size) {
        // Strided access crosses cache lines — prefetcher might not help
        double stride_penalty = std::log2(access_stride / hw_.l1d.line_size + 1);
        base_latency *= (1.0 + stride_penalty * 0.3);
    }
    
    return base_latency;
}

// ── Loop costs ───────────────────────────────────────

CostResult CostModel::loop_cost(const std::vector<std::string>& body_opcodes,
                                 uint64_t trip_count,
                                 uint64_t working_set_bytes) const {
    CostResult result;
    
    // Compute body throughput and latency
    double body_throughput = sequence_throughput(body_opcodes);
    double body_latency = chain_latency(body_opcodes);
    
    // Loop overhead: branch + counter increment
    double loop_overhead = instruction_cost("branch") + instruction_cost("add");
    
    // Total throughput = (body + overhead) * iterations
    result.throughput_cycles = (body_throughput + loop_overhead) * trip_count;
    
    // Latency = body latency chain * iterations (if fully serial)
    result.latency_cycles = body_latency * trip_count;
    
    // Cache pressure
    result.cache_pressure = cache_miss_probability(working_set_bytes, hw_.l1d);
    
    // Register pressure (estimate from number of operations)
    uint32_t estimated_live = std::min(static_cast<uint32_t>(body_opcodes.size() / 3 + 2), 
                                        GP_REGISTERS);
    result.register_pressure = static_cast<double>(estimated_live) / GP_REGISTERS;
    
    // Branch risk: loop back-edge is predictable (almost always taken)
    result.branch_risk = 1.0 / trip_count; // only the last iteration mispredicts
    
    return result;
}

CostResult CostModel::unrolled_loop_cost(const std::vector<std::string>& body_opcodes,
                                           uint64_t trip_count,
                                           uint32_t unroll_factor,
                                           uint64_t working_set_bytes) const {
    CostResult result;
    
    if (unroll_factor == 0 || unroll_factor == 1) {
        return loop_cost(body_opcodes, trip_count, working_set_bytes);
    }
    
    uint64_t unrolled_iterations = trip_count / unroll_factor;
    uint64_t remainder = trip_count % unroll_factor;
    
    // Unrolled body: N copies of the body, more ILP opportunities
    // Throughput improves because OoO can overlap independent iterations
    std::vector<std::string> unrolled_body;
    for (uint32_t i = 0; i < unroll_factor; i++) {
        unrolled_body.insert(unrolled_body.end(), 
                              body_opcodes.begin(), body_opcodes.end());
    }
    
    double body_throughput = sequence_throughput(unrolled_body);
    double loop_overhead = instruction_cost("branch") + instruction_cost("add");
    
    result.throughput_cycles = (body_throughput + loop_overhead) * unrolled_iterations;
    
    // Add remainder loop cost
    if (remainder > 0) {
        result.throughput_cycles += sequence_throughput(body_opcodes) * remainder;
    }
    
    // Latency — less loop overhead
    result.latency_cycles = chain_latency(body_opcodes) * trip_count;
    
    // Cache pressure increases with unrolling (more code in L1i)
    uint64_t code_size = body_opcodes.size() * unroll_factor * 4; // ~4 bytes per insn
    uint64_t l1i_bytes = static_cast<uint64_t>(hw_.l1i.size_kb) * 1024;
    
    // Data cache pressure from working set
    double data_pressure = cache_miss_probability(working_set_bytes, hw_.l1d);
    
    // I-cache pressure: unrolled code competes with other functions.
    // Reserve ~50% of L1i for other code. If unrolled body exceeds
    // the remaining half, i-cache thrashing becomes significant.
    double icache_pressure = cache_miss_probability(code_size * 2, hw_.l1i);
    
    // Hard penalty: body > L1i/4 is almost certainly too big
    if (code_size > l1i_bytes / 4) {
        icache_pressure = std::max(icache_pressure, 0.8);
    }
    
    result.cache_pressure = data_pressure + icache_pressure;
    result.cache_pressure = std::min(1.5, result.cache_pressure); // allow > 1.0 to kill bad unrolls
    
    // Register pressure: unrolling increases live ranges
    // For large bodies, register allocator will spill aggressively
    uint32_t base_live = static_cast<uint32_t>(body_opcodes.size() / 3 + 2);
    uint32_t unrolled_live = base_live + (unroll_factor - 1) * (base_live / 2);
    // Spill cost: each spill = store + load + L1d latency
    uint32_t spills = (unrolled_live > GP_REGISTERS) 
        ? (unrolled_live - GP_REGISTERS) : 0;
    result.register_pressure = static_cast<double>(unrolled_live) / GP_REGISTERS
        + spills * (instruction_cost("store") + instruction_cost("load") + hw_.l1d.latency_cycles)
          / std::max(result.throughput_cycles, 1.0);
    
    // Branch risk decreases (fewer branches)
    result.branch_risk = 1.0 / unrolled_iterations;
    
    return result;
}

CostResult CostModel::vectorized_loop_cost(const std::vector<std::string>& body_opcodes,
                                             uint64_t trip_count,
                                             uint32_t vector_width,
                                             uint64_t working_set_bytes) const {
    CostResult result;
    
    // How many scalar operations fit in one SIMD operation?
    uint32_t scalar_width = 32; // assume 32-bit elements
    uint32_t lanes = vector_width / scalar_width;
    
    // Vectorized iterations
    uint64_t vec_iterations = trip_count / lanes;
    uint64_t remainder = trip_count % lanes;
    
    // Replace scalar ops with SIMD equivalents
    std::vector<std::string> vec_body;
    for (const auto& op : body_opcodes) {
        if (op == "add" || op == "sub") vec_body.push_back("vadd");
        else if (op == "mul") vec_body.push_back("vmul");
        else if (op == "load") vec_body.push_back("vload");
        else if (op == "store") vec_body.push_back("vstore");
        else vec_body.push_back(op); // non-vectorizable ops stay scalar
    }
    
    double body_throughput = sequence_throughput(vec_body);
    double loop_overhead = instruction_cost("branch") + instruction_cost("add");
    
    result.throughput_cycles = (body_throughput + loop_overhead) * vec_iterations;
    
    // Add scalar remainder
    if (remainder > 0) {
        result.throughput_cycles += sequence_throughput(body_opcodes) * remainder;
    }
    
    result.latency_cycles = chain_latency(vec_body) * vec_iterations;
    
    // SIMD has higher memory bandwidth requirements
    result.cache_pressure = cache_miss_probability(working_set_bytes, hw_.l1d) * 1.2;
    result.cache_pressure = std::min(1.0, result.cache_pressure);
    
    // SIMD uses separate register file
    result.register_pressure = static_cast<double>(vec_body.size() / 3 + 2) / SIMD_REGISTERS;
    
    result.branch_risk = 1.0 / vec_iterations;
    
    return result;
}

// ── Function call costs ──────────────────────────────

double CostModel::call_overhead() const {
    // call instruction + stack frame setup + arg passing + ret
    return instruction_cost("call") 
         + instruction_cost("ret") 
         + 2.0 * instruction_cost("store")  // push rbp, save regs
         + 2.0 * instruction_cost("load");  // pop rbp, restore regs
}

CostResult CostModel::inline_cost(uint32_t instruction_count,
                                    double caller_register_pressure,
                                    uint64_t callee_working_set) const {
    CostResult result;
    
    // Inlined: no call overhead, but more code in caller
    result.throughput_cycles = instruction_count * 0.5; // average IPC ≈ 2
    result.latency_cycles = instruction_count * 0.3;    // ILP helps
    
    // More code = more L1i pressure
    uint64_t extra_code_bytes = instruction_count * 4;
    result.cache_pressure = cache_miss_probability(callee_working_set + extra_code_bytes, hw_.l1d);
    
    // Register pressure increases when inlined
    double combined_pressure = caller_register_pressure + 
                                (static_cast<double>(instruction_count) / (GP_REGISTERS * 3));
    result.register_pressure = std::min(1.5, combined_pressure);
    
    result.branch_risk = 0.0; // no call/ret branch
    
    return result;
}

CostResult CostModel::call_cost(uint32_t instruction_count,
                                  double caller_register_pressure) const {
    CostResult result;
    
    // Not inlined: call overhead + function body
    result.throughput_cycles = call_overhead() + instruction_count * 0.5;
    result.latency_cycles = call_overhead() + instruction_count * 0.5;
    
    // Separate function = separate L1i footprint (might be cold)
    result.cache_pressure = 0.1; // function might not be in L1i
    
    // Register pressure is isolated (callee-saved regs)
    result.register_pressure = caller_register_pressure;
    
    // call/ret has branch prediction cost
    result.branch_risk = 0.02; // indirect call mispredict rate ~2%
    
    return result;
}

// ── Branch costs ─────────────────────────────────────

double CostModel::branch_cost(double taken_probability) const {
    // Branch predictor models: assume 2-bit saturating counter
    // Mispredict rate ≈ min(taken_prob, 1 - taken_prob) for biased branches
    // For 50/50: mispredict ≈ 10-15% (modern predictors are better)
    
    double mispredict_prob;
    if (taken_probability < 0.1 || taken_probability > 0.9) {
        // Highly biased — predictor handles well
        mispredict_prob = std::min(taken_probability, 1.0 - taken_probability) * 0.5;
    } else {
        // Unpredictable — predictor struggles
        mispredict_prob = 0.15 + 0.2 * (0.5 - std::abs(taken_probability - 0.5));
    }
    
    return instruction_cost("branch") + 
           mispredict_prob * hw_.branch_mispredict_penalty;
}

double CostModel::cmov_cost() const {
    // Conditional move: always executes both paths' data deps, no branch
    return instruction_cost("cmov");
}

// ── Register pressure ────────────────────────────────

double CostModel::register_spill_cost(uint32_t live_variables) const {
    if (live_variables <= GP_REGISTERS) {
        return 0.0; // everything fits
    }
    
    // Each spilled variable = 1 store + 1 load
    uint32_t spills = live_variables - GP_REGISTERS;
    return spills * (instruction_cost("store") + instruction_cost("load") + 
                      hw_.l1d.latency_cycles);
}

} // namespace costforge
