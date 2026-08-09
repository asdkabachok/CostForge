#include "costforge/pattern_recognizer.h"
#include "costforge/loop_tiling.h"
#include <algorithm>
#include <cmath>
#include <regex>

namespace costforge {

const char* pattern_name(PatternKind kind) {
    switch (kind) {
        case PatternKind::None:             return "none";
        case PatternKind::ReductionSum:     return "reduction_sum";
        case PatternKind::ReductionMin:     return "reduction_min";
        case PatternKind::ReductionMax:     return "reduction_max";
        case PatternKind::ReductionProduct: return "reduction_product";
        case PatternKind::MatrixMultiply:   return "matrix_multiply";
        case PatternKind::MatrixTranspose:  return "matrix_transpose";
        case PatternKind::DotProduct:       return "dot_product";
        case PatternKind::AXPY:            return "axpy";
        case PatternKind::Stencil1D:       return "stencil_1d";
        case PatternKind::Stencil2D:       return "stencil_2d";
        case PatternKind::Memcopy:         return "memcopy";
        case PatternKind::Gather:          return "gather";
        case PatternKind::Scatter:         return "scatter";
        case PatternKind::StreamStore:     return "stream_store";
        case PatternKind::HistogramUpdate: return "histogram_update";
        case PatternKind::PrefixScan:      return "prefix_scan";
        case PatternKind::BranchChain:     return "branch_chain";
        case PatternKind::PredicatedLoop:  return "predicated_loop";
    }
    return "unknown";
}

PatternRecognizer::PatternRecognizer(const HWProfile& hw) : hw_(hw) {}

// ── Helpers ──────────────────────────────────────────

uint32_t PatternRecognizer::optimal_tile_size() const {
    // v0.3.2: delegate to the real cache-blocking model. Kept for
    // back-compat; returns the representative L1 contraction edge (Kc).
    LoopTiler tiler(hw_);
    return tiler.square_tile_edge(4 /* f32 */);
}

bool PatternRecognizer::has_fma() const {
    // FMA is available on AVX2+ (Haswell/Zen+)
    return hw_.simd.avx2;
}

bool PatternRecognizer::should_use_streaming_stores(uint64_t working_set) const {
    // Use non-temporal stores when output doesn't fit in L3
    // (avoids polluting cache with write-only data)
    uint64_t l3_bytes = static_cast<uint64_t>(hw_.l3.size_kb) * 1024;
    return working_set > l3_bytes;
}

// ── Reduction detection ──────────────────────────────

PatternMatch PatternRecognizer::detect_reduction(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    match.confidence = 0.0;
    
    if (!loop.has_accumulator) return match;
    
    // Map accumulator op to pattern kind
    if (loop.accumulator_op == "add") {
        match.kind = PatternKind::ReductionSum;
    } else if (loop.accumulator_op == "mul") {
        match.kind = PatternKind::ReductionProduct;
    } else if (loop.accumulator_op == "min") {
        match.kind = PatternKind::ReductionMin;
    } else if (loop.accumulator_op == "max") {
        match.kind = PatternKind::ReductionMax;
    } else {
        return match;
    }
    
    // Confidence: high if simple loop body + single array read + accumulator
    double conf = 0.5;
    
    // Simple body (few ops) → more likely a pure reduction
    if (loop.body_opcodes.size() <= 6) conf += 0.2;
    
    // Single array read → classic reduction
    uint32_t reads = 0;
    for (const auto& acc : loop.accesses) {
        if (acc.is_read && acc.index_expr == "i") reads++;
    }
    if (reads == 1) conf += 0.2;
    
    // No loop-carried dependency other than the accumulator
    if (!loop.has_loop_carried_dep) conf += 0.1;
    
    match.confidence = std::min(1.0, conf);
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    match.location = "loop";
    
    // Recommend horizontal SIMD reduction
    uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
    if (simd_lanes >= 4 && loop.trip_count >= simd_lanes * 4) {
        match.suggested_transform = "simd_horizontal_reduction_" 
            + std::to_string(hw_.simd.max_width()) + "bit";
        match.estimated_speedup = static_cast<double>(simd_lanes) * 0.7;
        // 0.7 factor: horizontal reduce adds overhead at the end
    } else {
        match.suggested_transform = "unroll_accumulate_4x";
        match.estimated_speedup = 2.5;
    }
    
    return match;
}

// ── Matrix multiply detection ────────────────────────

PatternMatch PatternRecognizer::detect_matrix_multiply(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // Matrix multiply signature:
    // - 3 nested loops (nesting_depth >= 2 for the innermost)
    // - 3 array accesses: A[i][k], B[k][j], C[i][j]
    // - Accumulator: C[i][j] += A[i][k] * B[k][j]
    
    if (loop.nesting_depth < 2) return match;
    if (!loop.has_accumulator || loop.accumulator_op != "add") return match;
    if (loop.accesses.size() < 3) return match;
    
    // Look for the characteristic access pattern:
    // one read with "i*N+k" or "i,k", one read with "k*N+j" or "k,j",
    // one read+write with "i*N+j" or "i,j"
    bool has_a_access = false, has_b_access = false, has_c_access = false;
    
    for (const auto& acc : loop.accesses) {
        const auto& idx = acc.index_expr;
        if (acc.is_read && !acc.is_write) {
            if (idx.find("k") != std::string::npos) {
                if (idx.find("i") != std::string::npos) has_a_access = true;
                if (idx.find("j") != std::string::npos) has_b_access = true;
            }
        }
        if (acc.is_read && acc.is_write) {
            if (idx.find("i") != std::string::npos && 
                idx.find("j") != std::string::npos &&
                idx.find("k") == std::string::npos) {
                has_c_access = true;
            }
        }
    }
    
    if (!has_a_access || !has_b_access || !has_c_access) return match;
    
    // Must have a multiply in the body
    bool has_mul = false;
    for (const auto& op : loop.body_opcodes) {
        if (op == "mul" || op == "imul" || op == "vmul" || op == "vfma") {
            has_mul = true;
            break;
        }
    }
    if (!has_mul) return match;
    
    match.kind = PatternKind::MatrixMultiply;
    match.confidence = 0.85;
    match.dimensions = 2;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    // Recommend tiled + vectorized matmul. Tile sizes now come from the
    // real cache-blocking model (LoopTiler), not a single square heuristic.
    uint32_t elem_bytes = std::max<uint32_t>(loop.data_width_bits / 8, 1);
    LoopTiler tiler(hw_);
    LoopTiler::TilingPlan plan = tiler.tile_matmul(elem_bytes, loop.trip_count);
    match.suggested_transform = "tiled_matmul_" + plan.descriptor();
    if (has_fma()) {
        match.suggested_transform += "_fma";
    }

    // Speedup estimate: cache-reuse speedup (from the model) times the
    // SIMD width factor. No magic constants — both factors are derived.
    uint64_t l1_bytes = static_cast<uint64_t>(hw_.l1d.size_kb) * 1024;
    if (loop.working_set_bytes > l1_bytes * 4) {
        uint32_t simd_factor = std::max<uint32_t>(
            hw_.simd.max_width() / std::max<uint32_t>(loop.data_width_bits, 1), 1);
        match.estimated_speedup = plan.estimated_speedup * simd_factor * 0.5;
        match.estimated_speedup = std::min(match.estimated_speedup, 20.0);
        match.estimated_speedup = std::max(match.estimated_speedup, 1.0);
    } else {
        match.estimated_speedup = 2.0;
    }
    
    return match;
}

// ── Dot product detection ────────────────────────────

PatternMatch PatternRecognizer::detect_dot_product(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // Dot product: sum += a[i] * b[i]
    // - Single loop (nesting_depth == 0)
    // - Accumulator with "add"
    // - Exactly 2 array reads with index "i"
    // - A multiply in the body
    
    if (loop.nesting_depth != 0) return match;
    if (!loop.has_accumulator || loop.accumulator_op != "add") return match;
    
    uint32_t linear_reads = 0;
    for (const auto& acc : loop.accesses) {
        if (acc.is_read && acc.index_expr == "i") linear_reads++;
    }
    if (linear_reads != 2) return match;
    
    bool has_mul = false;
    for (const auto& op : loop.body_opcodes) {
        if (op == "mul" || op == "vmul" || op == "vfma") {
            has_mul = true;
            break;
        }
    }
    if (!has_mul) return match;
    
    match.kind = PatternKind::DotProduct;
    match.confidence = 0.9;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    if (has_fma()) {
        match.suggested_transform = "simd_fma_dot_product";
        uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
        match.estimated_speedup = simd_lanes * 0.85; // FMA does mul+add in 1
    } else {
        match.suggested_transform = "simd_dot_product";
        uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
        match.estimated_speedup = simd_lanes * 0.6;
    }
    
    return match;
}

// ── AXPY detection ───────────────────────────────────

PatternMatch PatternRecognizer::detect_axpy(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // AXPY: y[i] = a * x[i] + y[i]
    // - Single loop, no accumulator across iterations
    // - 2 array accesses: one read (x), one read+write (y)
    // - multiply + add in body
    
    if (loop.nesting_depth != 0) return match;
    if (loop.has_loop_carried_dep) return match;
    
    uint32_t reads = 0, read_writes = 0;
    for (const auto& acc : loop.accesses) {
        if (acc.is_read && !acc.is_write && acc.index_expr == "i") reads++;
        if (acc.is_read && acc.is_write && acc.index_expr == "i") read_writes++;
    }
    if (reads != 1 || read_writes != 1) return match;
    
    bool has_mul = false, has_add = false;
    for (const auto& op : loop.body_opcodes) {
        if (op == "mul" || op == "vmul") has_mul = true;
        if (op == "add" || op == "vadd") has_add = true;
    }
    if (!has_mul || !has_add) return match;
    
    match.kind = PatternKind::AXPY;
    match.confidence = 0.85;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    if (has_fma()) {
        match.suggested_transform = "simd_fma_axpy";
    } else {
        match.suggested_transform = "simd_axpy";
    }
    
    uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
    match.estimated_speedup = simd_lanes * 0.8;
    
    if (should_use_streaming_stores(loop.working_set_bytes)) {
        match.suggested_transform += "_nontemporal";
        match.estimated_speedup *= 1.3;
    }
    
    return match;
}

// ── Stencil detection ────────────────────────────────

PatternMatch PatternRecognizer::detect_stencil(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // Stencil: out[i] = f(in[i-1], in[i], in[i+1])
    // - Array accesses with offset indices (i-1, i, i+1)
    // - Or for 2D: (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    
    bool has_offset_minus = false, has_offset_plus = false, has_center = false;
    std::string array_name;
    
    for (const auto& acc : loop.accesses) {
        if (!acc.is_read) continue;
        const auto& idx = acc.index_expr;
        
        if (idx.find("-1") != std::string::npos || 
            idx.find("- 1") != std::string::npos) {
            has_offset_minus = true;
            array_name = acc.array_name;
        }
        if (idx.find("+1") != std::string::npos || 
            idx.find("+ 1") != std::string::npos) {
            has_offset_plus = true;
        }
        if (idx == "i" || idx == "i,j" || idx == "i*N+j") {
            has_center = true;
        }
    }
    
    if (!has_offset_minus || !has_offset_plus || !has_center) return match;
    
    if (loop.nesting_depth >= 1) {
        match.kind = PatternKind::Stencil2D;
        match.dimensions = 2;
    } else {
        match.kind = PatternKind::Stencil1D;
        match.dimensions = 1;
    }
    
    match.confidence = 0.8;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    // Stencils benefit from vectorization + prefetching
    uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
    match.suggested_transform = "vectorized_stencil_" 
        + std::to_string(match.dimensions) + "d_prefetch";
    match.estimated_speedup = simd_lanes * 0.6; // data reuse limits gains
    
    return match;
}

// ── Memcopy detection ────────────────────────────────

PatternMatch PatternRecognizer::detect_memcopy(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // dst[i] = src[i]: one read + one write, both linear index "i"
    if (loop.has_accumulator) return match;
    if (loop.body_opcodes.size() > 4) return match; // should be trivial body
    
    uint32_t linear_reads = 0, linear_writes = 0;
    for (const auto& acc : loop.accesses) {
        if (acc.index_expr != "i") return match; // non-linear = not memcpy
        if (acc.is_read && !acc.is_write) linear_reads++;
        if (acc.is_write) linear_writes++;
    }
    
    if (linear_reads != 1 || linear_writes != 1) return match;
    
    match.kind = PatternKind::Memcopy;
    match.confidence = 0.95;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    if (should_use_streaming_stores(loop.working_set_bytes)) {
        match.suggested_transform = "rep_movsb_or_nontemporal_simd";
        match.estimated_speedup = 3.0;
    } else {
        match.suggested_transform = "simd_memcpy_" 
            + std::to_string(hw_.simd.max_width()) + "bit";
        uint32_t simd_lanes = hw_.simd.max_width() / loop.data_width_bits;
        match.estimated_speedup = simd_lanes * 0.9;
    }
    
    return match;
}

// ── Gather/Scatter detection ─────────────────────────

PatternMatch PatternRecognizer::detect_gather_scatter(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    for (const auto& acc : loop.accesses) {
        // Indirect indexing: idx[i], data[idx[i]], etc.
        if (acc.index_expr.find("idx") != std::string::npos ||
            acc.index_expr.find("[i]") != std::string::npos) {
            if (acc.is_read && !acc.is_write) {
                match.kind = PatternKind::Gather;
            } else if (acc.is_write) {
                match.kind = PatternKind::Scatter;
            }
        }
    }
    
    if (match.kind == PatternKind::None) return match;
    
    match.confidence = 0.7;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    if (hw_.simd.avx2) {
        match.suggested_transform = "avx2_gather_intrinsic";
        match.estimated_speedup = 1.5; // gather is slow, but better than scalar
    } else {
        match.suggested_transform = "prefetch_hint_gather";
        match.estimated_speedup = 1.2;
    }
    
    return match;
}

// ── Histogram detection ──────────────────────────────

PatternMatch PatternRecognizer::detect_histogram(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // hist[data[i]]++: indirect write with an increment
    if (!loop.has_accumulator) return match;
    if (loop.accumulator_op != "add") return match;
    
    bool has_indirect_write = false;
    for (const auto& acc : loop.accesses) {
        if (acc.is_write && acc.is_read &&
            (acc.index_expr.find("[") != std::string::npos ||
             acc.index_expr.find("data") != std::string::npos)) {
            has_indirect_write = true;
        }
    }
    
    if (!has_indirect_write) return match;
    
    match.kind = PatternKind::HistogramUpdate;
    match.confidence = 0.7;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    // Histograms are hard to vectorize due to conflicts
    // But we can replicate the histogram array per SIMD lane
    // and merge at the end
    if (loop.working_set_bytes <= static_cast<uint64_t>(hw_.l1d.size_kb) * 1024 / 4) {
        match.suggested_transform = "replicated_histogram_merge";
        match.estimated_speedup = 2.0;
    } else {
        match.suggested_transform = "prefetch_histogram";
        match.estimated_speedup = 1.3;
    }
    
    return match;
}

// ── Prefix scan detection ────────────────────────────

PatternMatch PatternRecognizer::detect_prefix_scan(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // out[i] = out[i-1] + in[i]: loop-carried dependency on output
    if (!loop.has_loop_carried_dep) return match;
    if (!loop.has_accumulator) return match;
    
    bool has_prev_access = false;
    for (const auto& acc : loop.accesses) {
        if (acc.is_read && 
            (acc.index_expr == "i-1" || acc.index_expr == "i - 1")) {
            has_prev_access = true;
        }
    }
    
    if (!has_prev_access) return match;
    
    match.kind = PatternKind::PrefixScan;
    match.confidence = 0.75;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    // Blelloch-style parallel prefix scan
    match.suggested_transform = "simd_prefix_scan_blelloch";
    match.estimated_speedup = 1.5; // limited by dependency chain
    
    return match;
}

// ── Branch chain detection ───────────────────────────

PatternMatch PatternRecognizer::detect_branch_chain(const BlockInfo& block) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    if (!block.is_chain_link) return match;
    if (block.chain_length < 4) return match; // not worth converting
    
    match.kind = PatternKind::BranchChain;
    match.confidence = 0.85;
    match.location = block.label;
    
    if (block.chain_length >= 8) {
        match.suggested_transform = "jump_table";
        match.estimated_speedup = 
            static_cast<double>(block.chain_length) / 2.0; // O(n) → O(1)
    } else {
        match.suggested_transform = "binary_branch_tree";
        match.estimated_speedup = 
            static_cast<double>(block.chain_length) / std::log2(block.chain_length);
    }
    
    return match;
}

// ── Predicated loop detection ────────────────────────

PatternMatch PatternRecognizer::detect_predicated_loop(const LoopInfo& loop) const {
    PatternMatch match;
    match.kind = PatternKind::None;
    
    // A loop where most iterations skip due to a branch inside
    uint32_t branch_ops = 0;
    for (const auto& op : loop.body_opcodes) {
        if (op == "branch" || op == "cmp") branch_ops++;
    }
    
    if (branch_ops < 2) return match;
    
    double branch_ratio = static_cast<double>(branch_ops) / loop.body_opcodes.size();
    if (branch_ratio < 0.2) return match;
    
    match.kind = PatternKind::PredicatedLoop;
    match.confidence = 0.6;
    match.trip_count = loop.trip_count;
    match.working_set = loop.working_set_bytes;
    
    match.suggested_transform = "masked_simd_predication";
    match.estimated_speedup = 1.8;
    
    return match;
}

// ── Main entry points ────────────────────────────────

std::vector<PatternMatch> PatternRecognizer::analyze_loop(const LoopInfo& loop) const {
    std::vector<PatternMatch> results;
    
    // Run all detectors, collect matches above confidence threshold
    auto try_add = [&](PatternMatch m) {
        if (m.kind != PatternKind::None && m.confidence >= 0.5) {
            results.push_back(std::move(m));
        }
    };
    
    try_add(detect_reduction(loop));
    try_add(detect_matrix_multiply(loop));
    try_add(detect_dot_product(loop));
    try_add(detect_axpy(loop));
    try_add(detect_stencil(loop));
    try_add(detect_memcopy(loop));
    try_add(detect_gather_scatter(loop));
    try_add(detect_histogram(loop));
    try_add(detect_prefix_scan(loop));
    try_add(detect_predicated_loop(loop));
    
    // Sort by estimated speedup (best opportunity first)
    std::sort(results.begin(), results.end(),
              [](const PatternMatch& a, const PatternMatch& b) {
                  return a.estimated_speedup > b.estimated_speedup;
              });
    
    return results;
}

std::vector<PatternMatch> PatternRecognizer::analyze_block(const BlockInfo& block) const {
    std::vector<PatternMatch> results;
    
    auto bc = detect_branch_chain(block);
    if (bc.kind != PatternKind::None && bc.confidence >= 0.5) {
        results.push_back(std::move(bc));
    }
    
    return results;
}

std::string PatternRecognizer::recommend_transform(const PatternMatch& match) const {
    return match.suggested_transform;
}

double PatternRecognizer::estimate_speedup(const PatternMatch& match) const {
    return match.estimated_speedup;
}

} // namespace costforge
