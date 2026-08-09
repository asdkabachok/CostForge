#include "costforge/calibrator.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>
#include <ctime>

#ifdef __x86_64__
#include <x86intrin.h>
#endif

namespace costforge {

Calibrator::Calibrator(const HWProfile& hw) : hw_(hw) {}

// ── Timing infrastructure ───────────────────────────

uint64_t Calibrator::rdtsc() {
#ifdef __x86_64__
    unsigned int aux;
    return __rdtscp(&aux);
#else
    // Fallback: use clock_gettime
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
}

double Calibrator::run_bench(BenchFn fn, uint32_t iterations) const {
    warmup(fn, 3);
    
    std::vector<double> samples(iterations);
    for (uint32_t i = 0; i < iterations; i++) {
        samples[i] = (this->*fn)();
    }
    
    // Return median (robust to outliers from interrupts/scheduling)
    std::sort(samples.begin(), samples.end());
    return samples[iterations / 2];
}

void Calibrator::warmup(BenchFn fn, uint32_t count) const {
    for (uint32_t i = 0; i < count; i++) {
        (this->*fn)();
    }
}

// ── ALU benchmark ───────────────────────────────────

double Calibrator::bench_alu_throughput() const {
    // Measure throughput of independent ADD instructions
    // Use 8 independent chains to saturate execution ports
    constexpr uint32_t ITERATIONS = 1000000;
    
    volatile uint64_t a = 1, b = 2, c = 3, d = 4;
    volatile uint64_t e = 5, f = 6, g = 7, h = 8;
    
    uint64_t start = rdtsc();
    
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        a += 1; b += 2; c += 3; d += 4;
        e += 5; f += 6; g += 7; h += 8;
    }
    
    uint64_t end = rdtsc();
    
    // Prevent dead-code elimination
    volatile uint64_t sink = a + b + c + d + e + f + g + h;
    (void)sink;
    
    // 8 independent adds per iteration
    return static_cast<double>(end - start) / (ITERATIONS * 8.0);
}

// ── Multiply benchmarks ─────────────────────────────

double Calibrator::bench_mul_latency() const {
    // Dependent multiply chain: each mul depends on previous result
    constexpr uint32_t ITERATIONS = 100000;
    
    volatile uint64_t x = 7;
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        x = x * 13 + 1; // +1 prevents degeneration to 0
    }
    uint64_t end = rdtsc();
    
    volatile uint64_t sink = x;
    (void)sink;
    
    return static_cast<double>(end - start) / ITERATIONS;
}

double Calibrator::bench_mul_throughput() const {
    constexpr uint32_t ITERATIONS = 500000;
    
    volatile uint64_t a = 3, b = 5, c = 7, d = 11;
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        a *= 13; b *= 17; c *= 19; d *= 23;
    }
    uint64_t end = rdtsc();
    
    volatile uint64_t sink = a + b + c + d;
    (void)sink;
    
    return static_cast<double>(end - start) / (ITERATIONS * 4.0);
}

// ── Divide benchmark ────────────────────────────────

double Calibrator::bench_div_latency() const {
    constexpr uint32_t ITERATIONS = 50000;
    
    volatile uint64_t x = 0xDEADBEEFCAFE0000ULL;
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        x = x / 7 + 0xDEADBEEF; // +const prevents convergence to 0
    }
    uint64_t end = rdtsc();
    
    volatile uint64_t sink = x;
    (void)sink;
    
    return static_cast<double>(end - start) / ITERATIONS;
}

// ── Cache latency benchmarks ────────────────────────

// Pointer-chasing: defeats prefetcher, measures true latency
// Allocate array of size = cache_level, fill with random linked list,
// then chase pointers.

static double pointer_chase_latency(uint64_t buffer_size_bytes) {
    // Build a random circular linked list of cache-line-sized nodes
    const uint32_t LINE_SIZE = 64;
    uint32_t num_nodes = buffer_size_bytes / LINE_SIZE;
    if (num_nodes < 16) num_nodes = 16;
    
    // Allocate aligned buffer
    std::vector<uint64_t> buffer(buffer_size_bytes / sizeof(uint64_t), 0);
    
    // Create random permutation for pointer chain
    std::vector<uint32_t> order(num_nodes);
    std::iota(order.begin(), order.end(), 0);
    
    // Fisher-Yates shuffle with simple LCG
    uint32_t rng = 12345;
    for (uint32_t i = num_nodes - 1; i > 0; i--) {
        rng = rng * 1103515245 + 12345;
        uint32_t j = rng % (i + 1);
        std::swap(order[i], order[j]);
    }
    
    // Wire up the linked list: each node points to the next in permutation
    for (uint32_t i = 0; i < num_nodes; i++) {
        uint32_t current = order[i];
        uint32_t next = order[(i + 1) % num_nodes];
        // Store pointer (as index) at the start of each cache line
        buffer[current * (LINE_SIZE / sizeof(uint64_t))] = 
            next * (LINE_SIZE / sizeof(uint64_t));
    }
    
    // Chase pointers
    constexpr uint32_t CHASES = 100000;
    uint64_t idx = 0;
    
    // Warmup
    for (uint32_t i = 0; i < 1000; i++) {
        idx = buffer[idx];
    }
    
    uint64_t start;
#ifdef __x86_64__
    unsigned int aux;
    start = __rdtscp(&aux);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
    
    for (uint32_t i = 0; i < CHASES; i++) {
        idx = buffer[idx];
    }
    
    uint64_t end;
#ifdef __x86_64__
    end = __rdtscp(&aux);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
    end = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
#endif
    
    // Prevent optimization
    volatile uint64_t sink = idx;
    (void)sink;
    
    return static_cast<double>(end - start) / CHASES;
}

double Calibrator::bench_load_l1() const {
    // Buffer fits in L1d (use half of L1d to be safe)
    uint64_t size = static_cast<uint64_t>(hw_.l1d.size_kb) * 1024 / 2;
    return pointer_chase_latency(size);
}

double Calibrator::bench_load_l2() const {
    // Buffer exceeds L1d but fits in L2
    uint64_t size = static_cast<uint64_t>(hw_.l2.size_kb) * 1024 / 2;
    return pointer_chase_latency(size);
}

double Calibrator::bench_load_l3() const {
    uint64_t size = static_cast<uint64_t>(hw_.l3.size_kb) * 1024 / 2;
    return pointer_chase_latency(size);
}

double Calibrator::bench_load_dram() const {
    // Buffer that doesn't fit in any cache
    uint64_t size = static_cast<uint64_t>(hw_.l3.size_kb) * 1024 * 4;
    if (size > 256 * 1024 * 1024) size = 256 * 1024 * 1024; // cap at 256MB
    return pointer_chase_latency(size);
}

// ── Store benchmark ─────────────────────────────────

double Calibrator::bench_store_throughput() const {
    constexpr uint32_t ITERATIONS = 1000000;
    constexpr uint32_t BUF_SIZE = 4096; // fits in L1d
    
    volatile uint64_t buffer[BUF_SIZE / sizeof(uint64_t)];
    uint32_t num_elems = BUF_SIZE / sizeof(uint64_t);
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        buffer[i % num_elems] = i;
    }
    uint64_t end = rdtsc();
    
    return static_cast<double>(end - start) / ITERATIONS;
}

// ── Branch misprediction benchmark ──────────────────

double Calibrator::bench_branch_mispredict() const {
    constexpr uint32_t ITERATIONS = 100000;
    
    // Generate a pseudo-random pattern that the branch predictor can't learn
    volatile uint32_t count = 0;
    uint32_t rng = 42;
    
    // First: measure with predictable pattern (baseline)
    uint64_t start_pred = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        if (i & 1) count++; // perfectly predictable alternating
    }
    uint64_t end_pred = rdtsc();
    double predictable = static_cast<double>(end_pred - start_pred) / ITERATIONS;
    
    // Second: measure with random pattern (mispredictions)
    count = 0;
    uint64_t start_rand = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        rng = rng * 1103515245 + 12345;
        if (rng & 0x80000000) count++; // ~50% taken, unpredictable
    }
    uint64_t end_rand = rdtsc();
    double random = static_cast<double>(end_rand - start_rand) / ITERATIONS;
    
    volatile uint32_t sink = count;
    (void)sink;
    
    // Misprediction penalty ≈ (random_time - predictable_time) / mispredict_rate
    // Mispredict rate for 50/50 random ≈ 40-50% on modern predictors
    double penalty = (random - predictable) / 0.45;
    return std::max(penalty, 5.0); // clamp to at least 5 cycles
}

// ── SIMD benchmarks ─────────────────────────────────

double Calibrator::bench_simd_add() const {
#ifdef __x86_64__
    constexpr uint32_t ITERATIONS = 500000;
    
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_set1_pd(3.0);
    __m256d d = _mm256_set1_pd(4.0);
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        a = _mm256_add_pd(a, b);
        c = _mm256_add_pd(c, d);
    }
    uint64_t end = rdtsc();
    
    // Prevent elimination
    volatile double sink = _mm256_cvtsd_f64(a) + _mm256_cvtsd_f64(c);
    (void)sink;
    
    return static_cast<double>(end - start) / (ITERATIONS * 2.0);
#else
    return 1.0; // no AVX
#endif
}

double Calibrator::bench_simd_mul() const {
#ifdef __x86_64__
    constexpr uint32_t ITERATIONS = 500000;
    
    __m256d a = _mm256_set1_pd(1.000001);
    __m256d b = _mm256_set1_pd(1.000002);
    __m256d c = _mm256_set1_pd(1.000003);
    __m256d d = _mm256_set1_pd(0.999999);
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        a = _mm256_mul_pd(a, d);
        b = _mm256_mul_pd(b, c);
    }
    uint64_t end = rdtsc();
    
    volatile double sink = _mm256_cvtsd_f64(a) + _mm256_cvtsd_f64(b);
    (void)sink;
    
    return static_cast<double>(end - start) / (ITERATIONS * 2.0);
#else
    return 1.0;
#endif
}

double Calibrator::bench_simd_fma() const {
#ifdef __x86_64__
    constexpr uint32_t ITERATIONS = 500000;
    
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(0.5);
    __m256d c = _mm256_set1_pd(0.1);
    
    uint64_t start = rdtsc();
    for (uint32_t i = 0; i < ITERATIONS; i++) {
        a = _mm256_fmadd_pd(a, b, c); // a = a*b + c
    }
    uint64_t end = rdtsc();
    
    volatile double sink = _mm256_cvtsd_f64(a);
    (void)sink;
    
    return static_cast<double>(end - start) / ITERATIONS;
#else
    return 1.0;
#endif
}

// ── Main calibration routine ────────────────────────

CalibrationProfile Calibrator::calibrate() {
    CalibrationProfile profile;
    profile.cpu_model = hw_.model;
    profile.microarch = hw_.microarch;
    
    // Timestamp
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    profile.timestamp = buf;
    
    auto add_point = [&](const std::string& op, double predicted, double measured) {
        CalibrationPoint pt;
        pt.operation = op;
        pt.predicted_cycles = predicted;
        pt.measured_cycles = measured;
        pt.error_percent = (predicted > 0) 
            ? std::abs(predicted - measured) / measured * 100.0 
            : 0.0;
        profile.points.push_back(pt);
    };
    
    // ALU
    double alu_measured = run_bench(&Calibrator::bench_alu_throughput);
    double alu_predicted = 0.25; // from cost table
    add_point("alu_throughput", alu_predicted, alu_measured);
    profile.alu_correction = alu_measured / alu_predicted;
    
    // Multiply
    double mul_lat = run_bench(&Calibrator::bench_mul_latency);
    add_point("mul_latency", 3.0, mul_lat);
    double mul_thr = run_bench(&Calibrator::bench_mul_throughput);
    add_point("mul_throughput", 1.0, mul_thr);
    profile.mul_correction = mul_lat / 3.0;
    
    // Divide
    double div_lat = run_bench(&Calibrator::bench_div_latency);
    add_point("div_latency", 20.0, div_lat);
    profile.div_correction = div_lat / 20.0;
    
    // Cache loads
    double l1_lat = run_bench(&Calibrator::bench_load_l1);
    add_point("load_l1", hw_.l1d.latency_cycles, l1_lat);
    profile.l1_latency_correction = l1_lat / hw_.l1d.latency_cycles;
    
    double l2_lat = run_bench(&Calibrator::bench_load_l2);
    add_point("load_l2", hw_.l2.latency_cycles, l2_lat);
    profile.l2_latency_correction = l2_lat / hw_.l2.latency_cycles;
    
    double l3_lat = run_bench(&Calibrator::bench_load_l3);
    add_point("load_l3", hw_.l3.latency_cycles, l3_lat);
    profile.l3_latency_correction = l3_lat / hw_.l3.latency_cycles;
    
    double dram_lat = run_bench(&Calibrator::bench_load_dram);
    double predicted_dram = hw_.memory_latency_ns * hw_.base_freq_mhz / 1000.0;
    add_point("load_dram", predicted_dram, dram_lat);
    profile.dram_latency_correction = dram_lat / predicted_dram;
    
    // Store
    double store_thr = run_bench(&Calibrator::bench_store_throughput);
    add_point("store_throughput", 1.0, store_thr);
    profile.store_correction = store_thr / 1.0;
    
    // Branch misprediction
    double branch_penalty = run_bench(&Calibrator::bench_branch_mispredict);
    add_point("branch_mispredict", hw_.branch_mispredict_penalty, branch_penalty);
    profile.branch_correction = branch_penalty / hw_.branch_mispredict_penalty;
    
    // SIMD
    double simd_add = run_bench(&Calibrator::bench_simd_add);
    add_point("simd_add", 0.5, simd_add);
    double simd_mul = run_bench(&Calibrator::bench_simd_mul);
    add_point("simd_mul", 1.0, simd_mul);
    profile.simd_correction = (simd_add / 0.5 + simd_mul / 1.0) / 2.0;
    
    // Compute mean absolute error
    double total_error = 0.0;
    for (const auto& pt : profile.points) {
        total_error += pt.error_percent;
    }
    profile.mean_absolute_error = total_error / profile.points.size();
    
    return profile;
}

// ── Apply corrections to CostModel ──────────────────

void Calibrator::apply(CostModel& model, const CalibrationProfile& cal) {
    if (!cal.is_valid()) return;
    
    CostModel::CorrectionFactors factors;
    factors.alu   = cal.alu_correction;
    factors.mul   = cal.mul_correction;
    factors.div   = cal.div_correction;
    factors.load  = cal.load_correction;
    factors.store = cal.store_correction;
    factors.branch = cal.branch_correction;
    factors.simd  = cal.simd_correction;
    factors.l1_latency   = cal.l1_latency_correction;
    factors.l2_latency   = cal.l2_latency_correction;
    factors.l3_latency   = cal.l3_latency_correction;
    factors.dram_latency = cal.dram_latency_correction;
    
    model.set_correction_factors(factors);
}

// ── JSON I/O ────────────────────────────────────────

void Calibrator::save(const CalibrationProfile& profile, 
                       const std::string& json_path) {
    std::ofstream out(json_path);
    if (!out.is_open()) return;
    
    out << "{\n";
    out << "  \"cpu_model\": \"" << profile.cpu_model << "\",\n";
    out << "  \"microarch\": \"" << profile.microarch << "\",\n";
    out << "  \"timestamp\": \"" << profile.timestamp << "\",\n";
    out << "  \"mean_absolute_error\": " << profile.mean_absolute_error << ",\n";
    out << "  \"corrections\": {\n";
    out << "    \"alu\": " << profile.alu_correction << ",\n";
    out << "    \"mul\": " << profile.mul_correction << ",\n";
    out << "    \"div\": " << profile.div_correction << ",\n";
    out << "    \"load\": " << profile.load_correction << ",\n";
    out << "    \"store\": " << profile.store_correction << ",\n";
    out << "    \"branch\": " << profile.branch_correction << ",\n";
    out << "    \"simd\": " << profile.simd_correction << ",\n";
    out << "    \"l1_latency\": " << profile.l1_latency_correction << ",\n";
    out << "    \"l2_latency\": " << profile.l2_latency_correction << ",\n";
    out << "    \"l3_latency\": " << profile.l3_latency_correction << ",\n";
    out << "    \"dram_latency\": " << profile.dram_latency_correction << "\n";
    out << "  },\n";
    out << "  \"benchmarks\": [\n";
    
    for (size_t i = 0; i < profile.points.size(); i++) {
        const auto& pt = profile.points[i];
        out << "    {"
            << "\"op\": \"" << pt.operation << "\", "
            << "\"predicted\": " << pt.predicted_cycles << ", "
            << "\"measured\": " << pt.measured_cycles << ", "
            << "\"error_pct\": " << pt.error_percent 
            << "}";
        if (i + 1 < profile.points.size()) out << ",";
        out << "\n";
    }
    
    out << "  ]\n}\n";
}

CalibrationProfile Calibrator::load(const std::string& json_path) {
    CalibrationProfile profile;
    
    std::ifstream in(json_path);
    if (!in.is_open()) {
        profile.mean_absolute_error = 999.0;
        return profile;
    }
    
    std::string content((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    
    // Minimal JSON key-value extractor (no external deps).
    // Format matches what save() produces — flat keys, known structure.
    
    auto extract_string = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\": \"";
        auto pos = content.find(search);
        if (pos == std::string::npos) return "";
        pos += search.size();
        auto end = content.find('"', pos);
        if (end == std::string::npos) return "";
        return content.substr(pos, end - pos);
    };
    
    auto extract_double = [&](const std::string& key) -> double {
        std::string search = "\"" + key + "\": ";
        auto pos = content.find(search);
        if (pos == std::string::npos) return 0.0;
        pos += search.size();
        try { return std::stod(content.substr(pos)); }
        catch (...) { return 0.0; }
    };
    
    profile.cpu_model = extract_string("cpu_model");
    profile.microarch = extract_string("microarch");
    profile.timestamp = extract_string("timestamp");
    profile.mean_absolute_error = extract_double("mean_absolute_error");
    
    profile.alu_correction   = extract_double("alu");
    profile.mul_correction   = extract_double("mul");
    profile.div_correction   = extract_double("div");
    profile.load_correction  = extract_double("load");
    profile.store_correction = extract_double("store");
    profile.branch_correction = extract_double("branch");
    profile.simd_correction  = extract_double("simd");
    profile.l1_latency_correction   = extract_double("l1_latency");
    profile.l2_latency_correction   = extract_double("l2_latency");
    profile.l3_latency_correction   = extract_double("l3_latency");
    profile.dram_latency_correction = extract_double("dram_latency");
    
    // Sanity: correction factors must be positive and reasonable
    auto clamp = [](double& v) {
        if (v <= 0.0 || v > 100.0 || std::isnan(v)) v = 1.0;
    };
    clamp(profile.alu_correction);
    clamp(profile.mul_correction);
    clamp(profile.div_correction);
    clamp(profile.load_correction);
    clamp(profile.store_correction);
    clamp(profile.branch_correction);
    clamp(profile.simd_correction);
    clamp(profile.l1_latency_correction);
    clamp(profile.l2_latency_correction);
    clamp(profile.l3_latency_correction);
    clamp(profile.dram_latency_correction);
    
    // Parse benchmark points from "benchmarks" array
    std::string bench_key = "\"benchmarks\":";
    auto bench_pos = content.find(bench_key);
    if (bench_pos != std::string::npos) {
        // Find each {op, predicted, measured, error_pct} object
        auto pos = bench_pos;
        while (true) {
            auto obj_start = content.find("{\"op\":", pos);
            if (obj_start == std::string::npos) break;
            auto obj_end = content.find("}", obj_start);
            if (obj_end == std::string::npos) break;
            
            std::string obj = content.substr(obj_start, obj_end - obj_start + 1);
            
            CalibrationPoint pt;
            // Extract op name
            auto op_pos = obj.find("\"op\": \"");
            if (op_pos != std::string::npos) {
                op_pos += 7;
                auto op_end = obj.find('"', op_pos);
                pt.operation = obj.substr(op_pos, op_end - op_pos);
            }
            // Extract numbers
            auto get_num = [&](const std::string& k) -> double {
                auto p = obj.find("\"" + k + "\": ");
                if (p == std::string::npos) return 0.0;
                p += k.size() + 4;
                try { return std::stod(obj.substr(p)); }
                catch (...) { return 0.0; }
            };
            pt.predicted_cycles = get_num("predicted");
            pt.measured_cycles = get_num("measured");
            pt.error_percent = get_num("error_pct");
            
            profile.points.push_back(pt);
            pos = obj_end + 1;
        }
    }
    
    return profile;
}

bool Calibrator::is_current(const CalibrationProfile& profile,
                             const HWProfile& current_hw,
                             uint32_t max_age_hours) {
    // Check CPU match
    if (profile.cpu_model != current_hw.model) return false;
    if (profile.microarch != current_hw.microarch) return false;
    
    // Check timestamp age
    if (!profile.timestamp.empty() && max_age_hours > 0) {
        // Parse ISO 8601 timestamp: "YYYY-MM-DDTHH:MM:SS"
        struct tm tm = {};
        if (strptime(profile.timestamp.c_str(), "%Y-%m-%dT%H:%M:%S", &tm)) {
            time_t profile_time = mktime(&tm);
            time_t now = std::time(nullptr);
            double age_hours = difftime(now, profile_time) / 3600.0;
            if (age_hours > max_age_hours)
                return false;
        }
    }
    
    return profile.is_valid();
}

} // namespace costforge
