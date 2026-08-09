#include "costforge/hwprobe.h"
#include <cpuid.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace costforge {

void HWProbe::cpuid(uint32_t leaf, uint32_t subleaf,
                     uint32_t& eax, uint32_t& ebx, 
                     uint32_t& ecx, uint32_t& edx) {
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
}

std::string HWProbe::detect_vendor() {
    uint32_t eax, ebx, ecx, edx;
    cpuid(0, 0, eax, ebx, ecx, edx);
    
    char vendor[13];
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    
    std::string v(vendor);
    if (v == "GenuineIntel") return "Intel";
    if (v == "AuthenticAMD") return "AMD";
    return v;
}

std::string HWProbe::detect_model_name() {
    char name[49];
    uint32_t eax, ebx, ecx, edx;
    
    for (uint32_t i = 0; i < 3; i++) {
        cpuid(0x80000002 + i, 0, eax, ebx, ecx, edx);
        memcpy(name + i * 16 + 0, &eax, 4);
        memcpy(name + i * 16 + 4, &ebx, 4);
        memcpy(name + i * 16 + 8, &ecx, 4);
        memcpy(name + i * 16 + 12, &edx, 4);
    }
    name[48] = '\0';
    
    // Trim leading spaces
    std::string result(name);
    size_t start = result.find_first_not_of(' ');
    return (start == std::string::npos) ? "" : result.substr(start);
}

std::string HWProbe::detect_microarch(const std::string& vendor,
                                        uint32_t family,
                                        uint32_t model) {
    if (vendor == "AMD") {
        if (family == 0x17) {
            if (model <= 0x0F) return "zen";
            if (model <= 0x2F) return "zen+";
            if (model <= 0x7F) return "zen2";
        }
        if (family == 0x19) {
            if (model <= 0x0F) return "zen3";
            if (model <= 0x4F) return "zen3+";
            if (model >= 0x60) return "zen4";
        }
        if (family == 0x1A) return "zen5";
    }
    
    if (vendor == "Intel") {
        // Simplified — real implementation would have full model table
        if (family == 6) {
            if (model == 0x8C || model == 0x8D) return "tigerlake";
            if (model == 0x97 || model == 0x9A) return "alderlake";
            if (model == 0xB7 || model == 0xBA) return "raptorlake";
            if (model == 0xBD) return "lunarlake";
        }
    }
    
    return "unknown";
}

SIMDCapabilities HWProbe::detect_simd() {
    SIMDCapabilities simd;
    uint32_t eax, ebx, ecx, edx;
    
    // CPUID leaf 1: SSE4.2
    cpuid(1, 0, eax, ebx, ecx, edx);
    simd.sse42 = (ecx >> 20) & 1;
    
    // AVX
    simd.avx = (ecx >> 28) & 1;
    
    // CPUID leaf 7: AVX2, AVX-512
    cpuid(7, 0, eax, ebx, ecx, edx);
    simd.avx2 = (ebx >> 5) & 1;
    simd.avx512 = (ebx >> 16) & 1; // AVX-512F
    
    return simd;
}

void HWProbe::detect_caches(HWProfile& profile) {
    uint32_t eax, ebx, ecx, edx;
    
    // Try AMD extended CPUID first (leaf 0x8000001D)
    cpuid(0x80000000, 0, eax, ebx, ecx, edx);
    bool use_amd_leaf = (eax >= 0x8000001D);
    
    uint32_t leaf = use_amd_leaf ? 0x8000001D : 4; // Intel uses leaf 4
    
    for (uint32_t index = 0; index < 10; index++) {
        cpuid(leaf, index, eax, ebx, ecx, edx);
        
        uint32_t type = eax & 0x1F;
        if (type == 0) break; // no more caches
        
        uint32_t level = (eax >> 5) & 0x7;
        uint32_t line_size = (ebx & 0xFFF) + 1;
        uint32_t partitions = ((ebx >> 12) & 0x3FF) + 1;
        uint32_t ways = ((ebx >> 22) & 0x3FF) + 1;
        uint32_t sets = ecx + 1;
        
        uint32_t size_kb = (ways * partitions * line_size * sets) / 1024;
        
        CacheLevel cache;
        cache.size_kb = size_kb;
        cache.line_size = line_size;
        cache.associativity = ways;
        
        if (level == 1 && type == 1) { // L1 data
            cache.latency_cycles = 4;
            profile.l1d = cache;
        } else if (level == 1 && type == 2) { // L1 instruction
            cache.latency_cycles = 4;
            profile.l1i = cache;
        } else if (level == 2) {
            cache.latency_cycles = 12;
            profile.l2 = cache;
        } else if (level == 3) {
            cache.latency_cycles = 35;
            profile.l3 = cache;
        }
    }
}

void HWProbe::estimate_execution_engine(HWProfile& profile) {
    // Estimate based on known microarchitectures
    // Real implementation would have a full database
    
    if (profile.microarch == "zen" || profile.microarch == "zen+") {
        profile.execution_ports = 10;
        profile.pipeline_depth = 19;
        profile.rob_size = 192;
        profile.load_buffer_size = 44;
        profile.store_buffer_size = 44;
        profile.branch_mispredict_penalty = 17;
    } else if (profile.microarch == "zen2") {
        profile.execution_ports = 10;
        profile.pipeline_depth = 19;
        profile.rob_size = 224;
        profile.load_buffer_size = 44;
        profile.store_buffer_size = 48;
        profile.branch_mispredict_penalty = 15;
    } else if (profile.microarch == "zen3") {
        profile.execution_ports = 10;
        profile.pipeline_depth = 19;
        profile.rob_size = 256;
        profile.load_buffer_size = 48;
        profile.store_buffer_size = 48;
        profile.branch_mispredict_penalty = 13;
    } else if (profile.microarch == "zen4" || profile.microarch == "zen5") {
        profile.execution_ports = 10;
        profile.pipeline_depth = 19;
        profile.rob_size = 320;
        profile.load_buffer_size = 56;
        profile.store_buffer_size = 56;
        profile.branch_mispredict_penalty = 11;
    } else if (profile.microarch == "alderlake" || profile.microarch == "raptorlake") {
        profile.execution_ports = 12;
        profile.pipeline_depth = 20;
        profile.rob_size = 512;
        profile.load_buffer_size = 128;
        profile.store_buffer_size = 72;
        profile.branch_mispredict_penalty = 14;
    } else {
        // Generic defaults
        profile.execution_ports = 8;
        profile.pipeline_depth = 16;
        profile.rob_size = 192;
        profile.load_buffer_size = 44;
        profile.store_buffer_size = 44;
        profile.branch_mispredict_penalty = 17;
    }
}

HWProfile HWProbe::scan() {
    HWProfile profile;
    
    // CPU identification
    profile.vendor = detect_vendor();
    profile.model = detect_model_name();
    
    // Family/model for microarch detection
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, eax, ebx, ecx, edx);
    
    uint32_t family = ((eax >> 8) & 0xF);
    uint32_t model_id = ((eax >> 4) & 0xF);
    
    if (family == 0xF) family += (eax >> 20) & 0xFF;
    if (family >= 6) model_id += ((eax >> 16) & 0xF) << 4;
    
    profile.microarch = detect_microarch(profile.vendor, family, model_id);
    
    // Topology
    profile.physical_cores = std::thread::hardware_concurrency() / 2;
    profile.logical_threads = std::thread::hardware_concurrency();
    
    // Frequency — read from /proc/cpuinfo on Linux
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                profile.base_freq_mhz = static_cast<uint32_t>(
                    std::stof(line.substr(colon + 1)));
            }
            break;
        }
    }
    profile.boost_freq_mhz = profile.base_freq_mhz; // TODO: read boost from sysfs
    
    // Cache hierarchy
    detect_caches(profile);
    
    // SIMD
    profile.simd = detect_simd();
    
    // Execution engine estimates
    estimate_execution_engine(profile);
    
    // Memory (rough estimates — real measurement would use STREAM benchmark)
    profile.memory_latency_ns = 80;      // typical DDR4
    profile.memory_bandwidth_gbs = 30;   // dual channel DDR4-3200
    
    return profile;
}

void HWProbe::dump(const HWProfile& profile) {
    std::cout << "=== CostForge Hardware Profile ===" << std::endl;
    std::cout << "CPU:           " << profile.model << std::endl;
    std::cout << "Vendor:        " << profile.vendor << std::endl;
    std::cout << "Microarch:     " << profile.microarch << std::endl;
    std::cout << "Cores/Threads: " << profile.physical_cores << "/" 
              << profile.logical_threads << std::endl;
    std::cout << "Frequency:     " << profile.base_freq_mhz << " MHz" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Cache Hierarchy:" << std::endl;
    std::cout << "  L1d: " << profile.l1d.size_kb << " KB, " 
              << profile.l1d.associativity << "-way, "
              << profile.l1d.latency_cycles << " cycles" << std::endl;
    std::cout << "  L1i: " << profile.l1i.size_kb << " KB, "
              << profile.l1i.associativity << "-way, "
              << profile.l1i.latency_cycles << " cycles" << std::endl;
    std::cout << "  L2:  " << profile.l2.size_kb << " KB, "
              << profile.l2.associativity << "-way, "
              << profile.l2.latency_cycles << " cycles" << std::endl;
    std::cout << "  L3:  " << profile.l3.size_kb << " KB, "
              << profile.l3.associativity << "-way, "
              << profile.l3.latency_cycles << " cycles" << std::endl;
    std::cout << std::endl;
    
    std::cout << "SIMD: ";
    if (profile.simd.avx512) std::cout << "AVX-512 ";
    if (profile.simd.avx2) std::cout << "AVX2 ";
    if (profile.simd.avx) std::cout << "AVX ";
    if (profile.simd.sse42) std::cout << "SSE4.2 ";
    std::cout << "(max width: " << profile.simd.max_width() << " bits)" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Execution Engine:" << std::endl;
    std::cout << "  Ports:              " << profile.execution_ports << std::endl;
    std::cout << "  Pipeline depth:     " << profile.pipeline_depth << std::endl;
    std::cout << "  ROB size:           " << profile.rob_size << std::endl;
    std::cout << "  Branch mispredict:  " << profile.branch_mispredict_penalty 
              << " cycles" << std::endl;
}

// TODO: implement save/load JSON with nlohmann/json or simdjson

void HWProbe::save(const HWProfile& profile, const std::string& json_path) {
    // Simplified JSON output — real implementation uses a JSON library
    std::ofstream out(json_path);
    out << "{\n";
    out << "  \"vendor\": \"" << profile.vendor << "\",\n";
    out << "  \"model\": \"" << profile.model << "\",\n";
    out << "  \"microarch\": \"" << profile.microarch << "\",\n";
    out << "  \"cores\": " << profile.physical_cores << ",\n";
    out << "  \"threads\": " << profile.logical_threads << ",\n";
    out << "  \"freq_mhz\": " << profile.base_freq_mhz << ",\n";
    out << "  \"cache\": {\n";
    out << "    \"l1d\": { \"size_kb\": " << profile.l1d.size_kb 
        << ", \"line_size\": " << profile.l1d.line_size
        << ", \"associativity\": " << profile.l1d.associativity
        << ", \"latency\": " << profile.l1d.latency_cycles << " },\n";
    out << "    \"l1i\": { \"size_kb\": " << profile.l1i.size_kb
        << ", \"line_size\": " << profile.l1i.line_size
        << ", \"associativity\": " << profile.l1i.associativity
        << ", \"latency\": " << profile.l1i.latency_cycles << " },\n";
    out << "    \"l2\": { \"size_kb\": " << profile.l2.size_kb
        << ", \"line_size\": " << profile.l2.line_size
        << ", \"associativity\": " << profile.l2.associativity
        << ", \"latency\": " << profile.l2.latency_cycles << " },\n";
    out << "    \"l3\": { \"size_kb\": " << profile.l3.size_kb
        << ", \"line_size\": " << profile.l3.line_size
        << ", \"associativity\": " << profile.l3.associativity
        << ", \"latency\": " << profile.l3.latency_cycles << " }\n";
    out << "  },\n";
    out << "  \"execution_ports\": " << profile.execution_ports << ",\n";
    out << "  \"pipeline_depth\": " << profile.pipeline_depth << ",\n";
    out << "  \"rob_size\": " << profile.rob_size << ",\n";
    out << "  \"branch_mispredict_penalty\": " << profile.branch_mispredict_penalty << ",\n";
    out << "  \"simd\": { \"sse42\": " << profile.simd.sse42
        << ", \"avx\": " << profile.simd.avx
        << ", \"avx2\": " << profile.simd.avx2
        << ", \"avx512\": " << profile.simd.avx512 << " }\n";
    out << "}\n";
}

HWProfile HWProbe::load(const std::string& json_path) {
    // TODO: implement JSON parsing
    // For now, just scan current hardware
    (void)json_path;
    return scan();
}

} // namespace costforge
