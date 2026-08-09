#pragma once

#include "costforge/types.h"
#include <string>

namespace costforge {

/// Hardware Probe — scans the current CPU and builds a HWProfile.
///
/// Uses CPUID instruction to detect:
/// - CPU vendor, model, microarchitecture
/// - Cache hierarchy (sizes, latencies, associativity)
/// - SIMD capabilities (SSE4.2, AVX, AVX2, AVX-512)
/// - Execution engine characteristics
///
/// Can also load/save profiles to JSON for cross-compilation scenarios
/// (compile on machine A with profile of machine B).

class HWProbe {
public:
    /// Scan the current CPU and return its hardware profile
    static HWProfile scan();
    
    /// Load a hardware profile from a JSON file
    static HWProfile load(const std::string& json_path);
    
    /// Save a hardware profile to a JSON file
    static void save(const HWProfile& profile, const std::string& json_path);
    
    /// Print a human-readable summary of the profile
    static void dump(const HWProfile& profile);

private:
    /// Read CPUID leaf and return EAX, EBX, ECX, EDX
    static void cpuid(uint32_t leaf, uint32_t subleaf,
                      uint32_t& eax, uint32_t& ebx, 
                      uint32_t& ecx, uint32_t& edx);
    
    /// Detect CPU vendor string ("GenuineIntel", "AuthenticAMD")
    static std::string detect_vendor();
    
    /// Detect CPU model name ("AMD Ryzen 5 3550H ...")
    static std::string detect_model_name();
    
    /// Detect microarchitecture from family/model/stepping
    static std::string detect_microarch(const std::string& vendor, 
                                         uint32_t family, 
                                         uint32_t model);
    
    /// Detect cache topology using CPUID leaf 4 (Intel) or leaf 0x8000001D (AMD)
    static void detect_caches(HWProfile& profile);
    
    /// Detect SIMD capabilities from CPUID feature flags
    static SIMDCapabilities detect_simd();
    
    /// Estimate execution engine parameters from microarchitecture
    static void estimate_execution_engine(HWProfile& profile);
    
    /// Load instruction cost table for a specific microarchitecture
    /// Sources: Agner Fog's instruction tables, uops.info
    static CostTable load_cost_table(const std::string& microarch);
};

} // namespace costforge
