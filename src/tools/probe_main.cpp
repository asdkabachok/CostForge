#include "costforge/hwprobe.h"
#include "costforge/cost_model.h"
#include "costforge/decision_engine.h"
#include <iostream>
#include <string>

using namespace costforge;

void demo_decisions(const HWProfile& hw) {
    CostModel model(hw);
    DecisionEngine engine(model);
    
    std::cout << "\n=== Demo: Optimization Decisions for Your CPU ===" << std::endl;
    
    // Should we inline a 20-instruction function called 100 times?
    auto inline_d = engine.should_inline(20, 0.5, 1024, 100);
    std::cout << "\n[INLINE] " << inline_d.description << std::endl;
    std::cout << "  Call cost:   " << inline_d.cost_a.total() << " cycles" << std::endl;
    std::cout << "  Inline cost: " << inline_d.cost_b.total() << " cycles" << std::endl;
    std::cout << "  Decision:    " << (inline_d.choose_b ? "INLINE" : "KEEP CALL") << std::endl;
    std::cout << "  Savings:     " << inline_d.savings_percent << "%" << std::endl;
    
    // Should we unroll a simple loop?
    std::vector<std::string> loop_body = {"load", "mul", "add", "store"};
    auto unroll_d = engine.should_unroll(loop_body, 1000, 8000);
    std::cout << "\n[UNROLL] loop with " << loop_body.size() << " ops, 1000 iterations" << std::endl;
    std::cout << "  Should unroll: " << (unroll_d.should_unroll ? "YES" : "NO") << std::endl;
    if (unroll_d.should_unroll) {
        std::cout << "  Optimal factor: " << unroll_d.optimal_factor << "x" << std::endl;
        std::cout << "  Savings: " << unroll_d.savings_percent << "%" << std::endl;
    }
    
    // Should we vectorize?
    auto vec_d = engine.should_vectorize(loop_body, 1000, 8000, false);
    std::cout << "\n[VECTORIZE] same loop" << std::endl;
    std::cout << "  Should vectorize: " << (vec_d.should_vectorize ? "YES" : "NO") << std::endl;
    if (vec_d.should_vectorize) {
        std::cout << "  Optimal width: " << vec_d.optimal_width << " bits" << std::endl;
        std::cout << "  Savings: " << vec_d.savings_percent << "%" << std::endl;
    }
    
    // Should we convert a 50/50 branch to cmov?
    auto branch_d = engine.should_branchless(0.5, 3, 3);
    std::cout << "\n[BRANCHLESS] 50/50 branch" << std::endl;
    std::cout << "  Branch cost: " << branch_d.cost_a.total() << " cycles" << std::endl;
    std::cout << "  Cmov cost:   " << branch_d.cost_b.total() << " cycles" << std::endl;
    std::cout << "  Decision:    " << (branch_d.choose_b ? "USE CMOV" : "KEEP BRANCH") << std::endl;
    
    // Summary
    auto stats = engine.stats();
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total decisions:    " << stats.total_decisions << std::endl;
    std::cout << "Inlines accepted:   " << stats.inlines_accepted << std::endl;
    std::cout << "Loops unrolled:     " << stats.loops_unrolled << std::endl;
    std::cout << "Loops vectorized:   " << stats.loops_vectorized << std::endl;
    std::cout << "Branches converted: " << stats.branches_converted << std::endl;
    std::cout << "Avg savings:        " << stats.total_estimated_savings_percent << "%" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "CostForge v0.1.0 — Hardware-Aware Optimization" << std::endl;
    std::cout << std::endl;
    
    // Scan hardware
    HWProfile hw = HWProbe::scan();
    HWProbe::dump(hw);
    
    // Save profile
    if (argc > 1 && std::string(argv[1]) == "--save") {
        std::string path = (argc > 2) ? argv[2] : "profiles/my_cpu.json";
        HWProbe::save(hw, path);
        std::cout << "\nProfile saved to: " << path << std::endl;
    }
    
    // Run demo decisions
    if (argc <= 1 || std::string(argv[1]) == "--demo") {
        demo_decisions(hw);
    }
    
    return 0;
}
