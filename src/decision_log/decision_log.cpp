#include "costforge/decision_log.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace costforge {

DecisionLog::DecisionLog() {}

void DecisionLog::begin_compilation(const std::string& source_file,
                                     const std::string& target_cpu) {
    report_.source_file = source_file;
    report_.target_cpu = target_cpu;
    
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    report_.timestamp = buf;
    
    report_.traces.clear();
    report_.by_category.clear();
    next_id_ = 0;
}

// ── Internal helpers ────────────────────────────────

std::string DecisionLog::format_cost(const CostResult& cost) const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "throughput=" << cost.throughput_cycles << "cy"
       << " latency=" << cost.latency_cycles << "cy"
       << " cache=" << (cost.cache_pressure * 100.0) << "%"
       << " regs=" << (cost.register_pressure * 100.0) << "%"
       << " branch=" << (cost.branch_risk * 100.0) << "%"
       << " → total=" << cost.total() << "cy";
    return ss.str();
}

double DecisionLog::compute_confidence(
        const std::vector<DecisionTrace::Option>& options,
        uint32_t chosen) const {
    if (options.size() < 2) return 1.0;
    
    double chosen_cost = options[chosen].cost.total();
    
    // Find the second-best option
    double second_best = 1e18;
    for (uint32_t i = 0; i < options.size(); i++) {
        if (i == chosen) continue;
        second_best = std::min(second_best, options[i].cost.total());
    }
    
    if (second_best <= 0.0) return 1.0;
    
    // Confidence = how much better chosen is vs runner-up
    // 0% gap → 0.5 confidence, 50%+ gap → ~1.0 confidence
    double gap = (second_best - chosen_cost) / second_best;
    return 0.5 + 0.5 * std::tanh(gap * 5.0);
}

std::string DecisionLog::generate_verdict(
        const std::vector<DecisionTrace::Option>& options,
        uint32_t chosen) const {
    if (options.size() < 2) return "only one option available";
    
    const auto& winner = options[chosen];
    uint32_t loser_idx = (chosen == 0) ? 1 : 0;
    const auto& loser = options[loser_idx];
    
    double savings = 0.0;
    if (loser.cost.total() > 0) {
        savings = (loser.cost.total() - winner.cost.total()) / loser.cost.total() * 100.0;
    }
    
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "chose '" << winner.name << "' over '" << loser.name << "'";
    
    if (savings > 0) {
        ss << " — saves " << savings << "%";
    }
    
    // Identify the dominant factor
    double d_throughput = std::abs(winner.cost.throughput_cycles - loser.cost.throughput_cycles);
    double d_cache = std::abs(winner.cost.cache_pressure - loser.cost.cache_pressure) * 50.0;
    double d_regs = std::abs(winner.cost.register_pressure - loser.cost.register_pressure) * 5.0;
    double d_branch = std::abs(winner.cost.branch_risk - loser.cost.branch_risk) * 17.0;
    
    double max_factor = std::max({d_throughput, d_cache, d_regs, d_branch});
    
    if (max_factor == d_throughput && d_throughput > 0.1) {
        ss << " (throughput dominant: " << d_throughput << " cycles difference)";
    } else if (max_factor == d_cache && d_cache > 0.1) {
        ss << " (cache pressure dominant: " 
           << std::abs(winner.cost.cache_pressure - loser.cost.cache_pressure) * 100.0 
           << "% difference)";
    } else if (max_factor == d_regs && d_regs > 0.1) {
        ss << " (register pressure dominant)";
    } else if (max_factor == d_branch && d_branch > 0.1) {
        ss << " (branch risk dominant)";
    }
    
    return ss.str();
}

// ── Recording decisions ─────────────────────────────

void DecisionLog::record(const std::string& function_name,
                          const std::string& location,
                          const std::string& decision_type,
                          const std::vector<DecisionTrace::Option>& options,
                          uint32_t chosen_index,
                          const std::vector<std::string>& factors,
                          std::optional<PatternKind> pattern) {
    DecisionTrace trace;
    trace.id = next_id_++;
    trace.timestamp = report_.timestamp;
    trace.function_name = function_name;
    trace.location = location;
    trace.decision_type = decision_type;
    trace.options = options;
    trace.chosen_index = chosen_index;
    trace.factors = factors;
    trace.detected_pattern = pattern;
    
    trace.confidence = compute_confidence(options, chosen_index);
    trace.verdict_summary = generate_verdict(options, chosen_index);
    
    report_.traces.push_back(std::move(trace));
    
    // Update category stats
    auto& cat = report_.by_category[decision_type];
    cat.considered++;
    if (chosen_index > 0) { // index 0 = "don't transform" by convention
        cat.applied++;
    }
    if (options.size() >= 2 && options[0].cost.total() > 0) {
        double savings = (options[0].cost.total() - options[chosen_index].cost.total()) 
                         / options[0].cost.total() * 100.0;
        cat.avg_savings = (cat.avg_savings * (cat.considered - 1) + savings) / cat.considered;
    }
}

void DecisionLog::record_decision(const std::string& function_name,
                                   const std::string& location,
                                   const Decision& decision,
                                   const std::vector<std::string>& factors) {
    DecisionTrace::Option opt_a;
    opt_a.name = "keep_original";
    opt_a.cost = decision.cost_a;
    opt_a.reasoning = format_cost(decision.cost_a);
    
    DecisionTrace::Option opt_b;
    opt_b.name = decision.name;
    opt_b.cost = decision.cost_b;
    opt_b.reasoning = format_cost(decision.cost_b);
    
    record(function_name, location, decision.name,
           {opt_a, opt_b},
           decision.choose_b ? 1 : 0,
           factors);
}

// ── Finalize ────────────────────────────────────────

CompilationReport DecisionLog::finalize() {
    report_.total_decisions = report_.traces.size();
    
    report_.transforms_applied = 0;
    for (const auto& [cat, stats] : report_.by_category) {
        report_.transforms_applied += stats.applied;
    }
    
    // Compute estimated total speedup
    double total_savings = 0.0;
    uint32_t count = 0;
    for (const auto& trace : report_.traces) {
        if (trace.chosen_index > 0 && trace.options.size() >= 2) {
            double base = trace.options[0].cost.total();
            double opt = trace.options[trace.chosen_index].cost.total();
            if (base > 0) {
                total_savings += (base - opt) / base * 100.0;
                count++;
            }
        }
    }
    report_.estimated_speedup_pct = (count > 0) ? total_savings / count : 0.0;
    
    return report_;
}

// ── JSON output ─────────────────────────────────────

void DecisionLog::write_json(const CompilationReport& report,
                              const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return;
    
    out << std::fixed << std::setprecision(2);
    
    out << "{\n";
    out << "  \"source_file\": \"" << report.source_file << "\",\n";
    out << "  \"target_cpu\": \"" << report.target_cpu << "\",\n";
    out << "  \"timestamp\": \"" << report.timestamp << "\",\n";
    out << "  \"summary\": {\n";
    out << "    \"total_decisions\": " << report.total_decisions << ",\n";
    out << "    \"transforms_applied\": " << report.transforms_applied << ",\n";
    out << "    \"estimated_speedup_pct\": " << report.estimated_speedup_pct << "\n";
    out << "  },\n";
    
    out << "  \"categories\": {\n";
    bool first_cat = true;
    for (const auto& [name, stats] : report.by_category) {
        if (!first_cat) out << ",\n";
        first_cat = false;
        out << "    \"" << name << "\": {"
            << "\"considered\": " << stats.considered << ", "
            << "\"applied\": " << stats.applied << ", "
            << "\"avg_savings_pct\": " << stats.avg_savings << "}";
    }
    out << "\n  },\n";
    
    out << "  \"decisions\": [\n";
    for (size_t i = 0; i < report.traces.size(); i++) {
        const auto& t = report.traces[i];
        out << "    {\n";
        out << "      \"id\": " << t.id << ",\n";
        out << "      \"function\": \"" << t.function_name << "\",\n";
        out << "      \"location\": \"" << t.location << "\",\n";
        out << "      \"type\": \"" << t.decision_type << "\",\n";
        out << "      \"confidence\": " << t.confidence << ",\n";
        out << "      \"verdict\": \"" << t.verdict_summary << "\",\n";
        
        if (t.detected_pattern) {
            out << "      \"pattern\": \"" << pattern_name(*t.detected_pattern) << "\",\n";
        }
        
        out << "      \"factors\": [";
        for (size_t f = 0; f < t.factors.size(); f++) {
            out << "\"" << t.factors[f] << "\"";
            if (f + 1 < t.factors.size()) out << ", ";
        }
        out << "],\n";
        
        out << "      \"options\": [\n";
        for (size_t o = 0; o < t.options.size(); o++) {
            const auto& opt = t.options[o];
            out << "        {\"name\": \"" << opt.name << "\", "
                << "\"total_cost\": " << opt.cost.total() << ", "
                << "\"chosen\": " << (o == t.chosen_index ? "true" : "false") << "}";
            if (o + 1 < t.options.size()) out << ",";
            out << "\n";
        }
        out << "      ]\n";
        
        out << "    }";
        if (i + 1 < report.traces.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

// ── Human-readable output ───────────────────────────

void DecisionLog::write_summary(const CompilationReport& report,
                                 std::ostream& out) {
    out << "╔══════════════════════════════════════════════╗\n";
    out << "║       CostForge Compilation Report          ║\n";
    out << "╚══════════════════════════════════════════════╝\n\n";
    
    out << "Source:    " << report.source_file << "\n";
    out << "CPU:       " << report.target_cpu << "\n";
    out << "Time:      " << report.timestamp << "\n\n";
    
    out << "── Summary ──────────────────────────────────\n";
    out << "  Decisions made:       " << report.total_decisions << "\n";
    out << "  Transforms applied:   " << report.transforms_applied << "\n";
    out << std::fixed << std::setprecision(1);
    out << "  Est. speedup:         " << report.estimated_speedup_pct << "%\n\n";
    
    out << "── By Category ──────────────────────────────\n";
    for (const auto& [name, stats] : report.by_category) {
        out << "  " << name << ": " 
            << stats.applied << "/" << stats.considered << " applied"
            << " (avg savings: " << stats.avg_savings << "%)\n";
    }
    out << "\n";
    
    // Show top decisions by impact
    auto sorted = report.traces;
    std::sort(sorted.begin(), sorted.end(),
              [](const DecisionTrace& a, const DecisionTrace& b) {
                  if (a.options.size() < 2 || b.options.size() < 2) return false;
                  double sa = a.options[0].cost.total() - 
                              a.options[a.chosen_index].cost.total();
                  double sb = b.options[0].cost.total() - 
                              b.options[b.chosen_index].cost.total();
                  return sa > sb;
              });
    
    out << "── Top Decisions (by impact) ────────────────\n";
    uint32_t shown = 0;
    for (const auto& t : sorted) {
        if (shown >= 10) break;
        out << "  [" << t.decision_type << "] " << t.function_name 
            << " @ " << t.location << "\n";
        out << "    " << t.verdict_summary << "\n";
        if (t.detected_pattern) {
            out << "    Pattern: " << pattern_name(*t.detected_pattern) << "\n";
        }
        out << "\n";
        shown++;
    }
}

std::string DecisionLog::explain(const DecisionTrace& trace) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    
    ss << "Decision #" << trace.id << ": " << trace.decision_type 
       << " in " << trace.function_name << "\n";
    ss << "Location: " << trace.location << "\n";
    ss << "Confidence: " << (trace.confidence * 100.0) << "%\n\n";
    
    ss << "Options evaluated:\n";
    for (size_t i = 0; i < trace.options.size(); i++) {
        const auto& opt = trace.options[i];
        ss << "  " << (i == trace.chosen_index ? "→ " : "  ") 
           << opt.name << ": " << opt.cost.total() << " cycles\n";
        ss << "    " << opt.reasoning << "\n";
    }
    
    ss << "\nVerdict: " << trace.verdict_summary << "\n";
    
    if (!trace.factors.empty()) {
        ss << "\nKey factors:\n";
        for (const auto& f : trace.factors) {
            ss << "  • " << f << "\n";
        }
    }
    
    if (trace.detected_pattern) {
        ss << "\nDetected pattern: " << pattern_name(*trace.detected_pattern) << "\n";
    }
    
    return ss.str();
}

std::string DecisionLog::diff_reports(const CompilationReport& before,
                                       const CompilationReport& after) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    
    ss << "── CostForge Report Diff ──\n\n";
    ss << "Before: " << before.target_cpu << " (" << before.timestamp << ")\n";
    ss << "After:  " << after.target_cpu << " (" << after.timestamp << ")\n\n";
    
    ss << "Decisions: " << before.total_decisions << " → " << after.total_decisions << "\n";
    ss << "Transforms: " << before.transforms_applied << " → " << after.transforms_applied << "\n";
    ss << "Est. speedup: " << before.estimated_speedup_pct 
       << "% → " << after.estimated_speedup_pct << "%\n\n";
    
    // Compare categories
    std::set<std::string> all_cats;
    for (const auto& [k, _] : before.by_category) all_cats.insert(k);
    for (const auto& [k, _] : after.by_category) all_cats.insert(k);
    
    for (const auto& cat : all_cats) {
        auto itb = before.by_category.find(cat);
        auto ita = after.by_category.find(cat);
        
        uint32_t b_applied = (itb != before.by_category.end()) ? itb->second.applied : 0;
        uint32_t a_applied = (ita != after.by_category.end()) ? ita->second.applied : 0;
        
        if (b_applied != a_applied) {
            ss << "  " << cat << ": " << b_applied << " → " << a_applied << "\n";
        }
    }
    
    return ss.str();
}

} // namespace costforge
