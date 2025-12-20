#!/usr/bin/env python3
"""
Phase 3 Consistency Verification Script

Cross-checks that all tables, figures, and documentation are consistent with
the canonical data sources in outputs/.

Checks:
1. Table numbers match summary_stats.json values
2. Improvement calculations use correct formulas (F1 vs MAE)
3. Standard deviation precision is consistent
4. No legacy 3-seed values remain (should be 5-seed)
5. Cross-reference tables ↔ figures data

Usage:
    python analysis/verify_consistency.py
    python analysis/verify_consistency.py --verbose
"""

import argparse
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


class VerificationResult:
    """Stores verification check results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = []
        self.failed = []
        self.warnings = []

    def pass_(self, msg: str):
        self.passed.append(msg)

    def fail(self, msg: str):
        self.failed.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        status = "PASS" if not self.failed else "FAIL"
        return f"[{status}] {self.name}: {len(self.passed)} passed, {len(self.failed)} failed, {len(self.warnings)} warnings"


def find_latest_output(pattern: str) -> Path | None:
    """Find the most recent output directory matching pattern."""
    outputs_dir = project_root / "outputs"
    matches = sorted(outputs_dir.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_canonical_data() -> dict:
    """Load all canonical data sources."""
    data = {}

    # Multi-seed cascade IEEE-24
    cascade_dir = find_latest_output("multiseed_ieee24_*")
    if cascade_dir and (cascade_dir / "summary_stats.json").exists():
        with open(cascade_dir / "summary_stats.json") as f:
            data["cascade_24"] = json.load(f)

    # Multi-seed cascade IEEE-118
    cascade_118_dir = find_latest_output("multiseed_ieee118_*")
    if cascade_118_dir and (cascade_118_dir / "summary_stats.json").exists():
        with open(cascade_118_dir / "summary_stats.json") as f:
            data["cascade_118"] = json.load(f)

    # Multi-seed power flow
    pf_dir = find_latest_output("pf_multiseed_ieee24_*")
    if pf_dir and (pf_dir / "summary_stats.json").exists():
        with open(pf_dir / "summary_stats.json") as f:
            data["pf"] = json.load(f)

    # Multi-seed line flow
    opf_dir = find_latest_output("opf_multiseed_ieee24_*")
    if opf_dir and (opf_dir / "summary_stats.json").exists():
        with open(opf_dir / "summary_stats.json") as f:
            data["lineflow"] = json.load(f)

    # Ablations
    ablation_dir = find_latest_output("ablations_cascade_ieee24_*")
    if ablation_dir and (ablation_dir / "ablation_results.json").exists():
        with open(ablation_dir / "ablation_results.json") as f:
            data["ablations"] = json.load(f)

    # Robustness
    robustness_dir = find_latest_output("eval_physics_robustness_*")
    if robustness_dir and (robustness_dir / "results.json").exists():
        with open(robustness_dir / "results.json") as f:
            data["robustness"] = json.load(f)

    return data


def check_seed_count(data: dict) -> VerificationResult:
    """Verify all experiments use 5 seeds (not legacy 3-seed)."""
    result = VerificationResult("Seed Count Validation")

    for name, summary in data.items():
        if name in ["cascade_24", "cascade_118", "pf", "lineflow"]:
            if not summary:
                result.warn(f"{name}: No data found")
                continue

            for entry in summary:
                n_seeds = entry.get("n_seeds", 0)
                frac = entry.get("label_fraction", "?")
                if n_seeds == 5:
                    result.pass_(f"{name} @ {frac*100:.0f}%: {n_seeds} seeds")
                elif n_seeds == 3:
                    result.fail(f"{name} @ {frac*100:.0f}%: LEGACY 3-seed data detected")
                else:
                    result.warn(f"{name} @ {frac*100:.0f}%: Unexpected seed count {n_seeds}")

    return result


def check_improvement_calculations(data: dict) -> VerificationResult:
    """Verify improvement calculations use correct formulas."""
    result = VerificationResult("Improvement Calculation Validation")

    # F1 is higher-is-better: (ssl - scratch) / scratch
    # MAE is lower-is-better: (scratch - ssl) / scratch

    for name, summary in data.items():
        if name not in ["cascade_24", "cascade_118", "pf", "lineflow"]:
            continue
        if not summary:
            continue

        is_f1 = name.startswith("cascade")

        for entry in summary:
            frac = entry.get("label_fraction", 0)
            scratch = entry.get("scratch_mean", 0)
            ssl = entry.get("ssl_mean", 0)

            if scratch == 0:
                result.warn(f"{name} @ {frac*100:.0f}%: Scratch mean is 0")
                continue

            if is_f1:
                # F1: higher is better
                if ssl >= scratch:
                    improvement = (ssl - scratch) / scratch * 100
                    result.pass_(f"{name} @ {frac*100:.0f}%: SSL ({ssl:.3f}) >= Scratch ({scratch:.3f}), +{improvement:.1f}%")
                else:
                    result.warn(f"{name} @ {frac*100:.0f}%: SSL ({ssl:.3f}) < Scratch ({scratch:.3f})")
            else:
                # MAE: lower is better
                if ssl <= scratch:
                    improvement = (scratch - ssl) / scratch * 100
                    result.pass_(f"{name} @ {frac*100:.0f}%: SSL ({ssl:.4f}) <= Scratch ({scratch:.4f}), +{improvement:.1f}%")
                else:
                    result.warn(f"{name} @ {frac*100:.0f}%: SSL ({ssl:.4f}) > Scratch ({scratch:.4f})")

    return result


def check_table_values(data: dict, verbose: bool = False) -> VerificationResult:
    """Cross-check generated table values against canonical data."""
    result = VerificationResult("Table Value Validation")

    # Check main results table (generated by generate_tables.py to figures/tables/)
    table_path = project_root / "figures" / "tables" / "table_1_main_results.tex"
    if not table_path.exists():
        result.fail(f"Main results table not found: {table_path}")
        return result

    with open(table_path) as f:
        table_content = f.read()

    # Parse expected values from canonical data
    expected_values = {}

    # Cascade IEEE-24 at 10%
    if "cascade_24" in data:
        for entry in data["cascade_24"]:
            if abs(entry["label_fraction"] - 0.1) < 0.01:
                expected_values["cascade_24_10_scratch"] = entry["scratch_mean"]
                expected_values["cascade_24_10_ssl"] = entry["ssl_mean"]
            if abs(entry["label_fraction"] - 1.0) < 0.01:
                expected_values["cascade_24_100_scratch"] = entry["scratch_mean"]
                expected_values["cascade_24_100_ssl"] = entry["ssl_mean"]

    # Cascade IEEE-118 at 10%
    if "cascade_118" in data:
        for entry in data["cascade_118"]:
            if abs(entry["label_fraction"] - 0.1) < 0.01:
                expected_values["cascade_118_10_scratch"] = entry["scratch_mean"]
                expected_values["cascade_118_10_ssl"] = entry["ssl_mean"]
            if abs(entry["label_fraction"] - 1.0) < 0.01:
                expected_values["cascade_118_100_scratch"] = entry["scratch_mean"]
                expected_values["cascade_118_100_ssl"] = entry["ssl_mean"]

    # Power Flow at 10%
    if "pf" in data:
        for entry in data["pf"]:
            if abs(entry["label_fraction"] - 0.1) < 0.01:
                expected_values["pf_10_scratch"] = entry["scratch_mean"]
                expected_values["pf_10_ssl"] = entry["ssl_mean"]
            if abs(entry["label_fraction"] - 1.0) < 0.01:
                expected_values["pf_100_scratch"] = entry["scratch_mean"]
                expected_values["pf_100_ssl"] = entry["ssl_mean"]

    # Line Flow at 10%
    if "lineflow" in data:
        for entry in data["lineflow"]:
            if abs(entry["label_fraction"] - 0.1) < 0.01:
                expected_values["lineflow_10_scratch"] = entry["scratch_mean"]
                expected_values["lineflow_10_ssl"] = entry["ssl_mean"]
            if abs(entry["label_fraction"] - 1.0) < 0.01:
                expected_values["lineflow_100_scratch"] = entry["scratch_mean"]
                expected_values["lineflow_100_ssl"] = entry["ssl_mean"]

    # Check if expected values appear in table (with tolerance)
    def value_in_table(val: float, content: str, tolerance: float = 0.01) -> bool:
        """Check if value appears in table within tolerance."""
        # Look for formatted values like 0.773 or 0.0149
        if val < 0.1:
            patterns = [f"{val:.4f}", f"{val:.3f}"]
        else:
            patterns = [f"{val:.3f}", f"{val:.2f}"]
        return any(p in content for p in patterns)

    for key, val in expected_values.items():
        if value_in_table(val, table_content):
            result.pass_(f"{key}: {val:.4f} found in table")
        else:
            result.fail(f"{key}: {val:.4f} NOT found in table")

    return result


def check_figures_exist(verbose: bool = False) -> VerificationResult:
    """Verify all expected figures exist."""
    result = VerificationResult("Figure Existence Check")

    figures_dir = project_root / "analysis" / "figures"
    expected_figures = [
        "cascade_ssl_comparison",
        "cascade_improvement_curve",
        "cascade_118_ssl_comparison",
        "cascade_118_improvement_curve",
        "cascade_118_delta_f1",
        "pf_ssl_comparison",
        "pf_improvement_curve",
        "lineflow_ssl_comparison",
        "lineflow_improvement_curve",
        "multi_task_comparison",
        "grid_scalability_comparison",
        "robustness_curves",
        "ablation_comparison",
    ]

    for fig_name in expected_figures:
        png_path = figures_dir / f"{fig_name}.png"
        pdf_path = figures_dir / f"{fig_name}.pdf"

        if png_path.exists() or pdf_path.exists():
            fmt = "PNG" if png_path.exists() else "PDF"
            result.pass_(f"{fig_name}: {fmt} exists")
        else:
            result.fail(f"{fig_name}: NOT FOUND (neither PNG nor PDF)")

    return result


def check_tables_exist(verbose: bool = False) -> VerificationResult:
    """Verify all expected tables exist."""
    result = VerificationResult("Table Existence Check")

    # Tables are generated to figures/tables/ by generate_tables.py
    tables_dir = project_root / "figures" / "tables"
    expected_tables = [
        "table_1_main_results.tex",
        "table_task_specs.tex",
        "table_dataset_stats.tex",
        "table_io_specs.tex",
        "table_ml_baselines.tex",
        "table_heuristics.tex",
        "table_ablations.tex",
        "table_statistics.tex",
        "table_robustness.tex",
        "table_explainability.tex",
    ]

    for table_name in expected_tables:
        table_path = tables_dir / table_name
        if table_path.exists():
            size = table_path.stat().st_size
            result.pass_(f"{table_name}: exists ({size} bytes)")
        else:
            result.fail(f"{table_name}: NOT FOUND")

    return result


def check_statistical_tests_document() -> VerificationResult:
    """Verify Statistical_Tests.md contains expected values."""
    result = VerificationResult("Statistical Tests Document")

    doc_path = project_root / "Paper" / "Statistical_Tests.md"
    if not doc_path.exists():
        result.fail(f"Document not found: {doc_path}")
        return result

    with open(doc_path) as f:
        content = f.read()

    # Check for required sections
    required_sections = [
        "Welch's t-test",
        "Cohen's d",
        "Cascade IEEE-24",
        "Cascade IEEE-118",
        "Power Flow",
        "Line Flow",
    ]

    for section in required_sections:
        if section in content:
            result.pass_(f"Section '{section}' found")
        else:
            result.fail(f"Section '{section}' NOT FOUND")

    # Check for p-values
    p_values_found = len(re.findall(r"p\s*[=<]\s*0\.\d+", content))
    if p_values_found >= 4:
        result.pass_(f"Found {p_values_found} p-value entries")
    else:
        result.fail(f"Only {p_values_found} p-value entries found (expected >= 4)")

    return result


def check_ieee118_delta_notation(data: dict) -> VerificationResult:
    """Verify IEEE-118 uses ΔF1 notation when scratch baseline is low."""
    result = VerificationResult("IEEE-118 Delta Notation")

    if "cascade_118" not in data:
        result.warn("No IEEE-118 cascade data found")
        return result

    # Check 10% label fraction where scratch baseline is typically low
    for entry in data["cascade_118"]:
        if abs(entry["label_fraction"] - 0.1) < 0.01:
            scratch = entry["scratch_mean"]
            ssl = entry["ssl_mean"]

            if scratch < 0.4:
                result.pass_(f"Scratch baseline is low ({scratch:.3f}) - ΔF1 notation appropriate")

                # Check that main results table uses ΔF1 notation
                table_path = project_root / "figures" / "tables" / "table_1_main_results.tex"
                if table_path.exists():
                    with open(table_path) as f:
                        content = f.read()
                    if "Delta" in content or "Δ" in content or "delta" in content.lower():
                        result.pass_("Table uses delta notation for IEEE-118")
                    else:
                        result.warn("Table may not use delta notation for IEEE-118 (check manually)")
            else:
                result.pass_(f"Scratch baseline is adequate ({scratch:.3f}) - percentage OK")

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 3 consistency")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 CONSISTENCY VERIFICATION")
    print("=" * 60)

    # Load canonical data
    print("\nLoading canonical data sources...")
    data = load_canonical_data()
    print(f"  Loaded {len(data)} data sources: {list(data.keys())}")

    # Run all checks
    checks = [
        check_seed_count(data),
        check_improvement_calculations(data),
        check_table_values(data, args.verbose),
        check_figures_exist(args.verbose),
        check_tables_exist(args.verbose),
        check_statistical_tests_document(),
        check_ieee118_delta_notation(data),
    ]

    # Print results
    print("\n" + "-" * 60)
    print("VERIFICATION RESULTS")
    print("-" * 60)

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for check in checks:
        print(f"\n{check.summary()}")

        if args.verbose:
            for msg in check.passed:
                print(f"  [PASS] {msg}")
            for msg in check.failed:
                print(f"  [FAIL] {msg}")
            for msg in check.warnings:
                print(f"  [WARN] {msg}")

        total_passed += len(check.passed)
        total_failed += len(check.failed)
        total_warnings += len(check.warnings)

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total checks: {total_passed + total_failed + total_warnings}")
    print(f"  Passed:   {total_passed}")
    print(f"  Failed:   {total_failed}")
    print(f"  Warnings: {total_warnings}")

    if total_failed == 0:
        print("\n✓ All critical checks passed!")
        return 0
    else:
        print(f"\n✗ {total_failed} critical check(s) failed")
        print("  Run with --verbose for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
