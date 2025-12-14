#!/usr/bin/env python3
"""
One-Command Reproducibility Script

Regenerates all figures and tables from logged experiment results.

Usage:
    python analysis/run_all.py
    python analysis/run_all.py --output_dir figures/
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np

# Style configuration for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def find_latest_output(pattern: str) -> Path | None:
    """Find the most recent output directory matching pattern."""
    outputs_dir = project_root / "outputs"
    matches = sorted(outputs_dir.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_json_results(path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_ssl_comparison_bar(
    results: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    higher_is_better: bool = True,
):
    """Create bar chart comparing SSL vs scratch across label fractions."""
    fractions = sorted(set(r["label_fraction"] for r in results))

    scratch_vals = []
    ssl_vals = []

    for frac in fractions:
        scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

        scratch_vals.append(scratch[0][metric] if scratch else 0)
        ssl_vals.append(ssl[0][metric] if ssl else 0)

    x = np.arange(len(fractions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, scratch_vals, width, label="Scratch", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x + width / 2, ssl_vals, width, label="SSL Pretrained", color="#3498db", alpha=0.8)

    ax.set_xlabel("Label Fraction")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.legend()

    # Add improvement annotations
    for i, (s, ssl) in enumerate(zip(scratch_vals, ssl_vals)):
        if higher_is_better:
            improvement = (ssl - s) / s * 100 if s > 0 else 0
        else:
            improvement = (s - ssl) / s * 100 if s > 0 else 0
        ax.annotate(
            f"+{improvement:.1f}%",
            xy=(x[i] + width / 2, ssl_vals[i]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2c3e50",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_improvement_curve(
    results: list[dict],
    metric: str,
    title: str,
    output_path: Path,
    higher_is_better: bool = True,
):
    """Create line plot showing improvement across label fractions."""
    fractions = sorted(set(r["label_fraction"] for r in results))

    improvements = []
    for frac in fractions:
        scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

        if scratch and ssl:
            s_val = scratch[0][metric]
            ssl_val = ssl[0][metric]
            if higher_is_better:
                imp = (ssl_val - s_val) / s_val * 100 if s_val > 0 else 0
            else:
                imp = (s_val - ssl_val) / s_val * 100 if s_val > 0 else 0
            improvements.append(imp)
        else:
            improvements.append(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [f * 100 for f in fractions],
        improvements,
        "o-",
        linewidth=2,
        markersize=10,
        color="#e74c3c",
    )
    ax.fill_between([f * 100 for f in fractions], improvements, alpha=0.3, color="#e74c3c")

    ax.set_xlabel("Label Fraction (%)")
    ax.set_ylabel("Improvement (%)")
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, max(improvements) * 1.2)

    # Add value labels
    for f, imp in zip(fractions, improvements):
        ax.annotate(
            f"{imp:.1f}%",
            xy=(f * 100, imp),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_multi_task_comparison(results_dict: dict, output_path: Path):
    """Create grouped bar chart comparing SSL improvement at 10% labels.

    Shows relative improvement (%) for each task, handling MAE (lower=better)
    and F1 (higher=better) correctly.
    """
    tasks = []
    improvements = []

    for task, (results, metric, higher_is_better) in results_dict.items():
        frac_results = [r for r in results if r["label_fraction"] == 0.1]
        scratch = [r for r in frac_results if r["init_type"] == "scratch"]
        ssl = [r for r in frac_results if r["init_type"] == "ssl_pretrained"]

        if scratch and ssl:
            s_val = scratch[0][metric]
            ssl_val = ssl[0][metric]

            if s_val > 0:
                if higher_is_better:
                    # F1: higher is better, improvement = (ssl - scratch) / scratch
                    imp = (ssl_val - s_val) / s_val * 100
                else:
                    # MAE: lower is better, improvement = (scratch - ssl) / scratch
                    imp = (s_val - ssl_val) / s_val * 100

                tasks.append(task)
                improvements.append(imp)

    x = np.arange(len(tasks))
    colors = ["#3498db" if imp > 0 else "#e74c3c" for imp in improvements]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, improvements, color=colors, alpha=0.8)

    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(
            f"+{imp:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Task")
    ax.set_ylabel("SSL Improvement (%)")
    ax.set_title("SSL Transfer Benefits at 10% Labels (All Tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylim(0, max(improvements) * 1.2)

    # Add note about metric direction
    ax.text(
        0.02, 0.98,
        "Improvement = SSL benefit vs scratch\n(MAE: reduction, F1: increase)",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        style="italic",
        alpha=0.7,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_latex_table(results: list[dict], metric: str, caption: str, label: str) -> str:
    """Generate LaTeX table from results."""
    fractions = sorted(set(r["label_fraction"] for r in results))

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Label Fraction & Scratch & SSL & Improvement \\\\",
        "\\midrule",
    ]

    for frac in fractions:
        scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

        if scratch and ssl:
            s_val = scratch[0][metric]
            ssl_val = ssl[0][metric]
            # Assume lower is better for MAE, higher for F1
            if "mae" in metric.lower():
                imp = (s_val - ssl_val) / s_val * 100 if s_val > 0 else 0
            else:
                imp = (ssl_val - s_val) / s_val * 100 if s_val > 0 else 0

            lines.append(f"{int(frac * 100)}\\% & {s_val:.4f} & {ssl_val:.4f} & +{imp:.1f}\\% \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_markdown_table(results: list[dict], metric: str, title: str, higher_is_better: bool = True) -> str:
    """Generate Markdown table from results - SINGLE SOURCE OF TRUTH."""
    fractions = sorted(set(r["label_fraction"] for r in results))

    metric_name = "F1 Score" if "f1" in metric.lower() else "MAE"
    lines = [
        f"### {title}",
        "",
        f"| Label % | Scratch {metric_name} | SSL {metric_name} | Improvement |",
        "|---------|------------|--------|-------------|",
    ]

    for frac in fractions:
        scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

        if scratch and ssl:
            s_val = scratch[0][metric]
            ssl_val = ssl[0][metric]

            if higher_is_better:
                imp = (ssl_val - s_val) / s_val * 100 if s_val > 0 else 0
            else:
                imp = (s_val - ssl_val) / s_val * 100 if s_val > 0 else 0

            lines.append(f"| {int(frac * 100)}% | {s_val:.4f} | {ssl_val:.4f} | **+{imp:.1f}%** |")

    return "\n".join(lines)


def plot_grid_scalability(results_24: list[dict], results_118: list[dict], output_path: Path):
    """Create figure showing SSL is essential for large grids."""
    fractions = [0.1, 0.2, 0.5, 1.0]

    # Extract F1 scores
    ieee24_scratch = []
    ieee24_ssl = []
    ieee118_scratch = []
    ieee118_ssl = []

    for frac in fractions:
        # IEEE 24
        s24 = [r for r in results_24 if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl24 = [r for r in results_24 if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]
        ieee24_scratch.append(s24[0]["test_f1"] if s24 else 0)
        ieee24_ssl.append(ssl24[0]["test_f1"] if ssl24 else 0)

        # IEEE 118
        s118 = [r for r in results_118 if r["label_fraction"] == frac and r["init_type"] == "scratch"]
        ssl118 = [r for r in results_118 if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]
        ieee118_scratch.append(s118[0]["test_f1"] if s118 else 0)
        ieee118_ssl.append(ssl118[0]["test_f1"] if ssl118 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(fractions))
    width = 0.35

    # IEEE 24-bus
    ax1 = axes[0]
    ax1.bar(x - width/2, ieee24_scratch, width, label="Scratch", color="#2ecc71", alpha=0.8)
    ax1.bar(x + width/2, ieee24_ssl, width, label="SSL", color="#3498db", alpha=0.8)
    ax1.set_xlabel("Label Fraction")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("IEEE 24-bus (Small Grid)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(f*100)}%" for f in fractions])
    ax1.legend()
    ax1.set_ylim(0, 1.0)

    # IEEE 118-bus
    ax2 = axes[1]
    ax2.bar(x - width/2, ieee118_scratch, width, label="Scratch", color="#2ecc71", alpha=0.8)
    ax2.bar(x + width/2, ieee118_ssl, width, label="SSL", color="#3498db", alpha=0.8)
    ax2.set_xlabel("Label Fraction")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("IEEE 118-bus (Large Grid)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(f*100)}%" for f in fractions])
    ax2.legend()
    ax2.set_ylim(0, 1.0)

    # Add annotation showing scratch fails
    ax2.annotate(
        "Scratch FAILS\n(predicts all negatives)",
        xy=(0, 0.1),
        xytext=(1, 0.3),
        fontsize=10,
        color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
        ha="center",
    )

    plt.suptitle("SSL is Essential for Large Grids", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate all figures and tables")
    parser.add_argument("--output_dir", type=str, default="analysis/figures", help="Output directory for figures")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER FIGURES AND TABLES")
    print("=" * 60)

    # Find latest results for ieee24
    cascade_dir = find_latest_output("comparison_ieee24_*")
    pf_dir = find_latest_output("pf_comparison_ieee24_*")
    opf_dir = find_latest_output("opf_comparison_ieee24_*")

    # Find latest results for ieee118
    cascade_dir_118 = find_latest_output("comparison_ieee118_*")

    results_dict = {}

    # === CASCADE FIGURES ===
    if cascade_dir:
        print(f"\n[Cascade] Loading from: {cascade_dir}")
        cascade_results = load_json_results(cascade_dir / "results.json")

        plot_ssl_comparison_bar(
            cascade_results,
            "test_f1",
            "Cascade Prediction: SSL vs Scratch",
            "F1 Score",
            output_dir / "cascade_ssl_comparison.png",
            higher_is_better=True,
        )

        plot_improvement_curve(
            cascade_results,
            "test_f1",
            "Cascade Prediction: SSL Improvement vs Label Fraction",
            output_dir / "cascade_improvement_curve.png",
            higher_is_better=True,
        )

        results_dict["Cascade"] = (cascade_results, "test_f1", True)

        # LaTeX table
        latex = generate_latex_table(cascade_results, "test_f1", "Cascade Prediction SSL Transfer Results", "cascade_ssl")
        with open(output_dir / "cascade_table.tex", "w") as f:
            f.write(latex)
        print(f"  Saved: {output_dir / 'cascade_table.tex'}")
    else:
        print("\n[Cascade IEEE-24] No results found")

    # === CASCADE IEEE-118 FIGURES ===
    if cascade_dir_118:
        print(f"\n[Cascade IEEE-118] Loading from: {cascade_dir_118}")
        cascade_results_118 = load_json_results(cascade_dir_118 / "results.json")

        plot_ssl_comparison_bar(
            cascade_results_118,
            "test_f1",
            "Cascade Prediction (IEEE 118-bus): SSL vs Scratch",
            "F1 Score",
            output_dir / "cascade_118_ssl_comparison.png",
            higher_is_better=True,
        )

        plot_improvement_curve(
            cascade_results_118,
            "test_f1",
            "Cascade Prediction (IEEE 118-bus): SSL Improvement",
            output_dir / "cascade_118_improvement_curve.png",
            higher_is_better=True,
        )

        results_dict["Cascade (118)"] = (cascade_results_118, "test_f1", True)

        # LaTeX table for IEEE-118
        latex_118 = generate_latex_table(cascade_results_118, "test_f1", "Cascade Prediction SSL Transfer Results (IEEE 118-bus)", "cascade_118_ssl")
        with open(output_dir / "cascade_118_table.tex", "w") as f:
            f.write(latex_118)
        print(f"  Saved: {output_dir / 'cascade_118_table.tex'}")

        # Grid scalability comparison (if we have both grids)
        if cascade_dir:
            print("\n[Grid Comparison] Generating scalability figure")
            plot_grid_scalability(cascade_results, cascade_results_118, output_dir / "grid_scalability_comparison.png")
    else:
        print("\n[Cascade IEEE-118] No results found")

    # === PF FIGURES ===
    if pf_dir:
        print(f"\n[PF] Loading from: {pf_dir}")
        pf_results = load_json_results(pf_dir / "results.json")

        plot_ssl_comparison_bar(
            pf_results,
            "test_mae",
            "Power Flow: SSL vs Scratch",
            "MAE",
            output_dir / "pf_ssl_comparison.png",
            higher_is_better=False,
        )

        plot_improvement_curve(
            pf_results,
            "test_mae",
            "Power Flow: SSL Improvement vs Label Fraction",
            output_dir / "pf_improvement_curve.png",
            higher_is_better=False,
        )

        results_dict["Power Flow"] = (pf_results, "test_mae", False)

        # LaTeX table
        latex = generate_latex_table(pf_results, "test_mae", "Power Flow SSL Transfer Results", "pf_ssl")
        with open(output_dir / "pf_table.tex", "w") as f:
            f.write(latex)
        print(f"  Saved: {output_dir / 'pf_table.tex'}")
    else:
        print("\n[PF] No results found")

    # === OPF FIGURES ===
    if opf_dir:
        print(f"\n[OPF] Loading from: {opf_dir}")
        opf_results = load_json_results(opf_dir / "results.json")

        plot_ssl_comparison_bar(
            opf_results,
            "test_mae",
            "Optimal Power Flow: SSL vs Scratch",
            "MAE",
            output_dir / "opf_ssl_comparison.png",
            higher_is_better=False,
        )

        plot_improvement_curve(
            opf_results,
            "test_mae",
            "Optimal Power Flow: SSL Improvement vs Label Fraction",
            output_dir / "opf_improvement_curve.png",
            higher_is_better=False,
        )

        results_dict["OPF"] = (opf_results, "test_mae", False)

        # LaTeX table
        latex = generate_latex_table(opf_results, "test_mae", "OPF SSL Transfer Results", "opf_ssl")
        with open(output_dir / "opf_table.tex", "w") as f:
            f.write(latex)
        print(f"  Saved: {output_dir / 'opf_table.tex'}")
    else:
        print("\n[OPF] No results found")

    # === MULTI-TASK SUMMARY ===
    if len(results_dict) > 1:
        print("\n[Summary] Generating multi-task comparison")
        plot_multi_task_comparison(results_dict, output_dir / "multi_task_comparison.png")

    # === SUMMARY TABLE ===
    print("\n[Summary] Generating summary table")
    summary_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Cross-Task SSL Transfer Summary (10\\% Labels)}",
        "\\label{tab:summary}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Task & Metric & Improvement \\\\",
        "\\midrule",
    ]

    for task, (results, metric, higher_is_better) in results_dict.items():
        frac_results = [r for r in results if r["label_fraction"] == 0.1]
        scratch = [r for r in frac_results if r["init_type"] == "scratch"]
        ssl = [r for r in frac_results if r["init_type"] == "ssl_pretrained"]

        if scratch and ssl:
            s_val = scratch[0][metric]
            ssl_val = ssl[0][metric]
            if higher_is_better:
                imp = (ssl_val - s_val) / s_val * 100
            else:
                imp = (s_val - ssl_val) / s_val * 100

            metric_name = "F1" if "f1" in metric else "MAE"
            summary_lines.append(f"{task} & {metric_name} & +{imp:.1f}\\% \\\\")

    summary_lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    with open(output_dir / "summary_table.tex", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved: {output_dir / 'summary_table.tex'}")

    # === GENERATE MARKDOWN TABLES (SINGLE SOURCE OF TRUTH) ===
    print("\n[Markdown] Generating unified results tables")
    md_lines = [
        "# Auto-Generated Results Tables",
        "",
        "**Generated from:** `results.json` files in `outputs/`",
        "",
        "**DO NOT EDIT MANUALLY** - Regenerate with `python analysis/run_all.py`",
        "",
        "---",
        "",
    ]

    if cascade_dir:
        md_lines.append(generate_markdown_table(
            cascade_results, "test_f1", "Cascade Prediction (IEEE 24-bus)", higher_is_better=True
        ))
        md_lines.append("")

    if cascade_dir_118:
        md_lines.append(generate_markdown_table(
            cascade_results_118, "test_f1", "Cascade Prediction (IEEE 118-bus)", higher_is_better=True
        ))
        md_lines.append("")

    if pf_dir:
        md_lines.append(generate_markdown_table(
            pf_results, "test_mae", "Power Flow (IEEE 24-bus)", higher_is_better=False
        ))
        md_lines.append("")

    if opf_dir:
        md_lines.append(generate_markdown_table(
            opf_results, "test_mae", "Optimal Power Flow (IEEE 24-bus)", higher_is_better=False
        ))
        md_lines.append("")

    # Write markdown file
    md_path = output_dir / "auto_generated_tables.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total figures: {len(list(output_dir.glob('*.png')))}")
    print(f"Total tables: {len(list(output_dir.glob('*.tex')))}")
    print(f"Markdown tables: {md_path}")


if __name__ == "__main__":
    main()
