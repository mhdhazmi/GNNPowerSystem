#!/usr/bin/env python3
"""
Generate Paper Tables

Creates publication-ready tables in LaTeX and Markdown formats.

Usage:
    python analysis/generate_tables.py
    python analysis/generate_tables.py --format latex
    python analysis/generate_tables.py --format markdown
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_output(pattern: str) -> Path | None:
    """Find the most recent output directory matching pattern."""
    outputs_dir = project_root / "outputs"
    matches = sorted(outputs_dir.glob(pattern), reverse=True)
    return matches[0] if matches else None


def load_json_results(path: Path) -> list[dict]:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_main_results_table(results_dict: dict, fmt: str = "latex") -> str:
    """Generate the main results table (Table 1 in paper)."""
    if fmt == "latex":
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{SSL Transfer Results Across Tasks and Label Fractions. "
            "Improvement shows relative gain of SSL pretraining over scratch training.}",
            "\\label{tab:main_results}",
            "\\begin{tabular}{ll|cccc|cccc}",
            "\\toprule",
            "& & \\multicolumn{4}{c|}{\\textbf{Scratch}} & \\multicolumn{4}{c}{\\textbf{SSL Pretrained}} \\\\",
            "\\textbf{Task} & \\textbf{Metric} & 10\\% & 20\\% & 50\\% & 100\\% & 10\\% & 20\\% & 50\\% & 100\\% \\\\",
            "\\midrule",
        ]

        for task_name, (results, metric, higher_is_better) in results_dict.items():
            metric_label = "F1" if "f1" in metric else "MAE"
            fractions = [0.1, 0.2, 0.5, 1.0]

            scratch_vals = []
            ssl_vals = []

            for frac in fractions:
                scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
                ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

                scratch_vals.append(f"{scratch[0][metric]:.4f}" if scratch else "-")
                ssl_vals.append(f"{ssl[0][metric]:.4f}" if ssl else "-")

            line = f"{task_name} & {metric_label} & "
            line += " & ".join(scratch_vals)
            line += " & "
            line += " & ".join(ssl_vals)
            line += " \\\\"
            lines.append(line)

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table*}",
            ]
        )
        return "\n".join(lines)

    else:  # markdown
        lines = [
            "## Main Results: SSL Transfer Across Tasks",
            "",
            "| Task | Metric | Scratch 10% | Scratch 20% | Scratch 50% | Scratch 100% | SSL 10% | SSL 20% | SSL 50% | SSL 100% |",
            "|------|--------|-------------|-------------|-------------|--------------|---------|---------|---------|----------|",
        ]

        for task_name, (results, metric, higher_is_better) in results_dict.items():
            metric_label = "F1" if "f1" in metric else "MAE"
            fractions = [0.1, 0.2, 0.5, 1.0]

            vals = [task_name, metric_label]

            for init_type in ["scratch", "ssl_pretrained"]:
                for frac in fractions:
                    r = [x for x in results if x["label_fraction"] == frac and x["init_type"] == init_type]
                    vals.append(f"{r[0][metric]:.4f}" if r else "-")

            lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)


def generate_improvement_table(results_dict: dict, fmt: str = "latex") -> str:
    """Generate improvement summary table."""
    if fmt == "latex":
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{SSL Improvement Over Scratch Training}",
            "\\label{tab:improvement}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Task} & \\textbf{10\\%} & \\textbf{20\\%} & \\textbf{50\\%} & \\textbf{100\\%} \\\\",
            "\\midrule",
        ]

        for task_name, (results, metric, higher_is_better) in results_dict.items():
            fractions = [0.1, 0.2, 0.5, 1.0]
            improvements = []

            for frac in fractions:
                scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
                ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

                if scratch and ssl:
                    s_val = scratch[0][metric]
                    ssl_val = ssl[0][metric]
                    if higher_is_better:
                        imp = (ssl_val - s_val) / s_val * 100
                    else:
                        imp = (s_val - ssl_val) / s_val * 100
                    improvements.append(f"+{imp:.1f}\\%")
                else:
                    improvements.append("-")

            lines.append(f"{task_name} & " + " & ".join(improvements) + " \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines)

    else:  # markdown
        lines = [
            "## SSL Improvement Over Scratch Training",
            "",
            "| Task | 10% Labels | 20% Labels | 50% Labels | 100% Labels |",
            "|------|------------|------------|------------|-------------|",
        ]

        for task_name, (results, metric, higher_is_better) in results_dict.items():
            fractions = [0.1, 0.2, 0.5, 1.0]
            improvements = [task_name]

            for frac in fractions:
                scratch = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "scratch"]
                ssl = [r for r in results if r["label_fraction"] == frac and r["init_type"] == "ssl_pretrained"]

                if scratch and ssl:
                    s_val = scratch[0][metric]
                    ssl_val = ssl[0][metric]
                    if higher_is_better:
                        imp = (ssl_val - s_val) / s_val * 100
                    else:
                        imp = (s_val - ssl_val) / s_val * 100
                    improvements.append(f"**+{imp:.1f}%**")
                else:
                    improvements.append("-")

            lines.append("| " + " | ".join(improvements) + " |")

        return "\n".join(lines)


def generate_model_config_table(fmt: str = "latex") -> str:
    """Generate model configuration table."""
    config = {
        "Hidden Dimension": "128",
        "Number of Layers": "4",
        "Dropout": "0.1",
        "Optimizer": "AdamW",
        "Learning Rate": "1e-3",
        "Weight Decay": "1e-4",
        "Batch Size": "64",
        "Epochs": "50",
        "Seed": "42",
    }

    if fmt == "latex":
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Model and Training Configuration}",
            "\\label{tab:config}",
            "\\begin{tabular}{ll}",
            "\\toprule",
            "\\textbf{Parameter} & \\textbf{Value} \\\\",
            "\\midrule",
        ]

        for key, value in config.items():
            lines.append(f"{key} & {value} \\\\")

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines)

    else:  # markdown
        lines = [
            "## Model Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ]

        for key, value in config.items():
            lines.append(f"| {key} | {value} |")

        return "\n".join(lines)


def generate_dataset_table(fmt: str = "latex") -> str:
    """Generate dataset statistics table."""
    if fmt == "latex":
        return """\\begin{table}[h]
\\centering
\\caption{Dataset Statistics (IEEE 24-bus)}
\\label{tab:dataset}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Split} & \\textbf{Samples} & \\textbf{Nodes} & \\textbf{Edges} \\\\
\\midrule
Train & 16,125 & 24 & 68 \\\\
Validation & 2,016 & 24 & 68 \\\\
Test & 2,016 & 24 & 68 \\\\
\\midrule
Total & 20,157 & - & - \\\\
\\bottomrule
\\end{tabular}
\\end{table}"""

    else:  # markdown
        return """## Dataset Statistics (IEEE 24-bus)

| Split | Samples | Nodes | Edges |
|-------|---------|-------|-------|
| Train | 16,125 | 24 | 68 |
| Validation | 2,016 | 24 | 68 |
| Test | 2,016 | 24 | 68 |
| **Total** | **20,157** | - | - |"""


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--format", type=str, default="both", choices=["latex", "markdown", "both"])
    parser.add_argument("--output_dir", type=str, default="analysis/tables")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER TABLES")
    print("=" * 60)

    # Load results
    results_dict = {}

    cascade_dir = find_latest_output("comparison_ieee24_*")
    if cascade_dir:
        results_dict["Cascade"] = (load_json_results(cascade_dir / "results.json"), "test_f1", True)

    pf_dir = find_latest_output("pf_comparison_ieee24_*")
    if pf_dir:
        results_dict["Power Flow"] = (load_json_results(pf_dir / "results.json"), "test_mae", False)

    opf_dir = find_latest_output("opf_multiseed_ieee24_*")
    if not opf_dir:
        opf_dir = find_latest_output("opf_comparison_ieee24_*")
    if opf_dir:
        # Use "Line Flow" naming (not OPF) per reviewer feedback
        results_dict["Line Flow"] = (load_json_results(opf_dir / "results.json"), "test_mae", False)

    formats = ["latex", "markdown"] if args.format == "both" else [args.format]

    for fmt in formats:
        ext = "tex" if fmt == "latex" else "md"
        print(f"\nGenerating {fmt.upper()} tables...")

        # Main results
        table = generate_main_results_table(results_dict, fmt)
        path = output_dir / f"main_results.{ext}"
        with open(path, "w") as f:
            f.write(table)
        print(f"  Saved: {path}")

        # Improvement table
        table = generate_improvement_table(results_dict, fmt)
        path = output_dir / f"improvement.{ext}"
        with open(path, "w") as f:
            f.write(table)
        print(f"  Saved: {path}")

        # Config table
        table = generate_model_config_table(fmt)
        path = output_dir / f"config.{ext}"
        with open(path, "w") as f:
            f.write(table)
        print(f"  Saved: {path}")

        # Dataset table
        table = generate_dataset_table(fmt)
        path = output_dir / f"dataset.{ext}"
        with open(path, "w") as f:
            f.write(table)
        print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print("TABLE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
