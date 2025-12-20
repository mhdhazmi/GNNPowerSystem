# Phase 3: Tables and Figures - Publication-Ready Artifacts
**AI Agent Execution Guide for IEEE Power Engineering Paper**

---

## Overview

**Phase:** Medium Priority (P2) - Paper Finalization
**Timeline:** Day 7-8
**Prerequisites:**
- Phase 1 (Critical Fixes) completed
- Phase 2 (Multi-seed experiments) completed
- All experimental results available in `results/` directory
**Output:** Camera-ready tables, figures, and LaTeX assets for IEEE submission

---

## Objective

Create a complete set of publication-ready tables and figures that:
1. Are **automatically generated** from canonical results
2. Are **internally consistent** (no number mismatches)
3. Meet **IEEE formatting standards**
4. Include all **required visualizations** for the narrative
5. Have **proper captions and labels**

---

## Part 1: Table Generation System

### T1: Main Results Table (IEEE Two-Column Format)

**Purpose:** Primary quantitative evidence table showing SSL benefits across all tasks

**File to generate:** `figures/tables/table_1_main_results.tex`

**Data source:** `results/multi_seed/summary_stats.json`

**Required elements:**
```latex
\begin{table*}[t]
\caption{SSL Transfer Benefits Across Tasks and Grid Scales}
\label{tab:main}
\centering
\begin{tabular}{llcccccc}
\toprule
\textbf{Task} & \textbf{Grid} & \textbf{Metric} & \textbf{Label \%} & \textbf{Scratch} & \textbf{SSL} & \textbf{Improvement} & \textbf{Seeds} \\
\midrule
\multirow{2}{*}{\textbf{Cascade}} & IEEE-24 & F1↑ & 10\% & 0.773±0.015 & \textbf{0.826±0.016} & +6.8\% & 5 \\
& & & 100\% & 0.955±0.007 & \textbf{0.958±0.005} & +0.3\% & 5 \\
\cmidrule{2-8}
& IEEE-118 & F1↑ & 10\% & 0.262±0.243 & \textbf{0.874±0.051} & ΔF1=+0.61 & 5 \\
& & & 100\% & 0.987±0.005 & \textbf{0.994±0.002} & +0.7\% & 5 \\
\midrule
\textbf{Power Flow} & IEEE-24 & MAE↓ & 10\% & 0.0149±0.0004 & \textbf{0.0106±0.0003} & +29.1\% & 5 \\
& & & 100\% & 0.0040±0.0002 & \textbf{0.0035±0.0001} & +13.0\% & 5 \\
\midrule
\textbf{Line Flow} & IEEE-24 & MAE↓ & 10\% & 0.0084±0.0003 & \textbf{0.0062±0.0002} & +26.4\% & 5 \\
& & & 100\% & 0.0022±0.00002 & \textbf{0.0021±0.0005} & +2.3\% & 5 \\
\bottomrule
\end{tabular}
\end{table*}
```

**Validation checks:**
- [ ] All numbers match `Simulation_Results.md` Table 1
- [ ] Standard deviations have correct precision (4 decimal places for MAE, 3 for F1)
- [ ] Improvement calculations verified: F1 uses (SSL-Scratch)/Scratch, MAE uses (Scratch-SSL)/Scratch
- [ ] IEEE-118 shows ΔF1 explicitly (not percentage)
- [ ] Bold formatting on better values

**Script command:**
```python
python analysis/generate_tables.py --table main_results --output figures/tables/table_1_main_results.tex --verify
```

---

### T2: Task Specifications Table

**Purpose:** Define inputs, outputs, metrics, and units for each task

**File to generate:** `figures/tables/table_task_specs.tex`

**Data source:** `Paper/Simulation_Results.md` - Task Definitions section

**Required elements:**
```latex
\begin{table}[t]
\caption{Task Specifications with Units}
\label{tab:tasks}
\centering
\begin{tabular}{llccc}
\toprule
\textbf{Task} & \textbf{Input} & \textbf{Output} & \textbf{Metric} & \textbf{Units} \\
\midrule
Cascade & Grid state $(P,S,V)$ & Binary label & F1 Score & [0,1] \\
Power Flow & Injections $(P,S)$ & Voltage $V_{\text{mag}}$ & MAE & p.u. \\
Line Flow & Bus states + params & Flows $(P_{ij},Q_{ij})$ & MAE & p.u. \\
\bottomrule
\end{tabular}
\end{table}
```

**Note:** Ensure consistency with feature audit - no target leakage in inputs

---

### T3: Dataset Statistics Table

**Purpose:** Document grid topology and data splits

**File to generate:** `figures/tables/table_dataset_stats.tex`

**Required elements:**
```latex
\begin{table}[t]
\caption{Dataset Statistics}
\label{tab:data}
\centering
\begin{tabular}{lccccc}
\toprule
\textbf{Grid} & \textbf{Buses} & \textbf{Lines} & \textbf{Train} & \textbf{Val} & \textbf{Test} \\
\midrule
IEEE-24 & 24 & 68 & 16,125 & 2,016 & 2,016 \\
IEEE-118 & 118 & 370 & 91,875 & 11,484 & 11,484 \\
\bottomrule
\end{tabular}
\end{table}
```

---

### T4: Per-Task Input/Output Specification Table

**Purpose:** Clarify feature usage to prevent "leakage perception"

**File to generate:** `figures/tables/table_io_specs.tex`

**Data source:** `Paper/ModelArchitecture.md` lines 96-109

**Required elements:**
```latex
\begin{table}[t]
\caption{Per-Task Input/Output Specification}
\label{tab:io}
\centering
\small
\begin{tabular}{lll}
\toprule
\textbf{Task} & \textbf{Edge Inputs} & \textbf{Targets} \\
\midrule
Power Flow & $[X, \text{rating}]$ & $V_{\text{mag}}$ (nodes) \\
Line Flow & $[X, \text{rating}]$ & $[P_{ij}, Q_{ij}]$ (edges) \\
Cascade & $[P_{ij}, Q_{ij}, X, \text{rating}]$ & Binary (graph) \\
\bottomrule
\end{tabular}
\end{table}
```

**Critical note:** For Line Flow, edge inputs = [X, rating] ONLY (no flows)

---

### T5: Baseline Comparison Tables

**T5.1: ML Baselines**

**File:** `figures/tables/table_ml_baselines.tex`

```latex
\begin{table}[t]
\caption{ML Baseline Comparison (Cascade, Power Flow)}
\label{tab:mlbaselines}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Features} & \textbf{Cascade F1} & \textbf{PF MAE} \\
\midrule
Random Forest & Aggregated & 0.68 & 0.0180 \\
XGBoost & Aggregated & 0.72 & 0.0165 \\
GNN (SSL, 10\%) & Graph-aware & \textbf{0.826} & \textbf{0.0106} \\
\bottomrule
\end{tabular}
\end{table}
```

**T5.2: Heuristic Baselines**

**File:** `figures/tables/table_heuristics.tex`

```latex
\begin{table}[t]
\caption{Heuristic Baselines (Cascade Prediction)}
\label{tab:heuristics}
\centering
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{Test F1} \\
\midrule
Always Negative & 0.00 \\
Max Loading Threshold ($\tau=0.8$) & 0.45 \\
Top-K Loading Check ($K=5$) & 0.52 \\
GNN (Scratch, 10\%) & 0.773 \\
GNN (SSL, 10\%) & \textbf{0.826} \\
\bottomrule
\end{tabular}
\end{table}
```

**Validation:** Ensure threshold values (τ, K) noted as "tuned on validation set only"

---

### T6: Ablation Study Table

**Purpose:** Show contribution of physics-guided architecture and SSL

**File:** `figures/tables/table_ablations.tex`

**Data source:** `results/ablations/ablation_summary.json`

```latex
\begin{table}[t]
\caption{Ablation Study: Architecture and Pretraining Components}
\label{tab:ablation}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Cascade F1} & \textbf{PF MAE} \\
\midrule
Vanilla GCN (Scratch) & 0.745 & 0.0158 \\
Vanilla GCN (SSL) & 0.798 & 0.0124 \\
PhysicsGuided (Scratch) & 0.773 & 0.0149 \\
PhysicsGuided (SSL) & \textbf{0.826} & \textbf{0.0106} \\
\bottomrule
\end{tabular}
\end{table}
```

**Script command:**
```bash
python scripts/run_ablations.py --output results/ablations/
python analysis/generate_tables.py --table ablations --output figures/tables/table_ablations.tex
```

---

### T7: Statistical Significance Table

**Purpose:** Report p-values and effect sizes for main claims

**File:** `figures/tables/table_statistics.tex`

**Data source:** `Paper/Statistical_Tests.md`

```latex
\begin{table}[t]
\caption{Statistical Significance of SSL Improvements}
\label{tab:stats}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Comparison} & \textbf{p-value} & \textbf{Cohen's d} & \textbf{Significant?} \\
\midrule
Cascade IEEE-24 & 0.0013 & 3.08 & Yes ($p<0.01$) \\
Cascade IEEE-118 & 0.0063 & 3.13 & Yes ($p<0.01$) \\
Power Flow & 0.000001 & 10.50 & Yes ($p<0.001$) \\
Line Flow & 0.000006 & 8.58 & Yes ($p<0.001$) \\
\bottomrule
\end{tabular}
\end{table}
```

---

### T8: Robustness Results Table

**Purpose:** Show OOD generalization under load scaling

**File:** `figures/tables/table_robustness.tex`

**Data source:** `results/robustness/ood_results.json`

```latex
\begin{table}[t]
\caption{Out-of-Distribution Robustness (Cascade F1 under Load Scaling)}
\label{tab:robust}
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Load Multiplier} & \textbf{Scratch} & \textbf{SSL} & \textbf{Advantage} \\
\midrule
1.0× (nominal) & 0.955 & 0.958 & +0.3\% \\
1.1× & 0.892 & 0.934 & +4.7\% \\
1.2× & 0.841 & 0.907 & +7.8\% \\
1.3× & 0.781 & 0.873 & +11.8\% \\
\bottomrule
\end{tabular}
\end{table}
```

**Note:** Add footnote: "Single-seed preliminary results (seed=42)"

---

### T9: Explainability Fidelity Table

**Purpose:** Validate edge attribution quality

**File:** `figures/tables/table_explainability.tex`

```latex
\begin{table}[t]
\caption{Edge Attribution Fidelity (AUC-ROC)}
\label{tab:explain}
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{AUC-ROC} & \textbf{Samples} \\
\midrule
Random Attribution & 0.50 & 489 \\
Gradient-based & 0.76 & 489 \\
PhysicsGuided + IG & \textbf{0.93} & 489 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Part 2: Figure Generation System

### F1: Cascade Prediction Figures (IEEE-24)

**F1.1: SSL vs Scratch Comparison**

**File:** `figures/cascade_ssl_comparison.pdf`

**Type:** Bar chart with error bars

**Script:**
```python
python analysis/plot_cascade_comparison.py \
  --grid ieee24 \
  --output figures/cascade_ssl_comparison.pdf \
  --format ieee \
  --dpi 300
```

**Requirements:**
- X-axis: Label fractions (10%, 20%, 50%, 100%)
- Y-axis: F1 Score (0 to 1.0)
- Two bar groups per x-tick: Scratch (blue), SSL (orange)
- Error bars: ± std (5 seeds)
- Legend: top-left corner
- Grid: major horizontal only
- Font: Times New Roman, 10pt

**Validation checks:**
- [ ] Values match Table 1 exactly
- [ ] Error bars visible and correct
- [ ] IEEE color scheme (avoid red/green)
- [ ] All text legible at publication size

**Caption:**
```latex
\caption{Cascade prediction on IEEE-24: SSL vs. scratch training across label fractions. Error bars show standard deviation (5 seeds). SSL provides largest improvement at 10\% labels (+6.8\%).}
\label{fig:cascade24a}
```

---

**F1.2: Improvement Curve**

**File:** `figures/cascade_improvement_curve.pdf`

**Type:** Line plot

**Script:**
```python
python analysis/plot_improvement_curves.py \
  --task cascade \
  --grid ieee24 \
  --output figures/cascade_improvement_curve.pdf
```

**Requirements:**
- X-axis: Label percentage (10 to 100, log scale optional)
- Y-axis: Improvement (%)
- Single line with markers
- Annotate 10% point prominently
- Diminishing returns narrative

**Caption:**
```latex
\caption{Relative improvement from SSL pretraining vs. label fraction. Improvement diminishes as labeled data increases, consistent with SSL's expected behavior.}
\label{fig:cascade24b}
```

---

### F2: Cascade Prediction Figures (IEEE-118)

**F2.1: SSL vs Scratch with Variance Highlighting**

**File:** `figures/cascade_118_ssl_comparison.pdf`

**Critical feature:** Show large variance at 10% for Scratch

**Script:**
```python
python analysis/plot_cascade_comparison.py \
  --grid ieee118 \
  --highlight_variance \
  --output figures/cascade_118_ssl_comparison.pdf
```

**Additional elements:**
- Annotation box at 10% scratch bar: "High variance (±0.243)"
- Annotation box at 10% SSL bar: "Stable (±0.051)"
- Consider box plots instead of bars for 10% to show distribution

**Caption:**
```latex
\caption{Scalability on IEEE-118: SSL stabilizes training at low labels. At 10\%, scratch training exhibits high variance (±0.243) while SSL remains stable (±0.051).}
\label{fig:cascade118a}
```

---

**F2.2: Absolute ΔF1 Plot**

**File:** `figures/cascade_118_delta_f1.pdf`

**Purpose:** Avoid misleading percentage when baseline is very low

**Script:**
```python
python analysis/plot_delta_metrics.py \
  --task cascade \
  --grid ieee118 \
  --metric_type delta_f1 \
  --output figures/cascade_118_delta_f1.pdf
```

**Y-axis:** ΔF1 (absolute difference: SSL F1 - Scratch F1)

**Caption:**
```latex
\caption{Absolute F1 gain (ΔF1) from SSL on IEEE-118. At 10\% labels, SSL provides ΔF1=+0.61, enabling reliable cascade prediction where scratch training fails.}
\label{fig:cascade118b}
```

---

### F3: Power Flow Figures

**F3.1: MAE Comparison**

**File:** `figures/pf_ssl_comparison.pdf`

**Type:** Bar chart (lower is better)

**Script:**
```python
python analysis/plot_pf_comparison.py \
  --output figures/pf_ssl_comparison.pdf
```

**Important:** MAE is "lower is better" - consider inverted axis or clear annotation

**Caption:**
```latex
\caption{Power flow prediction (voltage magnitude) on IEEE-24. SSL reduces MAE by 29.1\% at 10\% labels. Units in per-unit (p.u.) on 100 MVA base.}
\label{fig:pf_a}
```

---

**F3.2: Improvement Curve**

**File:** `figures/pf_improvement_curve.pdf`

**Caption:**
```latex
\caption{Power flow MAE improvement vs. label fraction. SSL provides substantial gains in low-label regime, converging with scratch at 100\%.}
\label{fig:pf_b}
```

---

### F4: Line Flow Figures

**F4.1: MAE Comparison**

**File:** `figures/lineflow_ssl_comparison.pdf`

**Script:**
```python
python analysis/plot_lineflow_comparison.py \
  --output figures/lineflow_ssl_comparison.pdf
```

**Caption:**
```latex
\caption{Line flow prediction (branch power flows) on IEEE-24. SSL achieves 26.4\% MAE reduction at 10\% labels.}
\label{fig:lineflow_a}
```

---

**F4.2: Improvement Curve**

**File:** `figures/lineflow_improvement_curve.pdf`

**Caption:**
```latex
\caption{Line flow improvement trend follows similar pattern to power flow: largest gains at low labels with diminishing returns.}
\label{fig:lineflow_b}
```

---

### F5: Multi-Task Synthesis Figure

**File:** `figures/multi_task_comparison.pdf`

**Type:** Grouped bar chart showing all tasks at 10% labels

**Script:**
```python
python analysis/plot_multitask_synthesis.py \
  --label_fraction 10 \
  --output figures/multi_task_comparison.pdf
```

**Requirements:**
- X-axis: Four task groups (Cascade-24, Cascade-118, PF, Line Flow)
- Y-axis: Normalized performance or dual axes
- Challenge: F1 (higher better) vs MAE (lower better)

**Solution options:**
1. Normalize both to "% improvement"
2. Two subplots side-by-side
3. Inverted MAE axis

**Caption:**
```latex
\caption{Cross-task summary at 10\% labels. SSL improves performance across cascade prediction, power flow, and line flow tasks, with largest gains where scratch training struggles (IEEE-118).}
\label{fig:synthesis}
```

---

### F6: Grid Scalability Comparison

**File:** `figures/grid_scalability_comparison.pdf`

**Type:** Side-by-side comparison of IEEE-24 vs IEEE-118

**Script:**
```python
python analysis/plot_scalability.py \
  --output figures/grid_scalability_comparison.pdf
```

**Caption:**
```latex
\caption{Grid scalability: SSL benefits are more pronounced on larger IEEE-118 network, particularly in stabilizing low-label training.}
\label{fig:scalability}
```

---

### F7: Method Overview Schematic

**File:** `figures/method_overview.pdf`

**Type:** Architecture diagram (create manually or with TikZ)

**Required elements:**
1. Input graph representation
2. PhysicsGuidedEncoder block
3. SSL pretraining loop (masked reconstruction)
4. Three task heads: Cascade, Power Flow, Line Flow
5. Transfer learning arrows

**Tools:**
- TikZ (LaTeX)
- draw.io → export PDF
- PowerPoint → export PDF (if necessary)

**Caption:**
```latex
\caption{Method overview: (a) SSL pretraining on unlabeled training data using masked reconstruction; (b) transfer to task-specific heads for cascade (graph-level), power flow (node-level), and line flow (edge-level) prediction.}
\label{fig:pipeline}
```

---

### F8: Explainability Visualization

**File:** `figures/explainability_example.pdf`

**Type:** Graph visualization with edge highlighting

**Script:**
```python
python analysis/visualize_explanations.py \
  --sample_id 42 \
  --output figures/explainability_example.pdf
```

**Requirements:**
- Grid topology (IEEE-24 or IEEE-118 subset)
- Edges colored by importance score
- Ground truth critical edges outlined
- Color bar for importance values

**Caption:**
```latex
\caption{Edge attribution example: Integrated Gradients highlights critical transmission lines for cascade propagation. Bold edges indicate ground-truth explanation (from PowerGraph).}
\label{fig:explain}
```

---

### F9: Robustness Curves

**File:** `figures/robustness_curves.pdf`

**Type:** Line plot with multiple perturbation types

**Script:**
```python
python analysis/plot_robustness.py \
  --output figures/robustness_curves.pdf
```

**Requirements:**
- X-axis: Perturbation magnitude (e.g., load multiplier 1.0 to 1.4)
- Y-axis: Cascade F1
- Two lines: Scratch, SSL
- Shaded regions for confidence if multi-seed available

**Caption:**
```latex
\caption{Out-of-distribution robustness under load scaling. SSL maintains higher performance under stress conditions. Single-seed preliminary results (seed=42).}
\label{fig:robust}
```

---

### F10: Ablation Results

**File:** `figures/ablation_comparison.pdf`

**Type:** Grouped bar chart

**Script:**
```python
python analysis/plot_ablations.py \
  --output figures/ablation_comparison.pdf
```

**Requirements:**
- Four configurations: Vanilla/Scratch, Vanilla/SSL, Physics/Scratch, Physics/SSL
- Show both Cascade F1 and PF MAE (consider subplot)

**Caption:**
```latex
\caption{Ablation study: both physics-guided architecture and SSL pretraining contribute to performance. Largest gains from combining both components.}
\label{fig:ablation}
```

---

## Part 3: Automated Figure/Table Generation Script

### Master Script: `analysis/run_all.py`

**Purpose:** One command to regenerate ALL figures and tables

**Usage:**
```bash
python analysis/run_all.py --mode publication --verify
```

**Script structure:**
```python
#!/usr/bin/env python3
"""
Master script for publication-ready artifact generation.
Ensures all tables and figures are consistent with canonical results.
"""

import json
import subprocess
from pathlib import Path

def verify_data_sources():
    """Check that all required result files exist."""
    required_files = [
        'results/multi_seed/summary_stats.json',
        'results/ablations/ablation_summary.json',
        'results/robustness/ood_results.json',
        'results/explainability/attribution_scores.json'
    ]
    for file in required_files:
        assert Path(file).exists(), f"Missing: {file}"
    print("✓ All data sources found")

def generate_tables():
    """Generate all LaTeX tables."""
    tables = [
        ('main_results', 'table_1_main_results.tex'),
        ('task_specs', 'table_task_specs.tex'),
        ('dataset_stats', 'table_dataset_stats.tex'),
        ('io_specs', 'table_io_specs.tex'),
        ('ml_baselines', 'table_ml_baselines.tex'),
        ('heuristics', 'table_heuristics.tex'),
        ('ablations', 'table_ablations.tex'),
        ('statistics', 'table_statistics.tex'),
        ('robustness', 'table_robustness.tex'),
        ('explainability', 'table_explainability.tex')
    ]

    for table_type, filename in tables:
        print(f"Generating {filename}...")
        subprocess.run([
            'python', 'analysis/generate_tables.py',
            '--table', table_type,
            '--output', f'figures/tables/{filename}',
            '--verify'
        ], check=True)
    print("✓ All tables generated")

def generate_figures():
    """Generate all figures."""
    figures = [
        ('cascade_ssl_comparison.pdf', ['analysis/plot_cascade_comparison.py', '--grid', 'ieee24']),
        ('cascade_improvement_curve.pdf', ['analysis/plot_improvement_curves.py', '--task', 'cascade', '--grid', 'ieee24']),
        ('cascade_118_ssl_comparison.pdf', ['analysis/plot_cascade_comparison.py', '--grid', 'ieee118', '--highlight_variance']),
        ('cascade_118_delta_f1.pdf', ['analysis/plot_delta_metrics.py', '--task', 'cascade', '--grid', 'ieee118']),
        ('pf_ssl_comparison.pdf', ['analysis/plot_pf_comparison.py']),
        ('pf_improvement_curve.pdf', ['analysis/plot_improvement_curves.py', '--task', 'pf']),
        ('lineflow_ssl_comparison.pdf', ['analysis/plot_lineflow_comparison.py']),
        ('lineflow_improvement_curve.pdf', ['analysis/plot_improvement_curves.py', '--task', 'lineflow']),
        ('multi_task_comparison.pdf', ['analysis/plot_multitask_synthesis.py', '--label_fraction', '10']),
        ('grid_scalability_comparison.pdf', ['analysis/plot_scalability.py']),
        ('explainability_example.pdf', ['analysis/visualize_explanations.py', '--sample_id', '42']),
        ('robustness_curves.pdf', ['analysis/plot_robustness.py']),
        ('ablation_comparison.pdf', ['analysis/plot_ablations.py'])
    ]

    for filename, cmd in figures:
        print(f"Generating {filename}...")
        subprocess.run(cmd + ['--output', f'figures/{filename}', '--format', 'ieee', '--dpi', '300'], check=True)
    print("✓ All figures generated")

def verify_consistency():
    """Cross-check that numbers in tables match figures."""
    print("Running consistency checks...")
    subprocess.run(['python', 'analysis/verify_consistency.py'], check=True)
    print("✓ All consistency checks passed")

def generate_supplementary():
    """Generate supplementary materials."""
    print("Generating supplementary PDF...")
    subprocess.run(['python', 'analysis/generate_supplementary.py'], check=True)
    print("✓ Supplementary materials generated")

if __name__ == '__main__':
    print("=== Publication Artifact Generation ===")
    verify_data_sources()
    generate_tables()
    generate_figures()
    verify_consistency()
    generate_supplementary()
    print("\n✓✓✓ All publication artifacts ready! ✓✓✓")
    print("\nOutput locations:")
    print("  Tables: figures/tables/*.tex")
    print("  Figures: figures/*.pdf")
    print("  Supplementary: figures/supplementary.pdf")
```

---

## Part 4: Consistency Verification System

### Script: `analysis/verify_consistency.py`

**Purpose:** Detect number mismatches between tables, figures, and text

**Checks performed:**
1. Table 1 numbers match `Simulation_Results.md` Table 1
2. Figure bar heights match corresponding table values
3. Improvement calculations correct (F1 vs MAE formulas)
4. No legacy numbers from old 3-seed runs
5. IEEE-118 imbalance rate consistent everywhere
6. All "±" standard deviations have same precision

**Usage:**
```bash
python analysis/verify_consistency.py --strict --report output/consistency_report.txt
```

**Output example:**
```
✓ Table 1 vs Simulation_Results.md: ALL MATCH
✓ Figure cascade_ssl_comparison.pdf: values verified
✓ Improvement calculations: correct formulas
✓ No legacy 3-seed values found
✗ WARNING: IEEE-118 imbalance rate inconsistent
  - Found "~5%" in line 245 of Results.md
  - Found "~20%" in line 789 of Progress_Report.md
  - RECOMMEND: standardize to canonical value

OVERALL: 1 warning, 0 errors
```

---

## Part 5: IEEE Formatting Compliance

### Requirements Checklist

**Tables:**
- [ ] Use `\toprule`, `\midrule`, `\bottomrule` (booktabs package)
- [ ] No vertical lines
- [ ] Caption above table
- [ ] Label after caption
- [ ] Units in column headers or caption
- [ ] Bold for best values
- [ ] Footnotes with `\tnote` if needed
- [ ] Two-column tables use `table*` environment

**Figures:**
- [ ] Vector format (PDF) for scalable graphics
- [ ] Minimum 300 DPI for rasterized elements
- [ ] Font sizes readable at publication size (≥8pt)
- [ ] Color-blind safe palette
- [ ] IEEE color scheme (avoid pure red/green)
- [ ] Caption below figure
- [ ] Label after caption
- [ ] Subfigures labeled (a), (b), etc.
- [ ] All axes labeled with units
- [ ] Legend present and readable
- [ ] Two-column figures use `figure*` environment

---

## Part 6: LaTeX Integration

### Main Paper: `paper/main.tex`

**Table placement strategy:**
```latex
% Section III: Problem Setup
\input{figures/tables/table_task_specs.tex}  % Table I
\input{figures/tables/table_io_specs.tex}     % Table II

% Section V: Experimental Setup
\input{figures/tables/table_dataset_stats.tex}  % Table III
\input{figures/tables/table_hparams.tex}        % Table IV

% Section VI: Results
\input{figures/tables/table_1_main_results.tex}  % Table V (two-column)

% Reference in text
As shown in Table~\ref{tab:main}, SSL provides consistent improvements...
```

**Figure placement strategy:**
```latex
% Section IV: Method
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/method_overview.pdf}
\caption{Method overview...}
\label{fig:pipeline}
\end{figure*}

% Section VI-B: Cascade Results
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/cascade_ssl_comparison.pdf}
\caption{Cascade prediction...}
\label{fig:cascade24}
\end{figure}
```

---

## Part 7: Supplementary Materials

### Document: `figures/supplementary.pdf`

**Contents:**
1. Extended results tables (20%, 50% label fractions)
2. Per-seed detailed breakdowns
3. Additional ablation configurations
4. Hyperparameter sensitivity analysis
5. Physics metric validation plots
6. Additional explainability examples
7. Code availability statement
8. Reproducibility checklist

**Generation:**
```bash
python analysis/generate_supplementary.py --output figures/supplementary.pdf
```

---

## Part 8: Final Verification Checklist

Before declaring "Phase 3 Complete":

**Data Integrity:**
- [ ] All numbers sourced from `results/multi_seed/summary_stats.json`
- [ ] No manual edits to generated tables
- [ ] All figures regenerated from scripts
- [ ] Consistency verification passes with 0 errors

**Completeness:**
- [ ] All 10 tables generated
- [ ] All 13 figures generated
- [ ] Supplementary PDF created
- [ ] LaTeX compiles without errors

**IEEE Compliance:**
- [ ] Booktabs formatting for all tables
- [ ] Vector PDFs for all figures
- [ ] Captions and labels correct
- [ ] Font sizes readable
- [ ] Color-blind safe palettes

**Narrative Alignment:**
- [ ] Table 1 is the quantitative anchor
- [ ] IEEE-118 stabilization story clear
- [ ] Low-label regime emphasis throughout
- [ ] Physics guidance + SSL synergy shown

**Publication Readiness:**
- [ ] No "TODO" comments in LaTeX
- [ ] No placeholder figures
- [ ] All cross-references resolve
- [ ] Bibliography formatted
- [ ] Supplementary referenced in main text

---

## Part 9: Execution Timeline

**Day 7 Morning:**
- Run multi-seed experiments if not complete (Phase 2 dependency)
- Verify all result JSON files present

**Day 7 Afternoon:**
- Execute `python analysis/run_all.py`
- Review generated tables for formatting
- Fix any table generation bugs

**Day 7 Evening:**
- Review all figures
- Adjust color schemes / fonts if needed
- Run consistency verification

**Day 8 Morning:**
- Generate supplementary materials
- Integrate tables/figures into LaTeX
- Compile paper and check float placement

**Day 8 Afternoon:**
- Final consistency sweep
- Address any IEEE formatting issues
- Generate final PDF for review

---

## Part 10: Common Issues and Solutions

### Issue 1: "Multi-task plot mixes MAE and F1"

**Problem:** Cannot directly compare metrics with opposite directions

**Solutions:**
1. Use "% improvement" for all (already normalized)
2. Two subplots: "Classification Tasks" (F1) and "Regression Tasks" (MAE)
3. Inverted axis for MAE: plot `1 - (MAE/max_MAE)`

**Recommended:** Option 2 (two subplots)

---

### Issue 2: "IEEE-118 percentage improvement is misleading"

**Problem:** +234% when baseline is 0.262 is technically correct but visually dominates

**Solution:**
- Use ΔF1 notation explicitly
- Add text annotation: "ΔF1=+0.61"
- Consider separate subplot for IEEE-118

---

### Issue 3: "Error bars too small to see"

**Problem:** Standard deviations are small (good!) but invisible in plot

**Solution:**
- Increase error bar cap size: `capsize=5`
- Use `linewidth=1.5` for error bars
- Add annotation: "Error bars: ±1 std (5 seeds)"

---

### Issue 4: "Tables don't fit in column width"

**Solution:**
- Use `\small` or `\footnotesize` font
- Abbreviate headers: "Improvement" → "Improv."
- Switch to landscape table environment if necessary
- Split into two tables if content permits

---

### Issue 5: "Figures are blurry in PDF"

**Problem:** Low DPI or rasterization

**Solution:**
```python
# In matplotlib scripts
plt.savefig('output.pdf', format='pdf', dpi=300, bbox_inches='tight')

# For raster elements within PDF
from matplotlib import rcParams
rcParams['savefig.dpi'] = 300
rcParams['figure.dpi'] = 300
```

---

## Part 11: AI Agent Execution Commands

**Full Phase 3 execution (from project root):**

```bash
# Step 1: Verify Phase 2 completion
python analysis/check_phase2_complete.py

# Step 2: Generate all tables
python analysis/run_all.py --tables-only --verify

# Step 3: Generate all figures
python analysis/run_all.py --figures-only --format ieee --dpi 300

# Step 4: Consistency check
python analysis/verify_consistency.py --strict

# Step 5: Generate supplementary
python analysis/generate_supplementary.py

# Step 6: Full integration test
python analysis/run_all.py --mode publication --verify

# Step 7: LaTeX compilation test
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Expected output:**
```
✓ All data sources verified
✓ 10/10 tables generated
✓ 13/13 figures generated
✓ Consistency check: 0 errors, 0 warnings
✓ Supplementary PDF created
✓ LaTeX compiles successfully
✓✓✓ Phase 3 COMPLETE - Paper ready for submission ✓✓✓
```

---

## Summary

Phase 3 delivers:
- **10 publication-ready LaTeX tables**
- **13 IEEE-compliant figures**
- **1 supplementary PDF**
- **Automated generation pipeline**
- **Consistency verification system**
- **Zero manual editing required**

All artifacts are reproducible, traceable to canonical results, and meet IEEE Power & Energy Society formatting standards. The paper is now ready for final author review and submission to PES General Meeting or IEEE Transactions.
