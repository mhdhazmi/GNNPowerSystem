# V. Experimental Setup

---

## V-A. Datasets and Splits

### P1: Dataset Description

We evaluate on the **PowerGraph benchmark** using two IEEE test cases of different scales:

**Table D: Dataset Statistics**

| Grid | Buses | Lines | Total Samples | Train (80%) | Val (10%) | Test (10%) |
|------|-------|-------|---------------|-------------|-----------|------------|
| IEEE 24-bus | 24 | 68 | 20,157 | 16,125 | 2,016 | 2,016 |
| IEEE 118-bus | 118 | 370 | 114,843 | 91,875 | 11,484 | 11,484 |

**Per-unit System**: All electrical quantities normalized to system base (100 MVA).

### P2: Split Protocol and Leakage Prevention

- **Split ratio**: 80/10/10 train/validation/test
- **Stratification**: Cascade prediction uses stratified sampling to maintain class distribution
- **SSL pretraining**: Uses training partition only (no validation/test exposure)
- **Metric reporting**: All results computed on held-out test set
- **Validation use**: Early stopping only; never used for final metric computation

**Table E: SSL Pretraining Data Split**

| Phase | Data Source | Samples | Labels Required |
|-------|-------------|---------|-----------------|
| **SSL Pretraining** | Train set only | 16,125 (IEEE-24) / 91,875 (IEEE-118) | None (self-supervised) |
| **Fine-tuning** | Train set subset | Variable (10%-100%) | Yes |
| **Validation** | Val set | 2,016 / 11,484 | Yes (for early stopping) |
| **Final Evaluation** | Held-out Test set | 2,016 / 11,484 | Yes (never seen during training) |

**Critical Disclosure**: SSL pretraining uses only the training partition with self-supervised objectives (masked reconstruction). Validation and test sets are never exposed during pretraining.

---

## V-B. Low-Label Protocol and Evaluation

### P1: Label Fraction Experiment

We evaluate transfer benefit across label fractions {10%, 20%, 50%, 100%} of the training set:
- **10% labels**: ~1,600 (IEEE-24) or ~9,200 (IEEE-118) training samples
- **100% labels**: Full training set

This simulates varying data availability scenarios common in power system applications.

### P2: Multi-Seed Validation

Results are reported as **mean ± standard deviation** across independent random seeds:

**Table F: Seed Count and Evaluation Protocol**

| Task | Grid | Seeds | Seed Values | Rationale |
|------|------|-------|-------------|-----------|
| Cascade Prediction | IEEE-24 | 5 | 42, 123, 456, 789, 1337 | Standard multi-seed validation |
| Cascade Prediction | IEEE-118 | 5 | 42, 123, 456, 789, 1337 | High variance regime |
| Power Flow | IEEE-24 | 5 | 42, 123, 456, 789, 1337 | Standard multi-seed validation |
| Line Flow | IEEE-24 | 5 | 42, 123, 456, 789, 1337 | Standard multi-seed validation |

**Note**: IEEE-118 cascade at 10% labels exhibits high variance (σ=0.243 for Scratch) due to limited positive samples (~490 cascade scenarios out of ~9,200 samples at 10%), making multi-seed validation essential for reliable conclusions.

### P3: Improvement Computation

- **For F1** (higher = better): $\text{Improvement} = \frac{\text{SSL} - \text{Scratch}}{\text{Scratch}} \times 100\%$
- **For MAE** (lower = better): $\text{Improvement} = \frac{\text{Scratch} - \text{SSL}}{\text{Scratch}} \times 100\%$

Both yield positive values when SSL outperforms Scratch.

---

## V-C. Model Configuration and Training

### P1: Architecture Settings

**Table G: Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Architecture | PhysicsGuidedEncoder |
| Hidden Dimension | 128 |
| Number of Layers | 4 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Epochs | 50 |
| Batch Size | 64 |
| Early Stopping | Patience = 10 epochs |

### P2: Optimization and Stopping

- **Optimizer**: AdamW with learning rate 1e-3 and default weight decay
- **Epochs**: Maximum 50 with early stopping based on validation metric
- **Batch size**: 64 graphs per batch
- **Early stopping**: Training stops if validation metric doesn't improve for 10 epochs
- **Checkpoint selection**: Best validation checkpoint used for test evaluation

---

## V-D. Baselines

### P1: Why Baselines Are Needed

Baselines establish that (1) GNNs provide value over feature-based ML, and (2) SSL provides value over scratch-trained GNNs. We compare against both machine learning and heuristic approaches.

### P2: Machine Learning Baselines

**Table H: ML Baseline Comparison**

| Task | Method | Metric | 100% Labels | 10% Labels |
|------|--------|--------|-------------|------------|
| **Cascade** | XGBoost (edge features) | F1 | 0.72 | 0.58 |
| | Random Forest | F1 | 0.68 | 0.54 |
| | **GNN (Scratch)** | F1 | **0.955** | **0.773** |
| | **GNN (SSL)** | F1 | **0.958** | **0.826** |
| **Power Flow** | Linear Regression | MAE | 0.089 | 0.095 |
| | XGBoost | MAE | 0.024 | 0.031 |
| | **GNN (Scratch)** | MAE | **0.0040** | **0.0149** |
| | **GNN (SSL)** | MAE | **0.0035** | **0.0106** |

**Feature Representation:**
- **Cascade**: Each graph represented by aggregated edge statistics (mean/max/std of line loading, power flow, reactance)
- **Power Flow**: Flattened bus injection features

**Hyperparameter Tuning**: XGBoost/RF hyperparameters tuned via 5-fold cross-validation on training set. Best hyperparameters selected by validation metric.

### P3: Heuristic Baselines (Cascade)

**Table I: Heuristic Baselines for Cascade Prediction**

| Method | F1 Score | Description |
|--------|----------|-------------|
| Always Predict Negative | 0.00 | Predict no cascade for all graphs |
| Max Loading Threshold | 0.41 | Predict cascade if max line loading > τ |
| Top-K Loading Check | 0.52 | Predict cascade if any of top-K lines > τ |
| **GNN (SSL, 10% labels)** | **0.826** | Learned graph-level predictor |

**Threshold Selection Protocol:**
- **Max Loading Threshold**: τ=0.8 selected by sweeping [0.5, 1.0] on validation set
- **Top-K Loading Check**: K=5, τ=0.7 selected via grid search on validation set
- **Global application**: Same threshold applied to all test graphs

**No Test Leakage Guarantee**: All hyperparameters and thresholds were tuned exclusively on the validation set. The test set was used only for final metric computation.

---

## V-E. Reproducibility and Artifacts

### P1: Code Availability

All experiments are reproducible via provided scripts:

```bash
# Generate all figures and tables
python scripts/generate_all_figures.py

# Run specific experiments
python scripts/finetune_cascade.py --grid ieee24 --seeds 42,123,456,789,1337
python scripts/train_pf_opf.py --task pf --seeds 42,123,456,789,1337
python scripts/train_pf_opf.py --task opf --seeds 42,123,456,789,1337
```

**Artifact List:**
- Pre-trained encoder checkpoints
- Multi-seed result JSON files
- LaTeX table generators
- Figure generation scripts

---

## LaTeX Draft

```latex
\section{Experimental Setup}

\subsection{Datasets and Splits}
We evaluate on the PowerGraph benchmark using IEEE 24-bus and 118-bus test cases (Table~\ref{tab:data}). All quantities are normalized to per-unit (100 MVA base). We use 80/10/10 train/validation/test splits with stratified sampling for cascade prediction.

\begin{table}[t]
\caption{PowerGraph benchmark datasets and train/val/test splits.}
\label{tab:data}
\centering
\begin{tabular}{lrrrrr}
\toprule
Grid & Buses & Lines & Train & Val & Test \\
\midrule
IEEE 24-bus & 24 & 68 & 16,125 & 2,016 & 2,016 \\
IEEE 118-bus & 118 & 370 & 91,875 & 11,484 & 11,484 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Low-Label Protocol}
We evaluate at label fractions \{10\%, 20\%, 50\%, 100\%\}. Results are mean $\pm$ std across 5 seeds (42, 123, 456, 789, 1337). All hyperparameters are tuned on validation only.

\subsection{Training Details}
PhysicsGuidedEncoder with 4 layers, hidden dim 128, dropout 0.1. AdamW optimizer (lr=1e-3), batch size 64, early stopping with patience 10.

\begin{table}[t]
\caption{Model and optimization hyperparameters.}
\label{tab:hparams}
\centering
\begin{tabular}{lr}
\toprule
Parameter & Value \\
\midrule
Hidden Dimension & 128 \\
Layers & 4 \\
Dropout & 0.1 \\
Optimizer & AdamW \\
Learning Rate & 1e-3 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Baselines}
We compare against XGBoost, Random Forest, and heuristic baselines (Table~\ref{tab:mlbaselines}, \ref{tab:heuristics}). GNNs outperform feature-based ML by 4-6$\times$ on power flow MAE.

\subsection{Reproducibility}
All experiments reproducible via provided scripts. Multi-seed results and LaTeX tables auto-generated.
```
