# III. Problem Formulation

---

## III-A. Graph Representation and Features

### P1: Grid as Graph

A power grid is naturally represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes $\mathcal{V}$ correspond to electrical buses (substations with loads and/or generators) and edges $\mathcal{E}$ correspond to transmission lines and transformers. Each node $i \in \mathcal{V}$ is associated with a feature vector $\mathbf{x}_i$, and each directed edge $(i,j) \in \mathcal{E}$ has features $\mathbf{e}_{ij}$.

### P2: Feature Definitions

We define node and edge features based on standard power system quantities, normalized to the per-unit (p.u.) system with 100 MVA base.

**Table A: Graph Feature Definitions**

| Feature | Symbol | Description | Typical Range |
|---------|--------|-------------|---------------|
| **Node Features** | | | |
| Net Active Power | $P_{net}$ | Real power injection (generation - load) | -1.0 to 1.0 p.u. |
| Net Apparent Power | $S_{net}$ | Complex power magnitude | 0 to 1.5 p.u. |
| Voltage Magnitude | $V$ | Voltage level at the bus | 0.95 to 1.05 p.u. |
| **Edge Features** | | | |
| Active Power Flow | $P_{ij}$ | Real power flowing on line | -1.0 to 1.0 p.u. |
| Reactive Power Flow | $Q_{ij}$ | Reactive power flowing | -0.5 to 0.5 p.u. |
| Line Reactance | $X_{ij}$ | Electrical impedance | 0.01 to 0.5 p.u. |
| Line Rating | $lr_{ij}$ | Thermal capacity limit | 0.5 to 2.0 p.u. |

```python
# Feature tensors
x = [P_net, S_net, V]        # Shape: [num_nodes, 3]
edge_attr = [P_ij, Q_ij, X, rating]  # Shape: [num_edges, 4]
edge_index = [[sources], [targets]]   # Shape: [2, num_edges]
```

---

## III-B. Downstream Tasks and Metrics

### P1: Task Definitions

We evaluate on three operationally relevant power system tasks spanning different granularities:

**Table B: Task Specifications with Units**

| Task | Input | Output | Granularity | Metric | Direction |
|------|-------|--------|-------------|--------|-----------|
| **Cascade Prediction** | Pre-outage grid state | Binary (cascade/no cascade) | Graph-level | F1 Score | Higher ↑ |
| **Power Flow** | Load injections ($P$, $Q$) | Voltage magnitudes ($V_{mag}$) | Node-level | MAE (p.u.) | Lower ↓ |
| **Line Flow** | Bus states ($P$, $Q$, $V$, $\theta$) | Line flows ($P_{ij}$, $Q_{ij}$) | Edge-level | MAE (p.u.) | Lower ↓ |

### P2: Metric Definitions

- **Cascade F1**: Standard binary F1 computed at graph level. Positive class = grid experiences cascading failures (Demand Not Served > 0 MW).
- **Power Flow MAE**: Mean absolute error in voltage magnitude, averaged over buses then graphs.
- **Line Flow MAE**: Mean absolute error over both $P_{ij}$ and $Q_{ij}$ components, averaged over edges then graphs.

**Improvement Computation:**
- For F1 (higher = better): $\text{Improvement} = \frac{\text{SSL} - \text{Scratch}}{\text{Scratch}} \times 100\%$
- For MAE (lower = better): $\text{Improvement} = \frac{\text{Scratch} - \text{SSL}}{\text{Scratch}} \times 100\%$

Both yield positive values when SSL outperforms Scratch.

---

## III-C. Inference-Time Observability

### P1: Deployment Assumptions

A key practical consideration is what information is available at inference time. We ensure all model inputs are observable from standard SCADA/PMU measurements without requiring oracle information.

**Table C: Required Inputs at Inference (Observability)**

| Task | Observable Inputs | Predicted Output | Real-Time Available? |
|------|-------------------|------------------|---------------------|
| **Cascade** | $P_{load}$, $Q_{load}$, $V_{mag}$, $V_{angle}$, line status | $P(\text{cascade})$ — graph-level binary | Yes (SCADA/PMU) |
| **Power Flow** | $P_{injection}$, $Q_{injection}$ | $V_{mag}$ at all buses | Yes (SCADA) |
| **Line Flow** | $P_{load}$, $Q_{load}$, $V_{mag}$, $V_{angle}$ | $P_{ij}$, $Q_{ij}$ on all lines | Yes (SCADA/PMU) |

**Deployment Note**: No oracle information (future failures, ground-truth power flows for PF/Line Flow tasks) is required at inference. All inputs are available from standard grid monitoring infrastructure.

---

## LaTeX Draft

```latex
\section{Problem Formulation}

\subsection{Graph Representation}
A power grid is represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where nodes correspond to buses and edges to transmission lines. Node features $\mathbf{x}_i = [P_{net}, S_{net}, V]$ capture power injection and voltage. Edge features $\mathbf{e}_{ij} = [P_{ij}, Q_{ij}, X_{ij}, lr_{ij}]$ capture power flow and line parameters. All quantities are normalized to per-unit (100 MVA base).

\begin{table}[t]
\caption{Task specifications, I/O, metrics, and units.}
\label{tab:tasks}
\centering
\begin{tabular}{llll}
\toprule
Task & Output & Granularity & Metric \\
\midrule
Cascade & Binary label & Graph-level & F1 \\
Power Flow & $V_{mag}$ & Node-level & MAE \\
Line Flow & $P_{ij}, Q_{ij}$ & Edge-level & MAE \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Tasks and Metrics}
We evaluate on three tasks (Table~\ref{tab:tasks}): cascade prediction (graph-level F1), power flow (node-level MAE), and line flow (edge-level MAE). Improvement is computed as relative gain, with positive values indicating SSL outperforms Scratch.

\subsection{Inference-Time Observability}
All model inputs are available from SCADA/PMU measurements. No oracle information is required at inference time.
```
