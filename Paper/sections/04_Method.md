# IV. Method: Physics-Guided SSL GNN

---

## IV-A. PhysicsGuidedConv Layer

### P1: From Standard to Physics-Guided Message Passing

Standard GNN message passing updates node representations by aggregating neighbor information:

$$h_i^{(l+1)} = \text{UPDATE}\left(h_i^{(l)}, \text{AGGREGATE}\left(\{h_j^{(l)} : j \in \mathcal{N}(i)\}\right)\right)$$

In power grids, however, information flow should respect electrical physics: power flows according to Kirchhoff's laws, with magnitude depending on voltage differences and line admittance. We embed this intuition by weighting messages based on learned line importance:

$$h_i^{(l+1)} = \text{UPDATE}\left(h_i^{(l)}, \sum_{j \in \mathcal{N}(i)} y_{ij} \cdot (h_j^{(l)} + e_{ij})\right)$$

where $y_{ij} = \sigma(W_y \cdot e_{ij})$ is a learned admittance-like weight derived from edge features.

### P2: Physical Intuition

The learned weight $y_{ij}$ mimics the role of line admittance in power flow equations: lines with high power flow or near thermal limits receive higher weights, as they are more electrically significant. Unlike hard physics constraints, this soft weighting allows the model to discover which edge characteristics matter for each task while maintaining physical intuition.

**PhysicsGuidedConv Implementation:**

```
Input: x [N, hidden], edge_index [2, E], edge_attr [E, hidden]

Step 1: Transform node features
  x_transformed = Linear(x)    [N, hidden] → [N, hidden]

Step 2: Compute admittance weights
  y_mag = sigmoid(Linear(edge_attr))   [E, hidden] → [E, 1]

Step 3: Transform edge features
  edge_emb = Linear(edge_attr)   [E, hidden] → [E, hidden]

Step 4: Physics-guided message passing
  For each edge (i, j):
    message_ij = y_mag_ij * (x_j + edge_emb_ij)
  For each node i:
    aggregated_i = SUM(message_ij for all j ∈ neighbors(i))

Output: aggregated [N, hidden]
```

---

## IV-B. PhysicsGuidedEncoder Stack

### P1: Encoder Architecture

The PhysicsGuidedEncoder stacks multiple PhysicsGuidedConv layers with residual connections and layer normalization for stable training:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      INPUT GRAPH                                     │
│  x: [N, 3]  edge_index: [2, E]  edge_attr: [E, 4]                   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INPUT PROJECTION                                │
│   Node Embed: Linear(3 → 128)     Edge Embed: Linear(4 → 128)       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHYSICS-GUIDED ENCODER                            │
│                                                                      │
│   4 × [PhysicsGuidedConv → LayerNorm → ReLU → Dropout → Residual]   │
│                                                                      │
│   Output: node_embeddings [N, 128]                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Configuration:**
- Hidden dimension: 128
- Number of layers: 4
- Dropout: 0.1
- Residual connections between layers

---

## IV-C. Task-Specific Heads

### P1: Shared Encoder, Multiple Heads Design

The core design principle is a **shared encoder** that produces general-purpose node embeddings, with **task-specific heads** adapting these representations to each downstream task. This enables SSL pretraining to benefit all tasks through a single pretrained encoder.

### P2: Power Flow Head (Node-Level Regression)

Predicts voltage magnitude $V_{mag}$ at each bus:

```python
class PowerFlowHead(nn.Module):
    def __init__(self, hidden_dim=128):
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: V_mag
        )
```

**Important**: The PF task uses a feature subset excluding $V$ from node inputs (since $V_{mag}$ is the target), preventing leakage.

### P3: Line Flow Head (Edge-Level Regression)

Predicts active and reactive power flow $(P_{ij}, Q_{ij})$ on each edge:

```python
class LineFlowHead(nn.Module):
    def __init__(self, hidden_dim=128):
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat [src, dst]
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Output: [P_ij, Q_ij]
        )
```

### P4: Cascade Head (Graph-Level Classification)

Predicts graph-level cascade probability via mean pooling and binary classification:

```python
class CascadeHead(nn.Module):
    def __init__(self, hidden_dim=128):
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Logit
            nn.Sigmoid()
        )

    def forward(self, node_embeddings, batch):
        graph_emb = global_mean_pool(node_embeddings, batch)
        return self.graph_head(graph_emb)
```

---

## IV-D. Self-Supervised Pretraining Objective

### P1: Motivation for SSL

Self-supervised pretraining enables learning from unlabeled grid topologies, which are abundant compared to labeled scenarios. By learning to reconstruct masked features, the encoder develops representations capturing power system structure—voltage patterns, power flow relationships, and topological importance—without task-specific labels.

### P2: Masked Reconstruction Objective

We mask 15% of input features and train the encoder to reconstruct them:

**Masking Strategy (BERT-style):**
- 80% of masked positions: replace with zero
- 10% of masked positions: replace with random value
- 10% of masked positions: keep original (verify encoder isn't ignoring)

**Reconstruction Targets:**
- Node features: $P_{net}$, $S_{net}$ (power injections)
- Edge features: $X_{ij}$, $lr_{ij}$ (line parameters)

**Loss Function:**
$$\mathcal{L}_{SSL} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| \hat{x}_i - x_i \|^2$$

where $\mathcal{M}$ is the set of masked positions.

### P3: No Label Leakage Disclosure

**Critical Design Decision**: SSL pretraining uses **only the training partition (80%)** with self-supervised objectives. The validation and test sets are never exposed during pretraining. This ensures:

1. No information leakage from held-out data
2. Fair comparison with scratch training
3. Realistic deployment scenario simulation

**Algorithm 1: SSL Pretraining and Fine-tuning Pipeline**

```
1. Pretrain:
   - Load training graphs (80% of data, no labels)
   - For each epoch:
     - Mask 15% of node/edge features
     - Forward pass through encoder + reconstruction head
     - Compute MSE loss on masked positions only
     - Update encoder weights
   - Save pretrained encoder weights

2. Fine-tune:
   - Initialize encoder from pretrained weights
   - Attach task-specific head (cascade/PF/line flow)
   - Train on labeled subset (10-100% of training set)
   - Validate on validation set (10%)
   - Early stop based on validation metric
   - Evaluate on held-out test set (10%)
```

---

## IV-E. Explainability Method

### P1: Integrated Gradients for Edge Attribution

For cascade prediction, we need to explain *which edges* the model considers important for its prediction. We use Integrated Gradients (IG), which computes attribution by accumulating gradients along a straight-line path from a baseline to the input:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

where $x'$ is a baseline (zero features) and $F$ is the model output.

**Advantages over alternatives:**
- **vs. Basic Gradient**: IG is less noisy, satisfies completeness axiom
- **vs. Attention**: IG reflects actual prediction mechanism, not just learned weights
- **vs. Heuristics**: IG captures non-linear, model-specific importance

**Evaluation Protocol**: We compute per-edge IG attributions and rank edges by importance. AUC-ROC is computed against ground-truth failure edge masks from simulation data.

---

## Figure 1: Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SELF-SUPERVISED PRETRAINING                      │
│                         (Training Set Only)                          │
│                                                                      │
│   [Grid Graph] → [Mask 15%] → [PhysicsGuidedEncoder] → [Reconstruct] │
│                                                                      │
│                     Loss: MSE on masked positions                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                         Save Pretrained Weights
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINE-TUNING                                  │
│                    (Labeled Training Subset)                         │
│                                                                      │
│   [Grid Graph] → [PhysicsGuidedEncoder] → [Task Head] → [Prediction] │
│                      (pretrained)                                    │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│   │  PF Head     │  │ Line Flow    │  │     Cascade Head         │   │
│   │  (V_mag)     │  │ (P_ij,Q_ij)  │  │  (graph probability)     │   │
│   └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## LaTeX Draft

```latex
\section{Method}

\subsection{Physics-Guided Message Passing}
Standard GNN message passing aggregates neighbor information uniformly. We introduce physics-guided weighting:
\begin{equation}
h_i^{(l+1)} = \text{UPDATE}\left(h_i^{(l)}, \sum_{j \in \mathcal{N}(i)} y_{ij} \cdot (h_j^{(l)} + e_{ij})\right)
\end{equation}
where $y_{ij} = \sigma(W_y \cdot e_{ij})$ is a learned admittance-like weight derived from edge features.

\subsection{Encoder and Task Heads}
The PhysicsGuidedEncoder stacks 4 layers with residual connections (hidden dim 128). Task-specific heads adapt the shared representations: node-level for power flow ($V_{mag}$), edge-level for line flow ($P_{ij}, Q_{ij}$), and graph-level for cascade prediction.

\subsection{Self-Supervised Pretraining}
SSL uses masked reconstruction on training data only (80\%). We mask 15\% of node/edge features and minimize MSE on masked positions. Validation and test sets are never exposed during pretraining.

\subsection{Explainability}
Integrated Gradients computes edge attribution by accumulating gradients from baseline to input. We evaluate against ground-truth failure masks using AUC-ROC.
```
