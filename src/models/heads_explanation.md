# Task-Specific Prediction Heads: Code Explanation & Numeric Example

## Overview

This code implements **task-specific prediction heads** that take node embeddings from the encoder and produce task-specific outputs:
- **PowerFlowHead**: Predicts voltage magnitude and phase angle
- **OPFHead**: Predicts generator setpoints and total cost
- **CascadeHead**: Predicts cascading failure severity (multi-class)
- **CascadeBinaryHead**: Binary cascade prediction (cascade vs no cascade)

---

## Key Concepts

### 1. **Head Architecture**
- Heads are **task-specific** neural networks
- Take **shared node embeddings** as input
- Produce **task-specific outputs**

### 2. **Task Types**
- **Node-level**: Predictions per node (PF, OPF generation)
- **Graph-level**: Single prediction per graph (Cascade, OPF cost)

### 3. **Pooling Strategies**
- **Mean pooling**: Average node embeddings
- **Attention pooling**: Weighted sum with learned attention
- **Global pooling**: Aggregate all nodes into single representation

---

## Code Structure

### Class 1: `PowerFlowHead` (Lines 13-52)

Predicts voltage magnitude and phase angle for each node.

**Key Features:**
- Uses **sin/cos representation** for angles (avoids discontinuity at ±π)
- **Normalization**: Enforces `sin²θ + cos²θ = 1`

**Architecture:**
```
Node Embedding [128] → MLP → [64] → Separate Heads
                                    ├── v_mag: [1]
                                    ├── sin_theta: [1]
                                    └── cos_theta: [1]
```

### Class 2: `OPFHead` (Lines 55-98)

Predicts generator setpoints (per node) and total cost (per graph).

**Architecture:**
```
Node Embedding [128] → Generator MLP → pg [1] (per node)
                    ↓
              Graph Pooling → Cost MLP → cost [1] (per graph)
```

### Class 3: `CascadeHead` (Lines 101-150)

Multi-class cascade prediction with attention-based pooling.

**Architecture:**
```
Node Embeddings [N, 128] → Attention Weights → Weighted Pooling
                                                      ↓
                                              Graph Embedding [128]
                                                      ↓
                                              Classifier → Logits [num_classes]
```

### Class 4: `CascadeBinaryHead` (Lines 153-183)

Simplified binary cascade prediction.

**Architecture:**
```
Node Embeddings [N, 128] → Mean Pooling → Graph Embedding [128]
                                              ↓
                                          MLP → Logit [1]
```

---

## Numeric Example: Power Flow Head

### Input: Node Embeddings

```python
# After encoder, we have embeddings for 3 nodes
# Shape: [3, 128]
node_emb = torch.tensor([
    [0.5, -0.2, 0.8, ..., 0.1],  # Node 0 (128 dims)
    [0.4, -0.1, 0.7, ..., 0.2],  # Node 1
    [0.3,  0.0, 0.6, ..., 0.3],  # Node 2
])
```

### Step 1: MLP Transformation

```python
# MLP: Linear(128 → 128) → ReLU → Linear(128 → 64) → ReLU
h = mlp(node_emb)
# Shape: [3, 64]
# Example output:
# h = [
#     [0.3, 0.1, 0.5, ..., 0.2],  # Node 0
#     [0.2, 0.2, 0.4, ..., 0.3],  # Node 1
#     [0.1, 0.3, 0.3, ..., 0.4],  # Node 2
# ]
```

### Step 2: Separate Predictions

```python
# Three separate linear heads
v_mag = v_mag_head(h)      # Linear(64 → 1)
sin_theta = sin_head(h)    # Linear(64 → 1)
cos_theta = cos_head(h)    # Linear(64 → 1)

# Example outputs (before normalization):
v_mag_raw = [1.05, 1.02, 0.98]
sin_theta_raw = [0.52, 0.32, 0.22]
cos_theta_raw = [0.90, 0.95, 0.97]
```

### Step 3: Normalize Sin/Cos to Unit Circle

```python
# Critical: Ensure sin² + cos² = 1
for each node:
    norm = sqrt(sin² + cos² + ε)
    sin_theta = sin_theta_raw / norm
    cos_theta = cos_theta_raw / norm

# Example:
# Node 0: norm = sqrt(0.52² + 0.90²) = sqrt(1.08) ≈ 1.04
#         sin_theta = 0.52 / 1.04 ≈ 0.50
#         cos_theta = 0.90 / 1.04 ≈ 0.87
#         Check: 0.50² + 0.87² ≈ 1.00 ✓

# Final outputs:
v_mag = [1.05, 1.02, 0.98]
sin_theta = [0.50, 0.32, 0.23]
cos_theta = [0.87, 0.95, 0.97]
```

**Why sin/cos?** Phase angles wrap around at ±π, causing discontinuity. Sin/cos representation is continuous and differentiable.

---

## Numeric Example: OPF Head

### Input: Node Embeddings + Generator Mask

```python
node_emb = torch.tensor([
    [0.5, -0.2, 0.8, ..., 0.1],  # Node 0
    [0.4, -0.1, 0.7, ..., 0.2],  # Node 1
    [0.3,  0.0, 0.6, ..., 0.3],  # Node 2
])

gen_mask = torch.tensor([1, 0, 1])  # Nodes 0 and 2 are generators
```

### Step 1: Generator Setpoint Prediction (Per Node)

```python
# MLP: Linear(128 → 128) → ReLU → Linear(128 → 1)
pg_raw = gen_mlp(node_emb)
# Shape: [3, 1]
# Example: pg_raw = [[0.6], [0.4], [0.3]]

# Apply generator mask
pg = pg_raw.squeeze(-1) * gen_mask
# pg = [0.6, 0.0, 0.3]
# Node 1 is not a generator → pg[1] = 0
```

### Step 2: Graph-Level Cost Prediction

```python
# Pool node embeddings to graph-level
if batch is None:  # Single graph
    graph_emb = node_emb.mean(dim=0)  # Average pooling
    # Shape: [128]
    # graph_emb = mean([node_0, node_1, node_2]) = [0.4, -0.1, 0.7, ..., 0.2]

# MLP: Linear(128 → 128) → ReLU → Linear(128 → 1)
cost = cost_mlp(graph_emb)
# Shape: [1]
# Example: cost = [125.5]  # Total generation cost in $
```

**Note:** For batched graphs, use `global_mean_pool()` which averages within each graph separately.

---

## Numeric Example: Cascade Head (Attention-Based)

### Input: Node Embeddings

```python
node_emb = torch.tensor([
    [0.5, -0.2, 0.8, ..., 0.1],  # Node 0
    [0.4, -0.1, 0.7, ..., 0.2],  # Node 1
    [0.3,  0.0, 0.6, ..., 0.3],  # Node 2
])
# Shape: [3, 128]
```

### Step 1: Compute Attention Weights

```python
# Gate: Linear(128 → 1)
att_logits = gate(node_emb)
# Shape: [3, 1]
# Example: att_logits = [[0.2], [-0.1], [0.3]]

# Apply sigmoid
att = torch.sigmoid(att_logits)
# att = [[0.55], [0.48], [0.57]]
# These are attention weights (importance scores per node)
```

### Step 2: Weighted Pooling

```python
# Weighted sum: sum(attention * embedding)
graph_emb = (att * node_emb).sum(dim=0)
# Shape: [128]

# Manual calculation:
# graph_emb = 0.55 * node_0 + 0.48 * node_1 + 0.57 * node_2
#           = 0.55 * [0.5, -0.2, ...] + 0.48 * [0.4, -0.1, ...] + 0.57 * [0.3, 0.0, ...]
#           = [0.275 + 0.192 + 0.171, -0.11 + -0.048 + 0.0, ...]
#           = [0.638, -0.158, ...]
```

### Step 3: Classification

```python
# Classifier: Linear(128 → 128) → ReLU → Dropout → Linear(128 → num_classes)
logits = classifier(graph_emb)
# Shape: [num_classes]
# Example (binary): logits = [0.2]  # Logit for cascade probability
# Example (multi-class): logits = [-0.5, 0.8, 0.1]  # Logits for 3 classes
```

**Output:**
```python
{
    'logits': [0.2],
    'attention': [[0.55], [0.48], [0.57]]  # Can be used for interpretability
}
```

---

## Numeric Example: Cascade Binary Head (Simplified)

### Input: Node Embeddings

```python
node_emb = torch.tensor([
    [0.5, -0.2, 0.8, ..., 0.1],  # Node 0
    [0.4, -0.1, 0.7, ..., 0.2],  # Node 1
    [0.3,  0.0, 0.6, ..., 0.3],  # Node 2
])
```

### Step 1: Mean Pooling

```python
# Simple average pooling
graph_emb = node_emb.mean(dim=0)
# Shape: [128]
# graph_emb = [0.4, -0.1, 0.7, ..., 0.2]
```

### Step 2: MLP Transformation

```python
# Pool MLP: Linear(128 → 128) → ReLU → Dropout
h = pool_mlp(graph_emb)
# Shape: [128]
```

### Step 3: Binary Classification

```python
# Classifier: Linear(128 → 1)
logits = classifier(h)
# Shape: [1]
# Example: logits = [0.2]

# To get probability:
prob = torch.sigmoid(logits)  # ≈ 0.55 (55% chance of cascade)
```

---

## Comparison: Cascade Head vs Cascade Binary Head

| Feature | CascadeHead | CascadeBinaryHead |
|---------|-------------|-------------------|
| Pooling | Attention-weighted | Mean pooling |
| Classes | Multi-class | Binary only |
| Output | Logits + attention | Logits only |
| Use Case | When you need interpretability | Simpler binary tasks |

**Attention vs Mean Pooling:**

```python
# Mean pooling (equal weights)
graph_emb_mean = mean([node_0, node_1, node_2])
# All nodes contribute equally

# Attention pooling (learned weights)
att = [0.55, 0.48, 0.57]  # Learned importance
graph_emb_att = 0.55*node_0 + 0.48*node_1 + 0.57*node_2
# Node 2 contributes most (highest attention)
```

---

## Key Insights

1. **Power Flow Head**:
   - Uses sin/cos to avoid angle discontinuity
   - Normalization ensures valid angle representation
   - Node-level predictions (one per bus)

2. **OPF Head**:
   - Dual outputs: node-level (generation) + graph-level (cost)
   - Generator mask zeros non-generator nodes
   - Cost requires graph-level pooling

3. **Cascade Head**:
   - Attention pooling provides interpretability
   - Can identify which nodes are most important
   - Supports multi-class classification

4. **Cascade Binary Head**:
   - Simpler architecture (mean pooling)
   - Faster computation
   - Good for binary classification tasks

5. **Pooling Strategies**:
   - **Mean**: Equal contribution from all nodes
   - **Attention**: Learned importance weights
   - **Global mean**: Handles batched graphs correctly

---

## Why These Design Choices?

- **Sin/Cos for Angles**: Continuous representation avoids ±π discontinuity
- **Separate Heads**: Each task has different output structure
- **Attention Pooling**: Provides interpretability (which nodes matter?)
- **Generator Mask**: Enforces physical constraint (only generators produce power)
- **Graph-Level Pooling**: Necessary for graph-level predictions (cost, cascade)

