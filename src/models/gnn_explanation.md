# PowerGraph GNN Models: Code Explanation & Numeric Example

## Overview

This code implements the **complete multi-task GNN model** for power grid analysis. It combines the physics-guided encoder with task-specific prediction heads for:
- **Power Flow (PF)**: Predict voltage magnitude and phase angle
- **Optimal Power Flow (OPF)**: Predict generator setpoints and total cost
- **Cascade Prediction**: Predict cascading failure severity

---

## Key Concepts

### 1. **Multi-Task Learning**
- Single encoder learns shared representations
- Multiple task-specific heads for different predictions
- Enables knowledge transfer across tasks

### 2. **Task Types**
- **Node-level tasks**: PF, OPF (predictions per bus/node)
- **Graph-level tasks**: Cascade (prediction for entire grid)

### 3. **Model Architecture**
```
Input Graph → Physics-Guided Encoder → Task-Specific Heads
                                      ├── PF Head (node-level)
                                      ├── OPF Head (node-level)
                                      └── Cascade Head (graph-level)
```

---

## Code Structure

### Class 1: `PowerGraphGNN` (Lines 17-113)

The main multi-task model combining encoder and task heads.

**Components:**
- `encoder`: Physics-guided or simple GNN encoder
- `pf_head`: Power flow prediction head
- `opf_head`: Optimal power flow prediction head
- `cascade_head`: Cascade prediction head

**Key Methods:**
- `forward()`: Runs encoder and selected task heads
- `get_embeddings()`: Extract node embeddings for analysis

**Usage:**
```python
model = PowerGraphGNN(
    node_in_dim=3,
    edge_in_dim=4,
    hidden_dim=128,
    num_layers=4,
    encoder_type="physics_guided"
)

# Forward pass with specific tasks
outputs = model(x, edge_index, edge_attr, tasks=("pf", "opf"))
```

### Class 2: `PFBaselineModel` (Lines 115-155)

Simplified model for power flow prediction only.

**Use Case:** Initial experiments before multi-task training

### Class 3: `CascadeBaselineModel` (Lines 158-332)

Model for cascade prediction with explanation methods.

**Key Features:**
- Cascade prediction head
- Three explanation methods:
  1. `get_edge_importance_gradient()`: Gradient-based attribution
  2. `get_edge_importance_attention()`: Attention-like scores
  3. `get_edge_importance_integrated_gradients()`: Integrated gradients

---

## Numeric Example: Multi-Task Forward Pass

### Setup: 3-Node Power Grid

```
    Node 0 ──── Node 1 ──── Node 2
              (edge 0)   (edge 1)
```

### Input Data

```python
# Node features: [voltage_magnitude, phase_angle, load]
x = torch.tensor([
    [1.0, 0.5, 0.3],  # Node 0
    [1.0, 0.3, 0.5],  # Node 1
    [0.9, 0.2, 0.4],  # Node 2
])

# Edge connectivity
edge_index = torch.tensor([
    [0, 1],  # Edge 0: 0 → 1
    [1, 2],  # Edge 1: 1 → 2
]).t()

# Edge features: [admittance_mag, resistance, reactance, capacity]
edge_attr = torch.tensor([
    [0.8, 0.1, 0.05, 1.0],  # Edge 0
    [0.6, 0.15, 0.08, 0.8], # Edge 1
])

# Generator mask (1 = generator, 0 = load)
gen_mask = torch.tensor([1, 0, 1])  # Nodes 0 and 2 are generators
```

### Step 1: Encoder Forward Pass

```python
# Encoder produces node embeddings
# Shape: [3, 128] (3 nodes, 128-dim embeddings)
node_emb = encoder(x, edge_index, edge_attr)
# Example output:
# node_emb = [
#     [0.5, -0.2, 0.8, ..., 0.1],  # Node 0 embedding (128 dims)
#     [0.4, -0.1, 0.7, ..., 0.2],  # Node 1 embedding
#     [0.3,  0.0, 0.6, ..., 0.3],  # Node 2 embedding
# ]
```

### Step 2: Power Flow Head (Node-Level)

```python
# PF head processes each node independently
pf_output = pf_head(node_emb)
# Outputs:
# {
#     'v_mag': [1.02, 1.01, 0.98],      # Predicted voltage magnitudes
#     'sin_theta': [0.48, 0.30, 0.20],  # Predicted sin(phase angle)
#     'cos_theta': [0.88, 0.95, 0.98],  # Predicted cos(phase angle)
# }
```

**Process:**
1. MLP transforms embeddings: `[128] → [64]`
2. Separate heads predict: `v_mag`, `sin_theta`, `cos_theta`
3. Normalize sin/cos to unit circle: `sin² + cos² = 1`

### Step 3: OPF Head (Node-Level + Graph-Level)

```python
# OPF head predicts generator setpoints and total cost
opf_output = opf_head(node_emb, gen_mask, batch=None)
# Outputs:
# {
#     'pg': [0.5, 0.0, 0.3],  # Generator setpoints (MW)
#                              # Node 1 = 0 (not a generator)
#     'cost': [125.5],        # Total generation cost ($)
# }
```

**Process:**
1. **Generator setpoints**: MLP predicts per-node generation
   - Apply `gen_mask` to zero non-generator nodes
   - Example: `pg[1] = 0` because `gen_mask[1] = 0`
   
2. **Total cost**: Graph-level pooling
   - Average all node embeddings: `graph_emb = mean(node_emb)`
   - MLP predicts cost: `[128] → [1]`

### Step 4: Cascade Head (Graph-Level)

```python
# Cascade head predicts binary classification
cascade_output = cascade_head(node_emb, batch=None)
# Outputs:
# {
#     'logits': [0.2],        # Logit for cascade probability
#     'attention': [[0.3],    # Attention weights per node
#                   [0.4],
#                   [0.3]]
# }
```

**Process:**
1. **Attention pooling**: Compute attention weights
   ```python
   att = sigmoid(gate(node_emb))  # [3, 1]
   # att = [[0.3], [0.4], [0.3]]
   ```

2. **Weighted aggregation**: Sum attention-weighted embeddings
   ```python
   graph_emb = sum(att * node_emb)  # [128]
   # Weighted combination of all node embeddings
   ```

3. **Classification**: MLP predicts cascade logit
   ```python
   logits = classifier(graph_emb)  # [1]
   ```

---

## Loss Functions

### Power Flow Loss (`pf_loss`)

```python
loss = loss_v_mag + loss_sin + loss_cos + λ * loss_norm

# Components:
# - loss_v_mag: MSE between predicted and true voltage magnitude
# - loss_sin: MSE between predicted and true sin(θ)
# - loss_cos: MSE between predicted and true cos(θ)
# - loss_norm: Soft constraint that sin² + cos² = 1
```

**Example:**
```python
pred = {'v_mag': [1.02, 1.01], 'sin_theta': [0.48, 0.30], 'cos_theta': [0.88, 0.95]}
target_v_mag = [1.0, 1.0]
target_sin = [0.5, 0.3]
target_cos = [0.87, 0.95]

loss_v = MSE([1.02, 1.01], [1.0, 1.0]) = 0.0002
loss_sin = MSE([0.48, 0.30], [0.5, 0.3]) = 0.0002
loss_cos = MSE([0.88, 0.95], [0.87, 0.95]) = 0.0001
loss_norm = MSE([0.48²+0.88², 0.30²+0.95²], [1.0, 1.0]) = 0.0001
total = 0.0002 + 0.0002 + 0.0001 + 0.1 * 0.0001 = 0.00051
```

### Cascade Loss (`cascade_loss`)

```python
# Binary classification: BCE with logits
loss = BCE_with_logits(logits, targets)

# Multi-class: Cross-entropy
loss = CrossEntropy(logits, targets)
```

**Example (Binary):**
```python
logits = [0.2, -0.5, 0.8]  # 3 graphs
targets = [1, 0, 1]         # Labels

loss = BCE_with_logits([0.2, -0.5, 0.8], [1, 0, 1])
     = -log(σ(0.2)) - log(1-σ(-0.5)) - log(σ(0.8))
     ≈ 0.60 + 0.47 + 0.31 = 1.38
```

---

## Edge Importance Explanation Methods

### Method 1: Gradient-Based Attribution

```python
# Compute gradient of prediction w.r.t. edge features
edge_attr.requires_grad = True
output = model(x, edge_index, edge_attr)
score = output['logits'].sum()
score.backward()

# Importance = magnitude of gradient
importance = edge_attr.grad.abs().mean(dim=1)
```

**Intuition:** Edges with large gradients are more important for prediction.

### Method 2: Attention-Based Scores

```python
# Combine three factors:
# 1. Admittance (1 / reactance)
admittance = 1.0 / (edge_attr[:, 2] + ε)

# 2. Edge feature magnitude
edge_magnitude = edge_attr.abs().mean(dim=1)

# 3. Node embedding similarity
src_emb = node_emb[edge_index[0]]
dst_emb = node_emb[edge_index[1]]
similarity = (src_emb * dst_emb).sum(dim=1)

# Weighted combination
importance = 0.4 * admittance_norm + 0.3 * edge_mag_norm + 0.3 * sim_norm
```

**Intuition:** High admittance + large features + similar endpoints = important edge.

### Method 3: Integrated Gradients

```python
# Integrate gradients along path from baseline to actual
baseline = zeros_like(edge_attr)
for α in [0.05, 0.10, ..., 1.0]:
    interpolated = baseline + α * (edge_attr - baseline)
    grad = compute_gradient(interpolated)
    accumulated += grad

importance = (edge_attr - baseline) * (accumulated / steps)
```

**Intuition:** More robust than simple gradients - averages over interpolation path.

---

## Key Insights

1. **Shared Encoder**: All tasks share the same encoder, enabling knowledge transfer

2. **Task-Specific Heads**: Each head is specialized for its task:
   - PF: Node-level regression (voltage, angle)
   - OPF: Node-level + graph-level (generation, cost)
   - Cascade: Graph-level classification

3. **Flexible Forward Pass**: Can run specific tasks via `tasks` parameter:
   ```python
   outputs = model(x, edge_index, edge_attr, tasks=("pf",))  # Only PF
   outputs = model(x, edge_index, edge_attr, tasks=("pf", "opf", "cascade"))  # All
   ```

4. **Explanation Methods**: Cascade model includes multiple ways to understand edge importance

5. **Baseline Models**: Simplified models for initial experiments before multi-task training

---

## Why This Architecture?

- **Efficiency**: Single encoder shared across tasks reduces parameters
- **Transfer Learning**: Encoder learns general power grid representations
- **Modularity**: Easy to add/remove task heads
- **Interpretability**: Explanation methods help understand model decisions

