# Self-Supervised Learning Models: Code Explanation & Numeric Example

## Overview

This code implements **self-supervised learning (SSL)** models for pretraining the physics-guided encoder on **unlabeled power grid data**. SSL enables the model to learn useful representations without requiring labeled data, similar to BERT-style pretraining.

**Key Idea**: Mask parts of the input and train the model to reconstruct them from context.

---

## Key Concepts

### 1. **Self-Supervised Learning**
- Learn from **unlabeled data** by creating supervision signals
- Pretrain encoder on large unlabeled datasets
- Transfer learned representations to downstream tasks

### 2. **Masking Strategies**
- **BERT-style masking**: Randomly mask features
- **80% mask token**: Replace with learnable mask token
- **10% random**: Replace with random values
- **10% unchanged**: Keep original (denoising effect)

### 3. **Reconstruction Tasks**
- **Node reconstruction**: Predict masked node features
- **Edge reconstruction**: Predict masked edge features
- **Combined**: Both node and edge reconstruction

---

## Code Structure

### Class 1: `MaskedNodeReconstruction` (Lines 17-168)

SSL model for masked node feature reconstruction.

**Architecture:**
```
Input Graph → Mask Nodes → Encoder → Reconstruction Head → Reconstructed Features
                                                              ↓
                                                          Loss (MSE)
```

**Components:**
- `mask_token`: Learnable mask token (replaces masked features)
- `encoder`: Physics-guided encoder (shared with downstream tasks)
- `reconstruction_head`: MLP that predicts original node features

### Class 2: `MaskedEdgeReconstruction` (Lines 171-284)

SSL model for masked edge feature reconstruction.

**Architecture:**
```
Input Graph → Mask Edges → Encoder → Edge Reconstruction Head → Reconstructed Edges
                                                                    ↓
                                                                Loss (MSE)
```

**Key Difference**: Edge reconstruction uses **endpoint node embeddings** (concatenate source + target).

### Class 3: `CombinedSSL` (Lines 287-419)

Combined SSL with both node and edge reconstruction.

**Architecture:**
```
Input Graph → Mask Nodes & Edges → Encoder → Two Heads
                                              ├── Node Head → Node Loss
                                              └── Edge Head → Edge Loss
                                                      ↓
                                              Combined Loss
```

---

## Numeric Example: Masked Node Reconstruction

### Setup: 3-Node Power Grid

```python
# Node features: [voltage_magnitude, phase_angle, load]
x = torch.tensor([
    [1.0, 0.5, 0.3],  # Node 0
    [1.0, 0.3, 0.5],  # Node 1
    [0.9, 0.2, 0.4],  # Node 2
])

edge_index = torch.tensor([[0, 1], [1, 2]]).t()
edge_attr = torch.tensor([[0.8, 0.1], [0.6, 0.15]])
```

### Step 1: Create Mask

```python
# Randomly select 15% of nodes to mask
mask_ratio = 0.15
num_nodes = 3
mask_indices = torch.rand(3) < 0.15
# Example: mask_indices = [False, True, False]  # Node 1 selected

# Store original
original_x = x.clone()
# original_x = [[1.0, 0.5, 0.3],
#               [1.0, 0.3, 0.5],
#               [0.9, 0.2, 0.4]]
```

### Step 2: Apply Masking Strategy

```python
masked_x = x.clone()

# For each masked node, randomly choose strategy:
# 80% → mask token, 10% → random, 10% → unchanged
rand_val = torch.rand(1)  # Example: 0.65

if rand_val < 0.8:  # 80% chance: mask token
    masked_x[1] = mask_token  # [0.0, 0.0, 0.0] (learned)
elif rand_val < 0.9:  # 10% chance: random
    masked_x[1] = x[random_node]  # Replace with random node's features
else:  # 10% chance: unchanged
    pass  # Keep original

# Example result:
# masked_x = [[1.0, 0.5, 0.3],  # Unchanged
#             [0.0, 0.0, 0.0],   # Masked (mask token)
#             [0.9, 0.2, 0.4]]  # Unchanged
```

### Step 3: Encode Masked Input

```python
# Encoder processes masked graph
node_emb = encoder(masked_x, edge_index, edge_attr)
# Shape: [3, 128]

# Node embeddings now encode:
# - Node 0: Original features + context from neighbors
# - Node 1: Mask token + context from neighbors (must infer features!)
# - Node 2: Original features + context from neighbors
```

### Step 4: Reconstruct Node Features

```python
# Reconstruction head: Linear(128 → 128) → ReLU → Dropout → Linear(128 → 3)
reconstructed = reconstruction_head(node_emb)
# Shape: [3, 3]

# Example output:
# reconstructed = [[1.02, 0.48, 0.31],  # Node 0 (close to original)
#                  [0.98, 0.32, 0.49],   # Node 1 (predicted from context!)
#                  [0.91, 0.19, 0.41]]   # Node 2 (close to original)
```

### Step 5: Compute Loss

```python
# Loss only on masked nodes
loss = MSE(reconstructed[mask_indices], original_x[mask_indices])
#      = MSE([0.98, 0.32, 0.49], [1.0, 0.3, 0.5])
#      = mean((0.98-1.0)² + (0.32-0.3)² + (0.49-0.5)²)
#      = mean(0.0004 + 0.0004 + 0.0001)
#      = 0.0003
```

**Key Point**: The model learns to predict Node 1's features from its neighbors (Node 0 and Node 2), learning useful graph structure!

---

## Numeric Example: Masked Edge Reconstruction

### Setup

```python
x = torch.tensor([[1.0, 0.5], [1.0, 0.3], [0.9, 0.2]])
edge_index = torch.tensor([[0, 1], [1, 2]]).t()
edge_attr = torch.tensor([
    [0.8, 0.1],  # Edge 0: 0 → 1
    [0.6, 0.15], # Edge 1: 1 → 2
])
```

### Step 1: Create Edge Mask

```python
mask_ratio = 0.15
num_edges = 2
edge_mask = torch.rand(2) < 0.15
# Example: edge_mask = [True, False]  # Edge 0 selected

original_edge_attr = edge_attr.clone()
```

### Step 2: Apply Masking

```python
masked_edge_attr = edge_attr.clone()
masked_edge_attr[0] = edge_mask_token  # [0.0, 0.0]

# masked_edge_attr = [[0.0, 0.0],    # Masked
#                     [0.6, 0.15]]    # Unchanged
```

### Step 3: Encode with Masked Edges

```python
# Encoder processes graph with masked edge
node_emb = encoder(x, edge_index, masked_edge_attr)
# Shape: [3, 128]

# Node embeddings encode:
# - Node 0: Features + masked edge to Node 1
# - Node 1: Features + masked edge from Node 0 + normal edge to Node 2
# - Node 2: Features + normal edge from Node 1
```

### Step 4: Reconstruct Edge Features

```python
# Edge reconstruction uses endpoint embeddings
src, dst = edge_index
# src = [0, 1], dst = [1, 2]

# Concatenate source and target embeddings
edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)
# Shape: [2, 256]  # 128 + 128

# For Edge 0 (0 → 1):
# edge_emb[0] = concat([node_emb[0], node_emb[1]]) = [128 dims from node 0, 128 dims from node 1]

# Edge reconstruction head: Linear(256 → 128) → ReLU → Dropout → Linear(128 → 2)
reconstructed = edge_reconstruction_head(edge_emb)
# Shape: [2, 2]

# Example output:
# reconstructed = [[0.79, 0.11],   # Edge 0 (predicted from node embeddings!)
#                 [0.61, 0.14]]    # Edge 1 (close to original)
```

### Step 5: Compute Loss

```python
# Loss only on masked edges
loss = MSE(reconstructed[edge_mask], original_edge_attr[edge_mask])
#      = MSE([0.79, 0.11], [0.8, 0.1])
#      = mean((0.79-0.8)² + (0.11-0.1)²)
#      = mean(0.0001 + 0.0001)
#      = 0.0001
```

**Key Point**: The model learns to predict edge features from the node embeddings at its endpoints!

---

## Numeric Example: Combined SSL

### Setup

```python
x = torch.tensor([[1.0, 0.5], [1.0, 0.3], [0.9, 0.2]])
edge_index = torch.tensor([[0, 1], [1, 2]]).t()
edge_attr = torch.tensor([[0.8, 0.1], [0.6, 0.15]])
```

### Step 1: Create Both Masks

```python
node_mask = torch.rand(3) < 0.15  # [False, True, False]
edge_mask = torch.rand(2) < 0.15  # [True, False]

original_x = x.clone()
original_edge_attr = edge_attr.clone()
```

### Step 2: Apply Masking

```python
masked_x = x.clone()
masked_x[1] = node_mask_token  # [0.0, 0.0]

masked_edge_attr = edge_attr.clone()
masked_edge_attr[0] = edge_mask_token  # [0.0, 0.0]

# Both node and edge are masked!
```

### Step 3: Encode

```python
# Encoder processes graph with both node and edge masking
node_emb = encoder(masked_x, edge_index, masked_edge_attr)
# Must infer both Node 1's features AND Edge 0's features from context!
```

### Step 4: Reconstruct Both

```python
# Node reconstruction
node_reconstructed = node_head(node_emb)
# Shape: [3, 2]

# Edge reconstruction
src, dst = edge_index
edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)
edge_reconstructed = edge_head(edge_emb)
# Shape: [2, 2]
```

### Step 5: Combined Loss

```python
# Node loss
node_loss = MSE(node_reconstructed[node_mask], original_x[node_mask])
#            = MSE([0.98, 0.32], [1.0, 0.3]) = 0.0002

# Edge loss
edge_loss = MSE(edge_reconstructed[edge_mask], original_edge_attr[edge_mask])
#            = MSE([0.79, 0.11], [0.8, 0.1]) = 0.0001

# Combined loss
total_loss = node_weight * node_loss + edge_weight * edge_loss
#           = 1.0 * 0.0002 + 1.0 * 0.0001
#           = 0.0003
```

---

## Why This Works: Learning Signal

### What the Model Learns

1. **Graph Structure**: Nodes learn to infer features from neighbors
2. **Edge Properties**: Edges learn to encode relationships between nodes
3. **Physics Constraints**: Encoder learns power grid-specific patterns
4. **Robust Representations**: Masking forces model to use context, not just memorize

### Example Learning Scenario

```python
# Scenario: Node 1 is masked
# Node 1 is connected to Node 0 (high voltage) and Node 2 (low voltage)
# Model must infer Node 1's voltage from:
#   - Node 0's features (high voltage, strong connection)
#   - Node 2's features (low voltage, weaker connection)
#   - Edge properties (admittance, resistance)

# The model learns:
#   - High admittance edges → strong influence
#   - Voltage tends to decrease along transmission lines
#   - Load affects voltage magnitude
```

---

## Pretraining Workflow

### Step 1: Pretrain on Unlabeled Data

```python
ssl_model = MaskedNodeReconstruction(...)

for batch in unlabeled_dataloader:
    outputs = ssl_model(x, edge_index, edge_attr)
    loss = outputs['loss']
    loss.backward()
    optimizer.step()
```

### Step 2: Transfer to Downstream Task

```python
# Extract encoder weights
encoder_state = ssl_model.get_encoder_state_dict()

# Initialize downstream model
downstream_model = PowerGraphGNN(...)

# Load pretrained encoder
downstream_model.encoder.load_state_dict(encoder_state)

# Fine-tune on labeled data
for batch in labeled_dataloader:
    outputs = downstream_model(x, edge_index, edge_attr, tasks=("pf",))
    loss = pf_loss(outputs['pf'], targets)
    loss.backward()
    optimizer.step()
```

---

## Key Insights

1. **Masking Strategy**:
   - 80% mask token: Forces model to use context
   - 10% random: Adds robustness (denoising)
   - 10% unchanged: Prevents overfitting to mask token

2. **Reconstruction Targets**:
   - **Node reconstruction**: Learns node-level representations
   - **Edge reconstruction**: Learns edge-level representations
   - **Combined**: Learns both simultaneously

3. **Transfer Learning**:
   - Pretrain on large unlabeled dataset
   - Fine-tune on small labeled dataset
   - Better performance with less labeled data

4. **Why SSL Works**:
   - Power grids have strong structural patterns
   - Masking forces model to learn these patterns
   - Learned representations transfer to downstream tasks

5. **Edge Reconstruction Trick**:
   - Uses **endpoint embeddings** (source + target)
   - Encodes relationship between connected nodes
   - Learns to predict edge properties from node context

---

## Comparison: Node vs Edge vs Combined SSL

| Method | What's Masked | What's Predicted | Use Case |
|--------|---------------|------------------|----------|
| Node SSL | Node features | Node features | Node-centric tasks |
| Edge SSL | Edge features | Edge features | Edge-centric tasks |
| Combined SSL | Both | Both | General pretraining |

**Recommendation**: Use Combined SSL for general pretraining, then fine-tune on specific tasks.

---

## Why This Matters for Power Grids

- **Large Unlabeled Data**: Power grids generate lots of unlabeled operational data
- **Limited Labels**: Labeled data (e.g., cascades) is rare and expensive
- **Transfer Learning**: SSL enables leveraging unlabeled data
- **Better Representations**: Pretrained encoder learns general power grid patterns
- **Faster Convergence**: Fine-tuning converges faster than training from scratch

