# Physics-Guided GNN Encoder: Code Explanation & Numeric Example

## Overview

This code implements a **Graph Neural Network (GNN)** encoder specifically designed for power grid analysis. It incorporates **physics principles** (Kirchhoff's laws) by weighting message passing with **line admittance** values.

---

## Key Concepts

### 1. **Graph Neural Networks (GNNs)**
- Power grids are represented as graphs: **nodes** = buses, **edges** = transmission lines
- GNNs learn node representations by aggregating information from neighboring nodes

### 2. **Message Passing**
- Each node sends "messages" to its neighbors
- Nodes aggregate incoming messages to update their features
- This allows nodes to learn from their local graph structure

### 3. **Physics Guidance**
- **Admittance (Y)**: Electrical property of transmission lines (inverse of impedance)
- Higher admittance = stronger connection = more current flow
- The model weights messages by admittance, embedding Kirchhoff's current law

---

## Code Structure

### Class 1: `PhysicsGuidedConv` (Lines 13-57)

A single message-passing layer that weights messages by admittance.

**Components:**
- `lin_node`: Transforms node features
- `lin_edge`: Transforms edge features  
- `admittance_scale`: Learns to extract admittance magnitude from edge features

**Flow:**
1. Transform node features: `x → lin_node(x)`
2. Compute admittance weights: `edge_attr → sigmoid(admittance_scale(edge_attr))`
3. Transform edge features: `edge_attr → lin_edge(edge_attr)`
4. Message passing: For each edge (i←j), compute `y_mag * (x_j + edge_emb)`
5. Aggregate: Sum all incoming messages for each node

### Class 2: `PhysicsGuidedEncoder` (Lines 60-106)

Multi-layer encoder that stacks `PhysicsGuidedConv` layers.

**Features:**
- Residual connections (skip connections)
- Layer normalization
- Dropout for regularization
- ReLU activation

### Class 3: `SimpleGNNEncoder` (Lines 109-159)

Baseline encoder using standard GINEConv (without physics weighting).

---

## Numeric Example

Let's walk through a concrete example with a small power grid.

### Setup: 3-Node Power Grid

```
    Node 0 ──── Node 1 ──── Node 2
              (edge 0)   (edge 1)
```

**Graph Structure:**
- 3 nodes (buses)
- 2 edges (transmission lines)
- Edge 0: connects node 0 ↔ node 1
- Edge 1: connects node 1 ↔ node 2

### Step-by-Step Forward Pass

#### **Initial Data**

```python
# Node features (e.g., voltage magnitude, phase angle, load)
# Shape: [num_nodes, node_in_dim]
x = torch.tensor([
    [1.0, 0.5],  # Node 0 features
    [1.0, 0.3],  # Node 1 features  
    [0.9, 0.2],  # Node 2 features
])

# Edge index: [2, num_edges] - defines graph connectivity
# First row = source nodes, second row = target nodes
edge_index = torch.tensor([
    [0, 1],  # Edge 0: 0→1
    [1, 2],  # Edge 1: 1→2
]).t().contiguous()  # Transpose to get [2, 2]

# Edge features (e.g., admittance magnitude, resistance, reactance)
# Shape: [num_edges, edge_in_dim]
edge_attr = torch.tensor([
    [0.8, 0.1],  # Edge 0: admittance=0.8, resistance=0.1
    [0.6, 0.15], # Edge 1: admittance=0.6, resistance=0.15
])
```

#### **Encoder Initialization**

```python
encoder = PhysicsGuidedEncoder(
    node_in_dim=2,      # Input node features: 2D
    edge_in_dim=2,      # Input edge features: 2D
    hidden_dim=4,       # Hidden dimension: 4 (small for example)
    num_layers=2,       # 2 layers
    dropout=0.1
)
```

#### **Layer 0: Input Embedding**

```python
# Step 1: Embed node features
# node_embed: Linear(2 → 4)
x_embedded = encoder.node_embed(x)
# Example output (with random weights):
# x_embedded = [
#     [0.5, -0.2, 0.8, 0.1],  # Node 0
#     [0.4, -0.1, 0.7, 0.2],  # Node 1
#     [0.3,  0.0, 0.6, 0.3],  # Node 2
# ]

# Step 2: Embed edge features
# edge_embed: Linear(2 → 4)
edge_attr_embedded = encoder.edge_embed(edge_attr)
# Example output:
# edge_attr_embedded = [
#     [0.6, -0.1, 0.5, 0.2],  # Edge 0
#     [0.4, -0.2, 0.3, 0.1],  # Edge 1
# ]
```

#### **Layer 1: First PhysicsGuidedConv**

Let's trace through the first convolution layer:

**Step 1: Node Transformation**
```python
# conv.lin_node: Linear(4 → 4)
x_transformed = conv.lin_node(x_embedded)
# Example: x_transformed = [
#     [0.3, 0.1, 0.6, 0.2],  # Node 0
#     [0.2, 0.2, 0.5, 0.3],  # Node 1
#     [0.1, 0.3, 0.4, 0.4],  # Node 2
# ]
```

**Step 2: Compute Admittance Weights**
```python
# admittance_scale: Linear(4 → 1), then sigmoid
# Input: edge_attr_embedded (shape: [2, 4])
y_mag = torch.sigmoid(conv.admittance_scale(edge_attr_embedded))
# Example output (scalar per edge):
# y_mag = [
#     [0.7],  # Edge 0 weight (strong connection)
#     [0.5],  # Edge 1 weight (weaker connection)
# ]
```

**Step 3: Edge Feature Transformation**
```python
# lin_edge: Linear(4 → 4)
edge_emb = conv.lin_edge(edge_attr_embedded)
# Example: edge_emb = [
#     [0.4, 0.1, 0.3, 0.2],  # Edge 0
#     [0.2, 0.0, 0.1, 0.1],  # Edge 1
# ]
```

**Step 4: Message Passing**

For each edge, compute messages:

**Edge 0: Node 0 → Node 1**
```python
# x_j = x_transformed[0] = [0.3, 0.1, 0.6, 0.2]
# edge_attr = edge_emb[0] = [0.4, 0.1, 0.3, 0.2]
# y_mag = 0.7

message_0_to_1 = y_mag[0] * (x_j + edge_attr)
                = 0.7 * ([0.3, 0.1, 0.6, 0.2] + [0.4, 0.1, 0.3, 0.2])
                = 0.7 * [0.7, 0.2, 0.9, 0.4]
                = [0.49, 0.14, 0.63, 0.28]
```

**Edge 1: Node 1 → Node 2**
```python
# x_j = x_transformed[1] = [0.2, 0.2, 0.5, 0.3]
# edge_attr = edge_emb[1] = [0.2, 0.0, 0.1, 0.1]
# y_mag = 0.5

message_1_to_2 = y_mag[1] * (x_j + edge_attr)
                = 0.5 * ([0.2, 0.2, 0.5, 0.3] + [0.2, 0.0, 0.1, 0.1])
                = 0.5 * [0.4, 0.2, 0.6, 0.4]
                = [0.20, 0.10, 0.30, 0.20]
```

**Step 5: Aggregate Messages (Sum)**

```python
# Node 0: No incoming messages → [0, 0, 0, 0]
# Node 1: Receives from Node 0 → [0.49, 0.14, 0.63, 0.28]
# Node 2: Receives from Node 1 → [0.20, 0.10, 0.30, 0.20]

aggr_out = [
    [0.00, 0.00, 0.00, 0.00],  # Node 0
    [0.49, 0.14, 0.63, 0.28],  # Node 1
    [0.20, 0.10, 0.30, 0.20],  # Node 2
]
```

**Step 6: Layer Normalization, ReLU, Dropout, Residual**

```python
# Layer normalization
x_new = norm(aggr_out)
# Example: x_new = [
#     [-0.71, -0.71, -0.71, -0.71],  # Node 0 (normalized zeros)
#     [ 0.45, -0.15,  1.35,  0.35],  # Node 1
#     [ 0.00, -0.50,  1.00,  0.50],  # Node 2
# ]

# ReLU activation
x_new = F.relu(x_new)
# Example: x_new = [
#     [0.00, 0.00, 0.00, 0.00],  # Node 0
#     [0.45, 0.00, 1.35, 0.35],  # Node 1
#     [0.00, 0.00, 1.00, 0.50],  # Node 2
# ]

# Dropout (10% chance to zero)
# Assume no dropout for this example

# Residual connection
x = x_embedded + x_new
# x = [
#     [0.5, -0.2, 0.8, 0.1] + [0.00, 0.00, 0.00, 0.00] = [0.5, -0.2, 0.8, 0.1],
#     [0.4, -0.1, 0.7, 0.2] + [0.45, 0.00, 1.35, 0.35] = [0.85, -0.1, 2.05, 0.55],
#     [0.3,  0.0, 0.6, 0.3] + [0.00, 0.00, 1.00, 0.50] = [0.3, 0.0, 1.6, 0.8],
# ]
```

#### **Layer 2: Second PhysicsGuidedConv**

The process repeats with the updated `x` values. After 2 layers, we get final node embeddings that encode:
- Local graph structure
- Physics-guided relationships (admittance-weighted)
- Multi-hop neighborhood information

---

## Key Insights

1. **Physics Weighting**: Messages from nodes connected by high-admittance lines have more influence (higher `y_mag`)

2. **Residual Connections**: Help preserve information across layers and enable deeper networks

3. **Multi-Layer**: Each layer aggregates information from neighbors, so after N layers, nodes can "see" N hops away

4. **Learnable Parameters**: The model learns how to:
   - Extract admittance from edge features (`admittance_scale`)
   - Transform node/edge features (`lin_node`, `lin_edge`)
   - Balance physics guidance with learned patterns

---

## Why This Matters for Power Grids

- **Kirchhoff's Laws**: Current conservation at nodes (sum of incoming = sum of outgoing)
- **Admittance**: Determines how much current flows through each line
- **Graph Structure**: Power flows follow the network topology
- **This Model**: Learns to respect these physical constraints while being flexible enough to capture complex patterns

