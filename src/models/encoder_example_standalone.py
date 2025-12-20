"""
Standalone numeric example demonstrating PhysicsGuidedEncoder concepts
No imports required - pure PyTorch demonstration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("PHYSICS-GUIDED GNN ENCODER: NUMERIC EXAMPLE")
print("=" * 70)

# ============================================================================
# SETUP: 3-Node Power Grid
# ============================================================================
print("\n📊 GRAPH STRUCTURE:")
print("   Node 0 ──── Node 1 ──── Node 2")
print("            (edge 0)    (edge 1)")

# Node features: [voltage_magnitude, phase_angle]
x = torch.tensor([
    [1.0, 0.5],  # Node 0
    [1.0, 0.3],  # Node 1
    [0.9, 0.2],  # Node 2
], dtype=torch.float32)

# Edge index: [source, target] format
edge_index = torch.tensor([
    [0, 1],  # Edge 0: 0 → 1
    [1, 2],  # Edge 1: 1 → 2
], dtype=torch.long).t().contiguous()

# Edge features: [admittance_magnitude, resistance]
edge_attr = torch.tensor([
    [0.8, 0.1],  # Edge 0: high admittance (strong connection)
    [0.6, 0.15], # Edge 1: lower admittance (weaker connection)
], dtype=torch.float32)

print(f"\nNode features (shape {x.shape}):")
print(x)
print(f"\nEdge index (shape {edge_index.shape}):")
print(edge_index)
print(f"\nEdge attributes (shape {edge_attr.shape}):")
print(edge_attr)

# ============================================================================
# STEP 1: Manual Message Passing Calculation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Manual Message Passing (Physics-Guided)")
print("=" * 70)

# Initialize simple linear layers
torch.manual_seed(42)
lin_node = nn.Linear(2, 4)
lin_edge = nn.Linear(2, 4)
admittance_scale = nn.Linear(2, 1)

# Initialize weights for demonstration
nn.init.xavier_uniform_(lin_node.weight)
nn.init.xavier_uniform_(lin_edge.weight)
nn.init.xavier_uniform_(admittance_scale.weight)
lin_node.bias.zero_()
lin_edge.bias.zero_()
admittance_scale.bias.zero_()

print("\n🔧 Layer Components:")
print(f"  - lin_node: Linear(2 → 4)")
print(f"  - lin_edge: Linear(2 → 4)")
print(f"  - admittance_scale: Linear(2 → 1)")

with torch.no_grad():
    # Step 1: Transform nodes
    x_transformed = lin_node(x)
    print(f"\n1️⃣ Node transformation (lin_node):")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_transformed.shape}")
    print(f"   Values:\n{x_transformed}")
    
    # Step 2: Compute admittance weights
    y_mag_logits = admittance_scale(edge_attr)
    y_mag = torch.sigmoid(y_mag_logits)
    print(f"\n2️⃣ Admittance weights (sigmoid(admittance_scale)):")
    print(f"   Edge attributes input:\n{edge_attr}")
    print(f"   Logits: {y_mag_logits.squeeze()}")
    print(f"   Weights: {y_mag.squeeze()}")
    print(f"   → Edge 0 weight: {y_mag[0].item():.3f} (stronger connection)")
    print(f"   → Edge 1 weight: {y_mag[1].item():.3f} (weaker connection)")
    
    # Step 3: Transform edges
    edge_emb = lin_edge(edge_attr)
    print(f"\n3️⃣ Edge embedding (lin_edge):")
    print(f"   Input shape: {edge_attr.shape}")
    print(f"   Output shape: {edge_emb.shape}")
    print(f"   Values:\n{edge_emb}")
    
    # Step 4: Message passing
    print(f"\n4️⃣ Message Passing:")
    print(f"   Formula: message = y_mag * (x_j + edge_emb)")
    print(f"   For each edge (source → target):")
    
    # Initialize output tensor
    output = torch.zeros_like(x_transformed)
    
    # Process each edge
    for e_idx in range(edge_index.size(1)):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        
        x_j = x_transformed[src]  # Source node features
        edge_e = edge_emb[e_idx]  # Edge embedding
        y_m = y_mag[e_idx]        # Admittance weight
        
        # Compute message
        message = y_m * (x_j + edge_e)
        
        # Aggregate (sum) into target node
        output[dst] += message
        
        print(f"\n   Edge {e_idx}: Node {src} → Node {dst}")
        print(f"     Source node features (x_j): {x_j}")
        print(f"     Edge embedding: {edge_e}")
        print(f"     Admittance weight: {y_m.item():.3f}")
        print(f"     Message = {y_m.item():.3f} * ({x_j} + {edge_e})")
        print(f"             = {message}")
    
    print(f"\n5️⃣ Aggregated output (sum of incoming messages per node):")
    print(f"   Shape: {output.shape}")
    print(f"   Values:\n{output}")
    print(f"\n   Explanation:")
    print(f"     - Node 0: No incoming messages → [0, 0, 0, 0]")
    print(f"     - Node 1: Receives message from Node 0 (weighted by admittance)")
    print(f"     - Node 2: Receives message from Node 1 (weighted by admittance)")

# ============================================================================
# STEP 2: Multi-Layer Encoder Simulation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Multi-Layer Encoder (2 layers)")
print("=" * 70)

print("\n🔧 Encoder Architecture:")
print(f"  - Input embedding: Linear(2 → 4)")
print(f"  - Edge embedding: Linear(2 → 4)")
print(f"  - 2 × PhysicsGuidedConv layers")
print(f"  - LayerNorm + ReLU + Residual connections")

# Initialize embedding layers
torch.manual_seed(42)
node_embed = nn.Linear(2, 4)
edge_embed = nn.Linear(2, 4)
norm1 = nn.LayerNorm(4)
norm2 = nn.LayerNorm(4)

nn.init.xavier_uniform_(node_embed.weight)
nn.init.xavier_uniform_(edge_embed.weight)
node_embed.bias.zero_()
edge_embed.bias.zero_()

with torch.no_grad():
    # Initial embeddings
    x_emb = node_embed(x)
    edge_attr_emb = edge_embed(edge_attr)
    
    print(f"\n📥 Initial embeddings:")
    print(f"   Node embeddings:\n{x_emb}")
    print(f"   Edge embeddings:\n{edge_attr_emb}")
    
    # Layer 1
    print(f"\n🔄 Layer 1:")
    x1 = lin_node(x_emb)
    y_mag1 = torch.sigmoid(admittance_scale(edge_attr_emb))
    edge_emb1 = lin_edge(edge_attr_emb)
    
    # Message passing
    output1 = torch.zeros_like(x1)
    for e_idx in range(edge_index.size(1)):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        message = y_mag1[e_idx] * (x1[src] + edge_emb1[e_idx])
        output1[dst] += message
    
    # Normalize, activate, residual
    output1_norm = norm1(output1)
    output1_relu = F.relu(output1_norm)
    x_layer1 = x_emb + output1_relu
    
    print(f"   After message passing:\n{output1}")
    print(f"   After LayerNorm + ReLU + Residual:\n{x_layer1}")
    
    # Layer 2
    print(f"\n🔄 Layer 2:")
    x2 = lin_node(x_layer1)
    y_mag2 = torch.sigmoid(admittance_scale(edge_attr_emb))
    edge_emb2 = lin_edge(edge_attr_emb)
    
    # Message passing
    output2 = torch.zeros_like(x2)
    for e_idx in range(edge_index.size(1)):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        message = y_mag2[e_idx] * (x2[src] + edge_emb2[e_idx])
        output2[dst] += message
    
    # Normalize, activate, residual
    output2_norm = norm2(output2)
    output2_relu = F.relu(output2_norm)
    x_final = x_layer1 + output2_relu
    
    print(f"   After message passing:\n{output2}")
    print(f"   After LayerNorm + ReLU + Residual:\n{x_final}")
    
    print(f"\n📤 Final node embeddings:")
    print(f"   Shape: {x_final.shape}")
    print(f"   Values:\n{x_final}")
    
    print(f"\n✨ Each node now has a 4D embedding that encodes:")
    print(f"   - Local graph structure")
    print(f"   - Physics-guided relationships (admittance-weighted)")
    print(f"   - Multi-hop neighborhood information (2 layers = 2 hops)")

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("\n🔍 Physics Guidance:")
print("   - Messages are weighted by admittance (y_mag)")
print("   - High admittance → Strong connection → Larger message weight")
print("   - Low admittance → Weak connection → Smaller message weight")
print("   - This embeds Kirchhoff's laws: current flow depends on admittance!")

print("\n🔄 Message Passing Flow:")
print("   1. Transform node features")
print("   2. Compute admittance-based weights from edge features")
print("   3. Transform edge features")
print("   4. For each edge: message = y_mag * (source_node + edge_emb)")
print("   5. Aggregate messages: sum all incoming messages per node")

print("\n📈 Multi-Layer Benefits:")
print("   - Layer 1: Nodes see their direct neighbors")
print("   - Layer 2: Nodes see neighbors of neighbors (2-hop)")
print("   - Residual connections preserve information across layers")

print("\n" + "=" * 70)
print("Example complete! 🎉")
print("=" * 70)

