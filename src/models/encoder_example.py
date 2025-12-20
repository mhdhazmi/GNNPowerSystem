"""
Runnable numeric example demonstrating PhysicsGuidedEncoder
"""

import torch
import torch.nn as nn
from encoder import PhysicsGuidedConv, PhysicsGuidedEncoder

# Set random seed for reproducibility
torch.manual_seed(42)

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
# STEP 1: Single PhysicsGuidedConv Layer
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Single PhysicsGuidedConv Layer")
print("=" * 70)

conv = PhysicsGuidedConv(in_channels=2, out_channels=4, edge_dim=2)

print("\n🔧 Layer Components:")
print(f"  - lin_node: Linear(2 → 4)")
print(f"  - lin_edge: Linear(2 → 4)")
print(f"  - admittance_scale: Linear(2 → 1)")

# Initialize with specific weights for demonstration
with torch.no_grad():
    # Simple initialization for clarity
    nn.init.xavier_uniform_(conv.lin_node.weight)
    nn.init.xavier_uniform_(conv.lin_edge.weight)
    nn.init.xavier_uniform_(conv.admittance_scale.weight)
    conv.lin_node.bias.zero_()
    conv.lin_edge.bias.zero_()
    conv.admittance_scale.bias.zero_()

print("\n📥 Input:")
print(f"  Node features:\n{x}")
print(f"  Edge attributes:\n{edge_attr}")

# Forward pass
with torch.no_grad():
    # Step 1: Transform nodes
    x_transformed = conv.lin_node(x)
    print(f"\n1️⃣ Node transformation (lin_node):")
    print(f"   Shape: {x_transformed.shape}")
    print(f"   Values:\n{x_transformed}")
    
    # Step 2: Compute admittance weights
    y_mag_logits = conv.admittance_scale(edge_attr)
    y_mag = torch.sigmoid(y_mag_logits)
    print(f"\n2️⃣ Admittance weights (sigmoid(admittance_scale)):")
    print(f"   Logits: {y_mag_logits.squeeze()}")
    print(f"   Weights: {y_mag.squeeze()}")
    print(f"   → Edge 0 weight: {y_mag[0].item():.3f} (stronger connection)")
    print(f"   → Edge 1 weight: {y_mag[1].item():.3f} (weaker connection)")
    
    # Step 3: Transform edges
    edge_emb = conv.lin_edge(edge_attr)
    print(f"\n3️⃣ Edge embedding (lin_edge):")
    print(f"   Shape: {edge_emb.shape}")
    print(f"   Values:\n{edge_emb}")
    
    # Step 4: Message passing
    print(f"\n4️⃣ Message Passing:")
    print(f"   For each edge (i←j), compute: y_mag * (x_j + edge_emb)")
    
    # Manually compute messages for clarity
    messages = []
    for e_idx in range(edge_index.size(1)):
        src = edge_index[0, e_idx].item()
        dst = edge_index[1, e_idx].item()
        x_j = x_transformed[src]
        edge_e = edge_emb[e_idx]
        y_m = y_mag[e_idx]
        
        message = y_m * (x_j + edge_e)
        messages.append((dst, message))
        
        print(f"\n   Edge {e_idx}: Node {src} → Node {dst}")
        print(f"     x_j (source node): {x_j}")
        print(f"     edge_emb: {edge_e}")
        print(f"     y_mag: {y_m.item():.3f}")
        print(f"     message = {y_m.item():.3f} * ({x_j} + {edge_e})")
        print(f"             = {message}")
    
    # Aggregate messages
    output = conv(x, edge_index, edge_attr)
    print(f"\n5️⃣ Aggregated output (sum of incoming messages):")
    print(f"   Shape: {output.shape}")
    print(f"   Values:\n{output}")
    print(f"\n   Note: Node 0 has no incoming messages → zeros")
    print(f"         Node 1 receives from Node 0")
    print(f"         Node 2 receives from Node 1")

# ============================================================================
# STEP 2: Full PhysicsGuidedEncoder
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Full PhysicsGuidedEncoder (2 layers)")
print("=" * 70)

encoder = PhysicsGuidedEncoder(
    node_in_dim=2,
    edge_in_dim=2,
    hidden_dim=4,
    num_layers=2,
    dropout=0.0  # Disable dropout for clarity
)

print("\n🔧 Encoder Architecture:")
print(f"  - Input embedding: Linear(2 → 4)")
print(f"  - Edge embedding: Linear(2 → 4)")
print(f"  - 2 × PhysicsGuidedConv layers")
print(f"  - LayerNorm + ReLU + Residual connections")

# Forward pass
with torch.no_grad():
    output = encoder(x, edge_index, edge_attr)
    
    print(f"\n📤 Final node embeddings:")
    print(f"   Shape: {output.shape}")
    print(f"   Values:\n{output}")
    
    print(f"\n✨ Each node now has a 4D embedding that encodes:")
    print(f"   - Local graph structure")
    print(f"   - Physics-guided relationships (admittance-weighted)")
    print(f"   - Multi-hop neighborhood information (2 layers = 2 hops)")

# ============================================================================
# COMPARISON: With vs Without Physics Guidance
# ============================================================================
print("\n" + "=" * 70)
print("COMPARISON: Message Weights")
print("=" * 70)

print("\n🔍 Key Insight:")
print("   In PhysicsGuidedConv, messages are weighted by admittance:")
print("   - High admittance → Strong connection → Larger message weight")
print("   - Low admittance → Weak connection → Smaller message weight")
print("\n   This embeds Kirchhoff's laws: current flow depends on admittance!")

print("\n" + "=" * 70)
print("Example complete! 🎉")
print("=" * 70)

