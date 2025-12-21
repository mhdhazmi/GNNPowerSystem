# Physics Consistency Reference

## Power Flow Formulation

### DC Power Flow Assumptions
- Flat voltage profile (|V| ≈ 1.0 p.u.)
- Small angle differences (sin θ ≈ θ)
- Negligible line resistance (R << X)
- No reactive power

**Check**: If paper claims DC-PF, verify these assumptions are stated.

### AC Power Flow
P_i = Σ_j |V_i||V_j|(G_ij cos θ_ij + B_ij sin θ_ij)
Q_i = Σ_j |V_i||V_j|(G_ij sin θ_ij - B_ij cos θ_ij)

**Check**: If AC-PF, verify both P and Q are predicted.

## Line Flow vs. OPF

⚠️ **CRITICAL DISTINCTION**:

| Aspect | Line Flow Prediction | Optimal Power Flow |
|--------|---------------------|-------------------|
| Task | Predict MW/MVA on lines | Optimize dispatch |
| Output | Line loading values | Generator setpoints |
| Objective | Accuracy (MAE/RMSE) | Cost minimization |
| Constraints | None (regression) | Gen limits, line limits |

**This paper does Line Flow prediction, NOT OPF.**

## Cascading Failure Modeling

### Required Assumptions to State
1. Protection relay model (overcurrent, distance, or simplified)
2. Failure propagation rule (N-1, N-k, or full cascade simulation)
3. Load redistribution model (DC-OPF redispatch or pro-rata)
4. Stopping criterion (stable state, blackout threshold)

### Graph Construction Checks
- Nodes: Buses (load + generation features)
- Edges: Lines (admittance, impedance, rating)
- Edge weights: Admittance-weighted? Inverse impedance?

## Per-Unit System

All values should be in per-unit (p.u.) with stated bases:
- S_base (typically 100 MVA)
- V_base (typically nominal voltage per bus)

**Check**: Are features normalized? Is per-unit conversion documented?

## Sanity Checks to Run

1. **Power balance**: Σ P_gen ≈ Σ P_load + losses
2. **Voltage bounds**: 0.9 ≤ |V| ≤ 1.1 p.u. for normal operation
3. **Angle bounds**: |θ_ij| < 30° for stable operation
4. **Line flow direction**: Sign conventions consistent?
5. **Admittance matrix**: Y symmetric? Diagonal dominant?
6. **Slack bus**: Exactly one per island?
7. **Topology connectivity**: Graph connected during training?
8. **Line ratings**: Predictions within thermal limits?
