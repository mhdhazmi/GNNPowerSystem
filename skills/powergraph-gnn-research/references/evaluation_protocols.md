# Evaluation Protocols Reference

## Overview

Comprehensive evaluation for publishable results:

1. **Task metrics** - PF/OPF accuracy, cascade F1
2. **Physics consistency** - KCL/KVL residuals
3. **Explanation fidelity** - AUC vs ground-truth masks
4. **Robustness** - Performance under perturbations
5. **Statistical significance** - Paired tests, confidence intervals

## Power Flow Metrics

### Voltage Magnitude

```python
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_voltage_magnitude(pred, target):
    """Evaluate voltage magnitude predictions."""
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    mae = mean_absolute_error(target_np, pred_np)
    rmse = np.sqrt(mean_squared_error(target_np, pred_np))
    mape = np.mean(np.abs((target_np - pred_np) / (target_np + 1e-8))) * 100
    
    # Max error (important for safety)
    max_error = np.abs(target_np - pred_np).max()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'max_error': max_error,
    }
```

### Voltage Angle

```python
def evaluate_angle(sin_pred, cos_pred, sin_target, cos_target):
    """
    Evaluate angle predictions using sin/cos representation.
    Handles wrap-around correctly.
    """
    # Recover angles
    pred_angle = torch.atan2(sin_pred, cos_pred)
    target_angle = torch.atan2(sin_target, cos_target)
    
    # Angular difference (handles wrap-around)
    diff = torch.abs(pred_angle - target_angle)
    diff = torch.min(diff, 2 * np.pi - diff)  # Shortest path on circle
    
    mae_rad = diff.mean().item()
    mae_deg = np.degrees(mae_rad)
    
    max_error_deg = np.degrees(diff.max().item())
    
    return {
        'mae_rad': mae_rad,
        'mae_deg': mae_deg,
        'max_error_deg': max_error_deg,
    }
```

### Line Flow

```python
def evaluate_line_flow(pred, target):
    """Evaluate line power flow predictions."""
    # Active and reactive flow
    metrics = {}
    
    for flow_type in ['p_flow', 'q_flow']:
        if f'{flow_type}_pred' in pred:
            pred_val = pred[f'{flow_type}_pred']
            target_val = target[flow_type]
            
            metrics[f'{flow_type}_mae'] = (pred_val - target_val).abs().mean().item()
            metrics[f'{flow_type}_rmse'] = torch.sqrt(
                ((pred_val - target_val) ** 2).mean()
            ).item()
    
    return metrics
```

## Physics Consistency Metrics

### KCL Mismatch (Power Balance)

```python
def compute_kcl_mismatch(
    v_mag,
    v_angle,
    edge_index,
    y_bus,  # Admittance matrix or line parameters
    p_inject,  # Net injection at each bus
    q_inject,
):
    """
    Compute Kirchhoff's Current Law mismatch.
    
    For each bus: P_inject - sum(P_flow_out) should be ~0
    """
    # This is a simplified version; full implementation requires
    # power flow equations with admittance matrix
    
    num_nodes = v_mag.shape[0]
    
    # Compute line flows from voltages (simplified DC approximation)
    # For AC: need full Y-bus and complex calculations
    
    p_mismatch = torch.zeros(num_nodes)
    q_mismatch = torch.zeros(num_nodes)
    
    # ... (implementation depends on data format)
    
    return {
        'p_mismatch_mean': p_mismatch.abs().mean().item(),
        'p_mismatch_max': p_mismatch.abs().max().item(),
        'q_mismatch_mean': q_mismatch.abs().mean().item(),
        'q_mismatch_max': q_mismatch.abs().max().item(),
    }


def physics_residual_metric(model_output, data):
    """
    High-level physics residual computation.
    
    Compare against ground-truth solver output as sanity check:
    - Ground truth should have near-zero mismatch
    - Model should not be much worse
    """
    # Predicted state
    pred_v_mag = model_output['v_mag']
    pred_angle = torch.atan2(
        model_output['sin_theta'],
        model_output['cos_theta']
    )
    
    # Ground truth state
    gt_v_mag = data.y_v_mag
    gt_angle = torch.atan2(data.y_v_ang_sin, data.y_v_ang_cos)
    
    # Compute mismatches for both
    # (Simplified: just track deviation from GT)
    
    v_diff = (pred_v_mag - gt_v_mag).abs()
    angle_diff = torch.abs(pred_angle - gt_angle)
    angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
    
    return {
        'v_residual_mean': v_diff.mean().item(),
        'v_residual_max': v_diff.max().item(),
        'angle_residual_mean': angle_diff.mean().item(),
        'angle_residual_max': angle_diff.max().item(),
    }
```

## OPF Metrics

```python
def evaluate_opf(pred, target, gen_mask):
    """Evaluate OPF predictions."""
    
    # Generator setpoints (only at generator buses)
    pg_pred = pred['pg'][gen_mask]
    pg_target = target['pg'][gen_mask]
    
    pg_mae = (pg_pred - pg_target).abs().mean().item()
    pg_rmse = torch.sqrt(((pg_pred - pg_target) ** 2).mean()).item()
    
    # Total cost
    cost_pred = pred['cost']
    cost_target = target['cost']
    cost_error = (cost_pred - cost_target).abs().item()
    cost_error_pct = cost_error / (cost_target.abs().item() + 1e-8) * 100
    
    return {
        'pg_mae': pg_mae,
        'pg_rmse': pg_rmse,
        'cost_mae': cost_error,
        'cost_mape': cost_error_pct,
    }
```

## Cascade Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

def evaluate_cascade_classification(pred_logits, target_labels):
    """Evaluate cascade severity classification."""
    
    pred_labels = pred_logits.argmax(dim=-1).cpu().numpy()
    target_np = target_labels.cpu().numpy()
    
    accuracy = accuracy_score(target_np, pred_labels)
    f1_macro = f1_score(target_np, pred_labels, average='macro')
    f1_weighted = f1_score(target_np, pred_labels, average='weighted')
    
    # Per-class metrics
    precision = precision_score(target_np, pred_labels, average=None)
    recall = recall_score(target_np, pred_labels, average=None)
    
    cm = confusion_matrix(target_np, pred_labels)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'confusion_matrix': cm,
    }
```

## Explanation Fidelity

This is crucial for publishability using PowerGraph's ground-truth masks.

```python
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_edge_importance(model, data, method='gradient'):
    """
    Compute edge importance scores for explanation.
    
    Methods:
    - 'gradient': Gradient of output w.r.t. edge features
    - 'attention': Attention weights (if using GAT)
    - 'perturbation': Output change when edge is removed
    """
    
    if method == 'gradient':
        return gradient_based_importance(model, data)
    elif method == 'attention':
        return attention_based_importance(model, data)
    elif method == 'perturbation':
        return perturbation_based_importance(model, data)
    else:
        raise ValueError(f"Unknown method: {method}")


def gradient_based_importance(model, data):
    """Gradient-based edge importance."""
    model.eval()
    
    data.edge_attr.requires_grad_(True)
    
    output = model(data.x, data.edge_index, data.edge_attr, tasks=['cascade'])
    
    # Use cascade prediction as target
    cascade_score = output['cascade']['logits'].sum()
    cascade_score.backward()
    
    # Importance = gradient magnitude
    importance = data.edge_attr.grad.abs().sum(dim=-1)
    
    data.edge_attr.requires_grad_(False)
    
    return importance.detach()


def evaluate_explanation_fidelity(
    edge_importance: torch.Tensor,
    ground_truth_mask: torch.Tensor,
):
    """
    Compare predicted edge importance to ground truth.
    
    Args:
        edge_importance: Predicted importance scores [num_edges]
        ground_truth_mask: Binary ground truth [num_edges], 1 = important
    """
    imp_np = edge_importance.cpu().numpy()
    gt_np = ground_truth_mask.cpu().numpy()
    
    # AUC-ROC
    auc = roc_auc_score(gt_np, imp_np)
    
    # Average Precision (better for imbalanced)
    ap = average_precision_score(gt_np, imp_np)
    
    # Precision@K
    k = int(gt_np.sum())  # Number of true important edges
    top_k_idx = np.argsort(imp_np)[-k:]
    precision_at_k = gt_np[top_k_idx].mean()
    
    # Recall@K
    recall_at_k = gt_np[top_k_idx].sum() / gt_np.sum()
    
    return {
        'explanation_auc': auc,
        'explanation_ap': ap,
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'k': k,
    }


def aggregate_explanation_metrics(model, test_loader, device='cuda'):
    """Aggregate explanation metrics over test set."""
    
    all_aucs = []
    all_aps = []
    all_p_at_k = []
    
    model.eval()
    
    for batch in test_loader:
        batch = batch.to(device)
        
        # Skip if no explanation mask
        if not hasattr(batch, 'exp_mask'):
            continue
        
        importance = compute_edge_importance(model, batch, method='gradient')
        
        metrics = evaluate_explanation_fidelity(importance, batch.exp_mask)
        
        all_aucs.append(metrics['explanation_auc'])
        all_aps.append(metrics['explanation_ap'])
        all_p_at_k.append(metrics['precision_at_k'])
    
    return {
        'mean_auc': np.mean(all_aucs),
        'std_auc': np.std(all_aucs),
        'mean_ap': np.mean(all_aps),
        'mean_p_at_k': np.mean(all_p_at_k),
    }
```

## Robustness Evaluation

```python
def evaluate_robustness(
    model,
    test_loader,
    perturbations: list = ['edge_drop', 'load_scale', 'noise'],
    device: str = 'cuda',
):
    """
    Evaluate model robustness under perturbations.
    """
    results = {'clean': evaluate_clean(model, test_loader, device)}
    
    for perturb in perturbations:
        if perturb == 'edge_drop':
            for drop_rate in [0.05, 0.1, 0.2]:
                transform = EdgeDropTransform(drop_rate)
                perturbed_loader = apply_transform(test_loader, transform)
                
                key = f'edge_drop_{drop_rate}'
                results[key] = evaluate_clean(model, perturbed_loader, device)
        
        elif perturb == 'load_scale':
            for scale in [0.9, 1.1, 1.2, 1.3]:
                transform = LoadScalingTransform((scale, scale))
                perturbed_loader = apply_transform(test_loader, transform)
                
                key = f'load_scale_{scale}'
                results[key] = evaluate_clean(model, perturbed_loader, device)
        
        elif perturb == 'noise':
            for noise_std in [0.01, 0.05, 0.1]:
                transform = GaussianNoiseTransform(noise_std)
                perturbed_loader = apply_transform(test_loader, transform)
                
                key = f'noise_{noise_std}'
                results[key] = evaluate_clean(model, perturbed_loader, device)
    
    return results


def plot_robustness_curves(results):
    """Plot robustness curves (metric vs perturbation strength)."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Edge drop
    drop_rates = [0, 0.05, 0.1, 0.2]
    metrics = [results['clean']['v_mae']]
    for rate in drop_rates[1:]:
        metrics.append(results[f'edge_drop_{rate}']['v_mae'])
    
    axes[0].plot(drop_rates, metrics, 'o-')
    axes[0].set_xlabel('Edge Drop Rate')
    axes[0].set_ylabel('Voltage MAE')
    axes[0].set_title('Edge Drop Robustness')
    
    # ... similar for other perturbations
    
    plt.tight_layout()
    return fig
```

## Statistical Significance

```python
from scipy import stats

def paired_t_test(metric_baseline, metric_proposed, alpha=0.05):
    """
    Paired t-test comparing proposed vs baseline.
    
    Args:
        metric_baseline: List of per-sample metrics from baseline
        metric_proposed: List of per-sample metrics from proposed method
        alpha: Significance level
    
    Returns:
        dict with t-statistic, p-value, and whether significant
    """
    t_stat, p_value = stats.ttest_rel(metric_baseline, metric_proposed)
    
    significant = p_value < alpha
    
    # Effect size (Cohen's d for paired samples)
    diff = np.array(metric_proposed) - np.array(metric_baseline)
    cohens_d = diff.mean() / diff.std()
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'cohens_d': cohens_d,
        'improvement': -diff.mean(),  # Negative because lower is better
    }


def bootstrap_confidence_interval(
    metric_values,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
):
    """Compute bootstrap confidence interval."""
    bootstrapped_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(metric_values, size=len(metric_values), replace=True)
        bootstrapped_means.append(sample.mean())
    
    lower = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
    
    return {
        'mean': np.mean(metric_values),
        'ci_lower': lower,
        'ci_upper': upper,
        'ci_width': upper - lower,
    }


def compare_methods(
    results_baseline: dict,
    results_proposed: dict,
    metrics: list = ['v_mae', 'angle_mae', 'cascade_f1'],
):
    """
    Comprehensive comparison with statistical tests.
    """
    comparison = {}
    
    for metric in metrics:
        baseline_vals = results_baseline[f'{metric}_per_sample']
        proposed_vals = results_proposed[f'{metric}_per_sample']
        
        # Statistical test
        test_result = paired_t_test(baseline_vals, proposed_vals)
        
        # Confidence intervals
        baseline_ci = bootstrap_confidence_interval(baseline_vals)
        proposed_ci = bootstrap_confidence_interval(proposed_vals)
        
        comparison[metric] = {
            'baseline_mean': baseline_ci['mean'],
            'baseline_ci': (baseline_ci['ci_lower'], baseline_ci['ci_upper']),
            'proposed_mean': proposed_ci['mean'],
            'proposed_ci': (proposed_ci['ci_lower'], proposed_ci['ci_upper']),
            **test_result,
        }
    
    return comparison
```

## Complete Evaluation Pipeline

```python
def full_evaluation(
    model,
    test_loader,
    baseline_model=None,
    device='cuda',
    save_dir='./results',
):
    """
    Run complete evaluation pipeline.
    """
    import json
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Task metrics
    print("Evaluating task metrics...")
    results['pf'] = evaluate_pf(model, test_loader, device)
    results['opf'] = evaluate_opf(model, test_loader, device)
    results['cascade'] = evaluate_cascade(model, test_loader, device)
    
    # 2. Physics consistency
    print("Evaluating physics consistency...")
    results['physics'] = evaluate_physics_consistency(model, test_loader, device)
    
    # 3. Explanation fidelity
    print("Evaluating explanation fidelity...")
    results['explanation'] = aggregate_explanation_metrics(model, test_loader, device)
    
    # 4. Robustness
    print("Evaluating robustness...")
    results['robustness'] = evaluate_robustness(model, test_loader, device=device)
    
    # 5. Statistical comparison (if baseline provided)
    if baseline_model is not None:
        print("Comparing to baseline...")
        baseline_results = {
            'pf': evaluate_pf(baseline_model, test_loader, device),
            'cascade': evaluate_cascade(baseline_model, test_loader, device),
        }
        results['comparison'] = compare_methods(baseline_results, results)
    
    # Save results
    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate summary report
    generate_report(results, save_dir / 'evaluation_report.md')
    
    return results
```
