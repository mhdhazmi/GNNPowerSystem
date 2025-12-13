# Uncertainty Quantification Reference

## Overview

Quantify model uncertainty for safe deployment of GNN surrogates:

1. **Epistemic uncertainty** (model uncertainty) - Reducible with more data
2. **Aleatoric uncertainty** (data noise) - Inherent randomness

## Important Note on OPF

OPF labels from deterministic solvers are NOT inherently stochastic. "Aleatoric uncertainty" only makes sense if:
- You model input noise (load forecast errors, renewable variability)
- You sample from an input distribution

Otherwise, focus on **epistemic uncertainty** only.

## Methods

### 1. Deep Ensembles (Recommended)

Train M independent models, aggregate predictions:

```python
import torch
import torch.nn as nn
from typing import List

class DeepEnsemble(nn.Module):
    """
    Ensemble of GNNs for uncertainty estimation.
    """
    
    def __init__(
        self,
        model_class,
        model_kwargs: dict,
        num_members: int = 5,
    ):
        super().__init__()
        
        self.members = nn.ModuleList([
            model_class(**model_kwargs)
            for _ in range(num_members)
        ])
    
    def forward(self, *args, **kwargs):
        """Return mean and variance across ensemble."""
        predictions = [m(*args, **kwargs) for m in self.members]
        
        # Stack predictions
        # Assuming output is dict with tensor values
        keys = predictions[0].keys()
        
        outputs = {}
        for key in keys:
            stacked = torch.stack([p[key] for p in predictions])
            outputs[f'{key}_mean'] = stacked.mean(dim=0)
            outputs[f'{key}_std'] = stacked.std(dim=0)  # Epistemic uncertainty
        
        return outputs


def train_ensemble(
    model_class,
    model_kwargs: dict,
    train_loader,
    num_members: int = 5,
    num_epochs: int = 50,
    device: str = 'cuda',
):
    """Train ensemble with different random seeds."""
    
    ensemble = DeepEnsemble(model_class, model_kwargs, num_members).to(device)
    
    for i, member in enumerate(ensemble.members):
        print(f"Training member {i+1}/{num_members}")
        
        # Different seed per member
        torch.manual_seed(42 + i)
        
        optimizer = torch.optim.AdamW(member.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            member.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                outputs = member(batch.x, batch.edge_index, batch.edge_attr)
                loss = compute_loss(outputs, batch)
                
                loss.backward()
                optimizer.step()
    
    return ensemble
```

### 2. MC Dropout

Use dropout at inference time for uncertainty:

```python
class MCDropoutModel(nn.Module):
    """
    Model with dropout enabled at inference for uncertainty.
    """
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Add dropout layers if not present
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Insert dropout after each layer."""
        # Implementation depends on base_model structure
        pass
    
    def forward(self, *args, num_samples: int = 30, **kwargs):
        """
        Forward pass with MC sampling.
        
        Args:
            num_samples: Number of stochastic forward passes
        """
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.base_model(*args, **kwargs)
                samples.append(out)
        
        # Aggregate
        keys = samples[0].keys()
        outputs = {}
        
        for key in keys:
            stacked = torch.stack([s[key] for s in samples])
            outputs[f'{key}_mean'] = stacked.mean(dim=0)
            outputs[f'{key}_std'] = stacked.std(dim=0)
        
        return outputs


def enable_mc_dropout(model: nn.Module):
    """Enable dropout in eval mode for MC sampling."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
```

### 3. Heteroscedastic Output (Aleatoric)

Model that predicts mean AND variance:

```python
class HeteroscedasticHead(nn.Module):
    """
    Predict both mean and log-variance (aleatoric uncertainty).
    """
    
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        mean = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        # Clamp for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        return mean, logvar
    
    def loss(self, x, target):
        """Negative log-likelihood loss."""
        mean, logvar = self.forward(x)
        var = torch.exp(logvar)
        
        # NLL = 0.5 * (logvar + (target - mean)^2 / var)
        nll = 0.5 * (logvar + (target - mean)**2 / var)
        
        return nll.mean()


class PFHeadWithUncertainty(nn.Module):
    """
    Power flow head with aleatoric uncertainty.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Voltage magnitude: mean + logvar
        self.v_mag = HeteroscedasticHead(hidden_dim, 1)
        self.sin_theta = HeteroscedasticHead(hidden_dim, 1)
        self.cos_theta = HeteroscedasticHead(hidden_dim, 1)
    
    def forward(self, node_embeddings):
        h = self.shared(node_embeddings)
        
        v_mean, v_logvar = self.v_mag(h)
        sin_mean, sin_logvar = self.sin_theta(h)
        cos_mean, cos_logvar = self.cos_theta(h)
        
        return {
            'v_mag_mean': v_mean.squeeze(-1),
            'v_mag_var': torch.exp(v_logvar).squeeze(-1),
            'sin_mean': sin_mean.squeeze(-1),
            'sin_var': torch.exp(sin_logvar).squeeze(-1),
            'cos_mean': cos_mean.squeeze(-1),
            'cos_var': torch.exp(cos_logvar).squeeze(-1),
        }
```

## Calibration

### Expected Calibration Error (ECE)

```python
import numpy as np

def compute_ece(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
):
    """
    Compute Expected Calibration Error.
    
    For regression: bin by predicted uncertainty, check if
    coverage matches confidence level.
    """
    # Sort by uncertainty
    sorted_idx = np.argsort(uncertainties)
    
    bin_size = len(predictions) // num_bins
    ece = 0
    
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size if i < num_bins - 1 else len(predictions)
        
        bin_pred = predictions[sorted_idx[start:end]]
        bin_unc = uncertainties[sorted_idx[start:end]]
        bin_target = targets[sorted_idx[start:end]]
        
        # Compute actual error in bin
        errors = np.abs(bin_pred - bin_target)
        mean_error = errors.mean()
        
        # Expected error based on uncertainty
        expected_error = bin_unc.mean()
        
        # ECE contribution
        ece += np.abs(mean_error - expected_error) * len(bin_pred) / len(predictions)
    
    return ece
```

### Conformal Prediction

```python
class ConformalCalibrator:
    """
    Post-hoc calibration using conformal prediction.
    Ensures valid coverage regardless of base model.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (1 - alpha = coverage level)
        """
        self.alpha = alpha
        self.quantile = None
    
    def calibrate(self, val_predictions, val_targets, val_uncertainties):
        """
        Fit calibration on validation set.
        
        Args:
            val_predictions: Model predictions on validation set
            val_targets: True targets
            val_uncertainties: Predicted uncertainties (std)
        """
        # Compute conformity scores (residuals normalized by uncertainty)
        residuals = np.abs(val_predictions - val_targets)
        scores = residuals / (val_uncertainties + 1e-8)
        
        # Quantile for desired coverage
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, q_level)
    
    def predict_intervals(self, predictions, uncertainties):
        """
        Generate calibrated prediction intervals.
        """
        if self.quantile is None:
            raise ValueError("Must call calibrate() first")
        
        half_width = self.quantile * uncertainties
        
        lower = predictions - half_width
        upper = predictions + half_width
        
        return lower, upper
    
    def compute_coverage(self, predictions, targets, uncertainties):
        """Check if coverage matches nominal level."""
        lower, upper = self.predict_intervals(predictions, uncertainties)
        
        in_interval = (targets >= lower) & (targets <= upper)
        coverage = in_interval.mean()
        
        return coverage


def apply_conformal_calibration(
    model,
    val_loader,
    test_loader,
    device: str = 'cuda',
    alpha: float = 0.1,
):
    """
    Apply conformal calibration end-to-end.
    """
    # Collect validation predictions
    model.eval()
    val_preds, val_targets, val_stds = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            val_preds.append(out['v_mag_mean'].cpu().numpy())
            val_stds.append(out['v_mag_std'].cpu().numpy())
            val_targets.append(batch.y_v_mag.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_stds = np.concatenate(val_stds)
    val_targets = np.concatenate(val_targets)
    
    # Calibrate
    calibrator = ConformalCalibrator(alpha=alpha)
    calibrator.calibrate(val_preds, val_targets, val_stds)
    
    # Evaluate on test
    test_preds, test_targets, test_stds = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            
            test_preds.append(out['v_mag_mean'].cpu().numpy())
            test_stds.append(out['v_mag_std'].cpu().numpy())
            test_targets.append(batch.y_v_mag.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_stds = np.concatenate(test_stds)
    test_targets = np.concatenate(test_targets)
    
    # Compute calibrated intervals
    lower, upper = calibrator.predict_intervals(test_preds, test_stds)
    coverage = calibrator.compute_coverage(test_preds, test_targets, test_stds)
    
    print(f"Target coverage: {1-alpha:.0%}")
    print(f"Actual coverage: {coverage:.1%}")
    print(f"Average interval width: {(upper - lower).mean():.4f}")
    
    return calibrator, coverage
```

## Separating Epistemic vs Aleatoric

```python
def decompose_uncertainty(ensemble_model, data, num_mc_samples=30):
    """
    Decompose total uncertainty into epistemic and aleatoric.
    
    Requires heteroscedastic ensemble members.
    """
    means = []
    aleatoric_vars = []
    
    for member in ensemble_model.members:
        out = member(data.x, data.edge_index, data.edge_attr)
        means.append(out['v_mag_mean'])
        aleatoric_vars.append(out['v_mag_var'])
    
    means = torch.stack(means)
    aleatoric_vars = torch.stack(aleatoric_vars)
    
    # Epistemic: variance of means across ensemble
    epistemic = means.var(dim=0)
    
    # Aleatoric: mean of predicted variances
    aleatoric = aleatoric_vars.mean(dim=0)
    
    # Total
    total = epistemic + aleatoric
    
    return {
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total,
    }
```

## Visualization

```python
import matplotlib.pyplot as plt

def plot_uncertainty_calibration(
    predictions,
    targets,
    uncertainties,
    title='Uncertainty Calibration',
):
    """Plot predicted uncertainty vs actual error."""
    errors = np.abs(predictions - targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter: uncertainty vs error
    axes[0].scatter(uncertainties, errors, alpha=0.3, s=5)
    axes[0].plot([0, uncertainties.max()], [0, uncertainties.max()], 'r--', label='Ideal')
    axes[0].set_xlabel('Predicted Uncertainty (std)')
    axes[0].set_ylabel('Actual Error')
    axes[0].set_title('Uncertainty vs Error')
    axes[0].legend()
    
    # Reliability diagram (binned)
    num_bins = 10
    bin_edges = np.quantile(uncertainties, np.linspace(0, 1, num_bins + 1))
    bin_centers = []
    bin_errors = []
    
    for i in range(num_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append(uncertainties[mask].mean())
            bin_errors.append(errors[mask].mean())
    
    axes[1].bar(range(len(bin_centers)), bin_errors, alpha=0.7, label='Actual RMSE')
    axes[1].plot(range(len(bin_centers)), bin_centers, 'ro-', label='Predicted Std')
    axes[1].set_xlabel('Uncertainty Bin')
    axes[1].set_ylabel('Error / Uncertainty')
    axes[1].set_title('Reliability Diagram')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_coverage_curve(
    predictions,
    targets,
    uncertainties,
):
    """Plot coverage vs confidence level."""
    alphas = np.linspace(0.05, 0.95, 19)
    coverages = []
    
    for alpha in alphas:
        # Interval based on uncertainty quantile
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        
        coverage = ((targets >= lower) & (targets <= upper)).mean()
        coverages.append(coverage)
    
    expected = 1 - alphas
    
    plt.figure(figsize=(8, 6))
    plt.plot(expected, coverages, 'b-o', label='Actual')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal')
    plt.xlabel('Expected Coverage')
    plt.ylabel('Actual Coverage')
    plt.title('Coverage Curve')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()
```
