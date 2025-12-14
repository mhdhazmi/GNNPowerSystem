#!/usr/bin/env python3
"""
Trivial baselines for cascade prediction.

This script evaluates simple baselines to demonstrate that GNN models
provide value beyond trivial heuristics.

Baselines:
1. Max Loading Threshold: Predict cascade if max(|P_flow|/rating) > threshold
2. XGBoost on Tabular Features: Train XGBoost on summary statistics

Usage:
    python scripts/trivial_baselines.py --grid ieee24
    python scripts/trivial_baselines.py --grid ieee118
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from datetime import datetime

# Optional: XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost baseline.")

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.powergraph import PowerGraphDataset


def extract_graph_features(data) -> dict:
    """Extract summary statistics from a single graph."""
    # Edge features: P_flow, Q_flow, X, rating
    edge_attr = data.edge_attr.numpy()

    if edge_attr.shape[1] >= 4:
        p_flow = edge_attr[:, 0]
        q_flow = edge_attr[:, 1]
        x = edge_attr[:, 2]
        rating = edge_attr[:, 3]
    else:
        # Fallback if different format
        p_flow = edge_attr[:, 0] if edge_attr.shape[1] > 0 else np.zeros(len(edge_attr))
        q_flow = edge_attr[:, 1] if edge_attr.shape[1] > 1 else np.zeros(len(edge_attr))
        rating = np.ones(len(edge_attr))  # Avoid division by zero

    # Compute loading: |S| / rating where S = sqrt(P^2 + Q^2)
    apparent_power = np.sqrt(p_flow**2 + q_flow**2)

    # Avoid division by zero
    rating_safe = np.where(rating > 1e-6, rating, 1e-6)
    loading = apparent_power / rating_safe

    # Also compute P-only loading
    p_loading = np.abs(p_flow) / rating_safe

    # Node features: P_net, S_net, V
    node_feat = data.x.numpy()
    p_net = node_feat[:, 0] if node_feat.shape[1] > 0 else np.zeros(len(node_feat))
    s_net = node_feat[:, 1] if node_feat.shape[1] > 1 else np.zeros(len(node_feat))
    v = node_feat[:, 2] if node_feat.shape[1] > 2 else np.ones(len(node_feat))

    features = {
        # Loading statistics
        'max_loading': np.max(loading),
        'mean_loading': np.mean(loading),
        'std_loading': np.std(loading),
        'p90_loading': np.percentile(loading, 90),
        'p95_loading': np.percentile(loading, 95),
        'p99_loading': np.percentile(loading, 99),
        'num_overloaded': np.sum(loading > 1.0),
        'frac_overloaded': np.mean(loading > 1.0),

        # P-only loading
        'max_p_loading': np.max(p_loading),
        'mean_p_loading': np.mean(p_loading),

        # Flow statistics
        'max_p_flow': np.max(np.abs(p_flow)),
        'mean_p_flow': np.mean(np.abs(p_flow)),
        'max_q_flow': np.max(np.abs(q_flow)),

        # Node statistics
        'max_p_net': np.max(np.abs(p_net)),
        'mean_p_net': np.mean(np.abs(p_net)),
        'min_v': np.min(v),
        'max_v': np.max(v),
        'std_v': np.std(v),

        # Graph size
        'num_nodes': len(node_feat),
        'num_edges': len(edge_attr),
    }

    return features


def threshold_baseline_proper(features_train: list, labels_train: np.ndarray,
                               features_test: list, labels_test: np.ndarray,
                               feature_name: str = 'max_loading') -> dict:
    """
    Proper threshold baseline: tune on TRAIN, evaluate on TEST.

    This avoids test leakage by only using training data for threshold selection.
    """
    train_values = np.array([f[feature_name] for f in features_train])
    test_values = np.array([f[feature_name] for f in features_test])

    # Tune threshold on TRAIN set only
    best_f1 = 0
    best_threshold = 0

    thresholds = np.linspace(train_values.min(), train_values.max(), 100)

    for thresh in thresholds:
        preds = (train_values > thresh).astype(int)
        f1 = f1_score(labels_train, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # Apply best threshold to TEST set (no leakage)
    test_preds = (test_values > best_threshold).astype(int)

    # Compute metrics on TEST set
    results = {
        'method': f'Threshold({feature_name})',
        'threshold': float(best_threshold),
        'train_f1': float(best_f1),
        'f1': float(f1_score(labels_test, test_preds, zero_division=0)),
        'precision': float(precision_score(labels_test, test_preds, zero_division=0)),
        'recall': float(recall_score(labels_test, test_preds, zero_division=0)),
        'accuracy': float(accuracy_score(labels_test, test_preds)),
    }

    # PR-AUC on test set
    precision_curve, recall_curve, _ = precision_recall_curve(labels_test, test_values)
    results['pr_auc'] = float(auc(recall_curve, precision_curve))

    return results


def threshold_baseline(features_list: list, labels: np.ndarray,
                       feature_name: str = 'max_loading') -> dict:
    """
    REMOVED: This function had test leakage (tuned threshold on test set).

    Use threshold_baseline_proper() instead, which tunes on train and evaluates on test.
    """
    raise NotImplementedError(
        "threshold_baseline() has been removed due to test leakage. "
        "Use threshold_baseline_proper(features_train, labels_train, features_test, labels_test) instead."
    )


def xgboost_baseline(features_train: list, labels_train: np.ndarray,
                     features_test: list, labels_test: np.ndarray) -> dict:
    """XGBoost baseline on tabular features."""
    if not HAS_XGBOOST:
        return {'method': 'XGBoost', 'error': 'XGBoost not installed'}

    # Convert to numpy arrays
    feature_names = list(features_train[0].keys())
    X_train = np.array([[f[k] for k in feature_names] for f in features_train])
    X_test = np.array([[f[k] for k in feature_names] for f in features_test])
    y_train = labels_train
    y_test = labels_test

    # Handle class imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    results = {
        'method': 'XGBoost',
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, preds)),
    }

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    results['pr_auc'] = float(auc(recall_curve, precision_curve))

    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    results['feature_importance'] = dict(sorted(importance.items(),
                                                 key=lambda x: x[1], reverse=True)[:5])

    return results


def main():
    parser = argparse.ArgumentParser(description='Trivial baselines for cascade prediction')
    parser.add_argument('--grid', type=str, default='ieee24', choices=['ieee24', 'ieee118'])
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"TRIVIAL BASELINES FOR CASCADE PREDICTION")
    print(f"{'='*60}")
    print(f"Grid: {args.grid}")

    # Load dataset - combine train/val/test
    print("\nLoading dataset...")
    train_dataset = PowerGraphDataset(
        root='data',
        name=args.grid,
        task='cascade',
        label_type='binary',
        split='train',
    )
    val_dataset = PowerGraphDataset(
        root='data',
        name=args.grid,
        task='cascade',
        label_type='binary',
        split='val',
    )
    test_dataset = PowerGraphDataset(
        root='data',
        name=args.grid,
        task='cascade',
        label_type='binary',
        split='test',
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Extract features and labels from train and test sets
    print("Extracting graph features...")

    features_train = []
    labels_train = []
    for data in train_dataset:
        features_train.append(extract_graph_features(data))
        labels_train.append(int(data.y.item()))
    labels_train = np.array(labels_train)

    features_test = []
    labels_test = []
    for data in test_dataset:
        features_test.append(extract_graph_features(data))
        labels_test.append(int(data.y.item()))
    labels_test = np.array(labels_test)

    # Also combine for overall stats
    all_labels = np.concatenate([labels_train, labels_test])
    print(f"Class distribution: {(all_labels==0).sum()} negative, {(all_labels==1).sum()} positive")
    print(f"Positive rate: {all_labels.mean()*100:.2f}%")
    print(f"\nUsing original splits - Train: {len(labels_train)}, Test: {len(labels_test)}")

    results = []

    # Baseline 1: Max Loading Threshold
    print("\n" + "-"*40)
    print("Baseline 1: Max Loading Threshold")
    print("-"*40)

    # Tune threshold on TRAIN set, then evaluate on TEST set (no leakage)
    thresh_result = threshold_baseline_proper(
        features_train, labels_train,
        features_test, labels_test,
        'max_loading'
    )
    results.append(thresh_result)

    print(f"Best threshold (tuned on train): {thresh_result['threshold']:.4f}")
    print(f"Test F1: {thresh_result['f1']:.4f}")
    print(f"Test Precision: {thresh_result['precision']:.4f}")
    print(f"Test Recall: {thresh_result['recall']:.4f}")
    print(f"Test PR-AUC: {thresh_result['pr_auc']:.4f}")

    # Baseline 1b: P95 Loading Threshold
    print("\n" + "-"*40)
    print("Baseline 1b: P95 Loading Threshold")
    print("-"*40)

    thresh_p95_result = threshold_baseline_proper(
        features_train, labels_train,
        features_test, labels_test,
        'p95_loading'
    )
    results.append(thresh_p95_result)

    print(f"Best threshold: {thresh_p95_result['threshold']:.4f}")
    print(f"F1: {thresh_p95_result['f1']:.4f}")
    print(f"PR-AUC: {thresh_p95_result['pr_auc']:.4f}")

    # Baseline 2: XGBoost
    print("\n" + "-"*40)
    print("Baseline 2: XGBoost on Tabular Features")
    print("-"*40)

    xgb_result = xgboost_baseline(features_train, labels_train, features_test, labels_test)
    results.append(xgb_result)

    if 'error' not in xgb_result:
        print(f"F1: {xgb_result['f1']:.4f}")
        print(f"Precision: {xgb_result['precision']:.4f}")
        print(f"Recall: {xgb_result['recall']:.4f}")
        print(f"PR-AUC: {xgb_result['pr_auc']:.4f}")
        print(f"Top features: {list(xgb_result['feature_importance'].keys())}")
    else:
        print(f"Error: {xgb_result['error']}")

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Trivial Baselines vs GNN")
    print("="*60)
    print(f"\n{'Method':<30} {'F1':>10} {'PR-AUC':>10}")
    print("-"*50)

    for r in results:
        if 'error' not in r:
            print(f"{r['method']:<30} {r['f1']:>10.4f} {r['pr_auc']:>10.4f}")

    # Add reference GNN results
    print("-"*50)
    print(f"{'GNN (Scratch, 100% labels)':<30} {'~0.99':>10} {'~0.99':>10}")
    print(f"{'GNN (SSL, 10% labels)':<30} {'~0.87':>10} {'~0.94':>10}")

    # Save results
    output_dir = Path(args.output_dir) / f"trivial_baselines_{args.grid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump({
            'grid': args.grid,
            'num_train': len(labels_train),
            'num_test': len(labels_test),
            'positive_rate': float(all_labels.mean()),
            'baselines': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
