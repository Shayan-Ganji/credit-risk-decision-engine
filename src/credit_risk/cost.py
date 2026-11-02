"""
Cost Matrix and Decision Threshold Optimization module.
Translates statistical prediction probabilities into optimal financial loan approval decisions.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix

from .config import DEFAULT_CONFIG, CreditRiskConfig


def compute_financial_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 12.0,
) -> Dict[str, Any]:
    """
    Calculate absolute financial decision costs and confusion matrix components.

    Loss Matrix:
                     Predicted: Healthy (0)      Predicted: Default (1)
    Actual: Healthy (0)     0 (TN)                       1 (FP - Forfeited Profit)
    Actual: Default (1)    12 (FN - Default Loss)        0 (TP)
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    total_cost = (fp * cost_fp) + (fn * cost_fn)
    n_samples = len(y_true_arr)
    expected_cost_per_1000 = (total_cost / max(1, n_samples)) * 1000.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(total_cost),
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "n_samples": int(n_samples),
        "expected_cost_per_1000": float(expected_cost_per_1000),
    }


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 12.0,
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Sweep across a fine grid of decision thresholds [0.01 to 0.99] to find the cost-minimizing threshold.
    Returns:
        (optimal_threshold, min_cost_per_1000, grid_results)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 197)

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob)

    best_threshold = 0.50
    min_cost_per_1000 = float("inf")
    grid_results: List[Dict[str, Any]] = []

    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        res = compute_financial_cost(y_true_arr, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)
        cost_1k = res["expected_cost_per_1000"]

        entry = {
            "threshold": round(float(t), 4),
            "cost_per_1000": round(cost_1k, 2),
            "total_cost": res["total_cost"],
            "fp": res["fp"],
            "fn": res["fn"],
        }
        grid_results.append(entry)

        if cost_1k < min_cost_per_1000:
            min_cost_per_1000 = cost_1k
            best_threshold = float(t)

    return best_threshold, min_cost_per_1000, grid_results


def compute_baseline_cost(
    y_true: np.ndarray,
    strategy: str = "approve_all",
    cost_fp: float = 1.0,
    cost_fn: float = 12.0,
) -> Dict[str, Any]:
    """
    Compute baseline cost prior to machine learning optimization.
    - 'approve_all': Common financial institutional baseline (predict all healthy, 0).
      Cost is solely driven by False Negatives (all defaults occur).
    - 'reject_all': Extreme risk-aversion baseline (predict all default, 1).
      Cost is driven by False Positives (lost profit on all healthy applicants).
    """
    y_true_arr = np.asarray(y_true).astype(int)
    n = len(y_true_arr)

    if strategy == "approve_all":
        y_pred = np.zeros(n, dtype=int)
    elif strategy == "reject_all":
        y_pred = np.ones(n, dtype=int)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")

    return compute_financial_cost(y_true_arr, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)

