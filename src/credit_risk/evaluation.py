"""
Model Evaluation, Cross-Validation Stability, and Demographic Fairness Audit module.
Provides comprehensive multi-metric benchmarking, dispersion analysis, and fairness diagnostics.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import DEFAULT_CONFIG, CreditRiskConfig
from .cost import compute_financial_cost, evaluate_threshold_grid


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    cost_fp: float = 1.0,
    cost_fn: float = 12.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive suite of statistical metrics and financial decision costs.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    cost_metrics = compute_financial_cost(
        y_true_arr, y_pred, cost_fp=cost_fp, cost_fn=cost_fn
    )

    acc = accuracy_score(y_true_arr, y_pred)
    prec = precision_score(y_true_arr, y_pred, zero_division=0)
    rec = recall_score(y_true_arr, y_pred, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true_arr, y_prob_arr)
    except Exception:
        auc_roc = 0.5

    try:
        pr_auc = average_precision_score(y_true_arr, y_prob_arr)
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc_roc),
        "pr_auc": float(pr_auc),
        **cost_metrics,
    }


def evaluate_cross_validation_stability(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    cost_fp: float = 1.0,
    cost_fn: float = 12.0,
) -> Dict[str, Any]:
    """
    Perform Stratified K-Fold Cross-Validation to assess performance stability and metric dispersion.
    Evaluates both discrimination (ROC-AUC) and economic cost across distinct folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        val_prob = pipeline.predict_proba(X_val)[:, 1]

        best_t, min_cost_1k, _ = evaluate_threshold_grid(
            y_val, val_prob, cost_fp=cost_fp, cost_fn=cost_fn
        )

        metrics = evaluate_predictions(
            y_val, val_prob, threshold=best_t, cost_fp=cost_fp, cost_fn=cost_fn
        )
        metrics["fold"] = fold_idx + 1
        metrics["optimal_threshold"] = best_t
        fold_metrics.append(metrics)

    # Compute mean and standard deviation across folds
    keys = ["roc_auc", "pr_auc", "expected_cost_per_1000", "recall", "precision", "f1"]
    summary: Dict[str, Any] = {"folds": fold_metrics}
    for k in keys:
        vals = [f[k] for f in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))

    return summary


def audit_fairness_by_demographic_group(
    df_eval: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    group_col: str = "age_group",
) -> pd.DataFrame:
    """
    Audit model decisions across demographic subgroups to evaluate fairness,
    disparate impact, and approval rate differentials.
    """
    df = df_eval.copy()
    df["actual_default"] = y_true.values
    df["predicted_probability"] = y_prob
    df["predicted_high_risk"] = (y_prob >= threshold).astype(int)
    df["approved"] = (df["predicted_high_risk"] == 0).astype(int)

    records = []
    for grp, group_df in df.groupby(group_col):
        total = len(group_df)
        defaults = group_df["actual_default"].sum()
        default_rate = defaults / max(1, total)

        approvals = group_df["approved"].sum()
        approval_rate = approvals / max(1, total)

        # False positive: healthy applicant rejected
        fp = ((group_df["actual_default"] == 0) & (group_df["predicted_high_risk"] == 1)).sum()
        healthy_total = (group_df["actual_default"] == 0).sum()
        fpr = fp / max(1, healthy_total)

        # False negative: defaulter approved
        fn = ((group_df["actual_default"] == 1) & (group_df["approved"] == 1)).sum()
        fnr = fn / max(1, defaults)

        records.append({
            "subgroup": grp,
            "total_applicants": total,
            "share_of_population": total / len(df),
            "actual_default_rate": default_rate,
            "approval_rate": approval_rate,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        })

    return pd.DataFrame(records).sort_values("total_applicants", ascending=False)

