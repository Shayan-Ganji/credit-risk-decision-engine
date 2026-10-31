"""
Credit Risk Decision Engine
===========================
Cost-sensitive Machine Learning Package for Credit Scoring & Default Prevention.
"""

from .config import DEFAULT_CONFIG, CreditRiskConfig
from .cost import compute_baseline_cost, compute_financial_cost, evaluate_threshold_grid
from .data import load_credit_dataset, split_data_leak_free
from .evaluation import (
    audit_fairness_by_demographic_group,
    evaluate_cross_validation_stability,
    evaluate_predictions,
)
from .features import CreditFeatureEngineer, FEATURE_DEFINITIONS
from .pipeline import create_model_pipeline
from .predict import load_model_artifact, predict_applicants, save_model_artifact

__version__ = "0.1.0"
__author__ = "Shayan Ganji"

__all__ = [
    "DEFAULT_CONFIG",
    "CreditRiskConfig",
    "load_credit_dataset",
    "split_data_leak_free",
    "CreditFeatureEngineer",
    "FEATURE_DEFINITIONS",
    "create_model_pipeline",
    "compute_financial_cost",
    "compute_baseline_cost",
    "evaluate_threshold_grid",
    "evaluate_predictions",
    "evaluate_cross_validation_stability",
    "audit_fairness_by_demographic_group",
    "save_model_artifact",
    "load_model_artifact",
    "predict_applicants",
]

