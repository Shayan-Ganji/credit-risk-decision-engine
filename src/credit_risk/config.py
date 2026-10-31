"""
Configuration module for Credit Risk Decision Engine.
Centralizes database connection, cost matrix parameters, random seed, and evaluation settings.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CreditRiskConfig:
    """Immutable configuration container for Credit Risk ML pipeline."""

    # Database connection parameters
    db_url: str = "postgresql+psycopg2://postgres@127.0.0.1:5432/olist_db"
    source_view: str = "credit.view_prepared_credit_features"

    # Reproducibility
    random_seed: int = 42
    test_size: float = 0.20
    cv_splits: int = 5

    # Target variable
    target_column: str = "serious_dlq_in_2yrs"

    # Business Cost Matrix (Primary: 12:1 Asymmetric Loss)
    cost_fp: float = 1.0   # False Positive: Healthy applicant rejected (Forfeited interest profit)
    cost_fn: float = 12.0  # False Negative: Defaulting applicant approved (Defaulted principal lost)
    cost_tp: float = 0.0
    cost_tn: float = 0.0

    # Sensitivity Analysis Cost Ratio (4:1 Scenario)
    cost_fp_alt: float = 1.0
    cost_fn_alt: float = 4.0

    # Model and artifacts paths
    models_dir: Path = Path("data/curated/models")
    artifacts_dir: Path = Path("data/curated")


DEFAULT_CONFIG = CreditRiskConfig()

