"""
Feature Engineering module for Credit Risk Modeling.
Implements scikit-learn compliant transformers with explicit business logic hypotheses
and formal temporal data leakage audits.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Formal Feature Registry documenting Business Hypothesis and Temporal Leakage Audit
FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "net_discretionary_income": {
        "hypothesis": (
            "Borrowers with higher surplus disposable income after servicing monthly debt "
            "possess a liquidity buffer to absorb unexpected financial shocks without defaulting."
        ),
        "formula": "monthly_income * (1.0 - debt_ratio) [for valid income and ratio <= 1.0]",
        "temporal_availability_audit": (
            "Available at T0. Monthly income is verified via paystubs/tax filings and debt obligations "
            "are pulled from credit bureau reports at the exact moment of loan underwriting."
        ),
    },
    "income_per_dependent": {
        "hypothesis": (
            "Gross income shared across multiple dependents dilutes actual debt repayment capacity; "
            "per-capita household income provides a superior measure of real financial solvency."
        ),
        "formula": "monthly_income / (num_dependents + 1.0)",
        "temporal_availability_audit": (
            "Available at T0. Number of household dependents is legally declared on the credit application "
            "prior to loan disbursement."
        ),
    },
    "severe_delinquency_ratio": {
        "hypothesis": (
            "Borrowers whose historical delinquencies are dominated by severe 90+ day delays exhibit "
            "chronic insolvency risk compared to individuals with minor 30-day operational oversights."
        ),
        "formula": "clean_90_days_late / (total_past_due_count + 1e-4)",
        "temporal_availability_audit": (
            "Available at T0. Represents historical 24-month past-due counts retrieved from the credit bureau "
            "prior to decisioning; does not reference any future repayment events."
        ),
    },
    "weighted_delinquency_score": {
        "hypothesis": (
            "Longer payment delays represent exponentially higher credit hazard; a weighted composite score "
            "(1x for 30-59d, 2x for 60-89d, 4x for 90d+) captures borrower degradation trajectory."
        ),
        "formula": "1.0 * clean_30_59 + 2.0 * clean_60_89 + 4.0 * clean_90",
        "temporal_availability_audit": (
            "Available at T0. All three constituent delinquency counts originate from historical credit records."
        ),
    },
    "debt_to_credit_lines_ratio": {
        "hypothesis": (
            "High debt concentrated across very few credit lines indicates restricted borrowing options "
            "and elevated liquidity vulnerability."
        ),
        "formula": "debt_ratio / (num_open_credit_lines + 1.0)",
        "temporal_availability_audit": (
            "Available at T0. Open credit lines count and debt obligations are standard credit bureau metrics."
        ),
    },
}


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless scikit-learn transformer generating domain-engineered financial features.
    Guarantees zero data leakage by computing row-independent ratios and interactions.
    """

    def __init__(self, include_engineered: bool = True) -> None:
        self.include_engineered = include_engineered

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CreditFeatureEngineer":
        """Stateless transformer: fit is a no-op."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive domain features on input DataFrame without mutating original state.
        """
        X_out = X.copy()

        if not self.include_engineered:
            return X_out

        # Safe extraction of base series
        income = X_out["monthly_income"].fillna(0.0)
        debt_ratio = X_out["debt_ratio"].fillna(0.0)
        dependents = X_out["num_dependents"].fillna(0.0)
        c30 = X_out["clean_30_59_days_past_due"].fillna(0.0)
        c60 = X_out["clean_60_89_days_past_due"].fillna(0.0)
        c90 = X_out["clean_90_days_late"].fillna(0.0)
        total_past_due = X_out["total_past_due_count"].fillna(0.0)
        open_lines = X_out["num_open_credit_lines"].fillna(0.0)

        # 1. Net Discretionary Income
        # For valid ratio <= 1.0, calculate net surplus. For ratio > 1.0, capture negative balance.
        # When income is missing (0.0), net is 0.0.
        X_out["net_discretionary_income"] = np.where(
            debt_ratio <= 1.0,
            income * (1.0 - debt_ratio),
            income - (income * debt_ratio),
        )

        # 2. Income per Dependent (Household Per-Capita Income)
        X_out["income_per_dependent"] = income / (dependents + 1.0)

        # 3. Severe Delinquency Ratio (Proportion of chronic defaults)
        X_out["severe_delinquency_ratio"] = c90 / (total_past_due + 1e-4)

        # 4. Weighted Delinquency Composite Score
        X_out["weighted_delinquency_score"] = 1.0 * c30 + 2.0 * c60 + 4.0 * c90

        # 5. Debt to Credit Lines Ratio (Debt concentration)
        X_out["debt_to_credit_lines_ratio"] = debt_ratio / (open_lines + 1.0)

        return X_out

