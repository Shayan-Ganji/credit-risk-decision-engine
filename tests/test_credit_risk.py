"""
Offline Unit Test Suite for Credit Risk Decision Engine.
Implements 10 rigorous tests verifying feature engineering mathematics, cost matrix evaluation,
imputation resilience, threshold optimization, and artifact serialization.
"""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.cost import (
    compute_baseline_cost,
    compute_financial_cost,
    evaluate_threshold_grid,
)
from credit_risk.features import CreditFeatureEngineer
from credit_risk.pipeline import create_model_pipeline
from credit_risk.predict import (
    load_model_artifact,
    predict_applicants,
    save_model_artifact,
)


class TestCreditRiskDecisionEngine(unittest.TestCase):
    """Test suite covering core machine learning, cost matrix, and feature logic."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)

        # Synthetic applicant DataFrame matching view_prepared_credit_features schema
        self.sample_df = pd.DataFrame(
            [
                {
                    "serious_dlq_in_2yrs": 0,
                    "revolving_utilization": 0.25,
                    "age": 40,
                    "debt_ratio": 0.30,
                    "monthly_income": 5000.0,
                    "num_open_credit_lines": 6,
                    "num_real_estate_loans": 1,
                    "num_dependents": 1.0,
                    "clean_30_59_days_past_due": 1.0,
                    "clean_60_89_days_past_due": 0.0,
                    "clean_90_days_late": 0.0,
                    "total_past_due_count": 1.0,
                    "has_bureau_derogatory_flag": 0,
                    "monthly_income_is_missing": 0,
                    "num_dependents_is_missing": 0,
                    "debt_ratio_is_absolute_debt": 0,
                    "age_group": "middle_age",
                },
                {
                    "serious_dlq_in_2yrs": 1,
                    "revolving_utilization": 1.10,
                    "age": 28,
                    "debt_ratio": 0.90,
                    "monthly_income": 3000.0,
                    "num_open_credit_lines": 3,
                    "num_real_estate_loans": 0,
                    "num_dependents": 0.0,
                    "clean_30_59_days_past_due": 2.0,
                    "clean_60_89_days_past_due": 1.0,
                    "clean_90_days_late": 2.0,
                    "total_past_due_count": 5.0,
                    "has_bureau_derogatory_flag": 0,
                    "monthly_income_is_missing": 0,
                    "num_dependents_is_missing": 0,
                    "debt_ratio_is_absolute_debt": 0,
                    "age_group": "young",
                },
                {
                    "serious_dlq_in_2yrs": 0,
                    "revolving_utilization": 0.05,
                    "age": 65,
                    "debt_ratio": 1500.0,
                    "monthly_income": np.nan,  # Missing income
                    "num_open_credit_lines": 4,
                    "num_real_estate_loans": 1,
                    "num_dependents": np.nan,  # Missing dependents
                    "clean_30_59_days_past_due": 0.0,
                    "clean_60_89_days_past_due": 0.0,
                    "clean_90_days_late": 0.0,
                    "total_past_due_count": 0.0,
                    "has_bureau_derogatory_flag": 0,
                    "monthly_income_is_missing": 1,
                    "num_dependents_is_missing": 1,
                    "debt_ratio_is_absolute_debt": 1,
                    "age_group": "senior",
                },
                {
                    "serious_dlq_in_2yrs": 1,
                    "revolving_utilization": 0.95,
                    "age": 50,
                    "debt_ratio": 0.50,
                    "monthly_income": 4000.0,
                    "num_open_credit_lines": 5,
                    "num_real_estate_loans": 1,
                    "num_dependents": 3.0,
                    "clean_30_59_days_past_due": np.nan,  # Sentinel 98 pruned to NaN
                    "clean_60_89_days_past_due": np.nan,
                    "clean_90_days_late": np.nan,
                    "total_past_due_count": 0.0,
                    "has_bureau_derogatory_flag": 1,
                    "monthly_income_is_missing": 0,
                    "num_dependents_is_missing": 0,
                    "debt_ratio_is_absolute_debt": 0,
                    "age_group": "middle_age",
                },
            ]
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # 1. Feature Engineering: Discretionary Income
    def test_feature_engineer_discretionary_income(self) -> None:
        """Test calculation of net monthly cash surplus after servicing debt."""
        engineer = CreditFeatureEngineer()
        out = engineer.transform(self.sample_df)

        # Row 0: 5000 * (1 - 0.30) = 3500.0
        self.assertAlmostEqual(out.loc[0, "net_discretionary_income"], 3500.0)
        # Row 1: 3000 * (1 - 0.90) = 300.0
        self.assertAlmostEqual(out.loc[1, "net_discretionary_income"], 300.0)
        # Row 2: Missing income -> 0.0
        self.assertAlmostEqual(out.loc[2, "net_discretionary_income"], 0.0)

    # 2. Feature Engineering: Per Capita Income
    def test_feature_engineer_per_capita_income(self) -> None:
        """Test household income division across dependents."""
        engineer = CreditFeatureEngineer()
        out = engineer.transform(self.sample_df)

        # Row 0: 5000 / (1 + 1) = 2500.0
        self.assertAlmostEqual(out.loc[0, "income_per_dependent"], 2500.0)
        # Row 1: 3000 / (0 + 1) = 3000.0
        self.assertAlmostEqual(out.loc[1, "income_per_dependent"], 3000.0)
        # Row 3: 4000 / (3 + 1) = 1000.0
        self.assertAlmostEqual(out.loc[3, "income_per_dependent"], 1000.0)

    # 3. Feature Engineering: Severe Delinquency Ratio
    def test_feature_engineer_severe_delinquency_ratio(self) -> None:
        """Test ratio of 90+ day severe delinquencies over total past-due events."""
        engineer = CreditFeatureEngineer()
        out = engineer.transform(self.sample_df)

        # Row 1: 2 late 90d / (5 total + 1e-4) ~ 0.40
        self.assertAlmostEqual(out.loc[1, "severe_delinquency_ratio"], 2.0 / 5.0, places=3)
        # Row 0: 0 late 90d -> 0.0
        self.assertAlmostEqual(out.loc[0, "severe_delinquency_ratio"], 0.0, places=3)

    # 4. Feature Engineering: Weighted Delinquency Composite
    def test_feature_engineer_weighted_score(self) -> None:
        """Test exponential weighting (1x 30d + 2x 60d + 4x 90d)."""
        engineer = CreditFeatureEngineer()
        out = engineer.transform(self.sample_df)

        # Row 1: 1.0*2 + 2.0*1 + 4.0*2 = 2 + 2 + 8 = 12.0
        self.assertAlmostEqual(out.loc[1, "weighted_delinquency_score"], 12.0)
        # Row 0: 1.0*1 = 1.0
        self.assertAlmostEqual(out.loc[0, "weighted_delinquency_score"], 1.0)

    # 5. Cost Matrix Mathematics
    def test_cost_matrix_exact_math(self) -> None:
        """Verify asymmetric cost loss: FP=1 unit, FN=12 units."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])  # TN=1, FP=1, FN=1, TP=1

        res = compute_financial_cost(y_true, y_pred, cost_fp=1.0, cost_fn=12.0)
        # Total cost = 1*1 + 1*12 = 13.0
        self.assertEqual(res["total_cost"], 13.0)
        self.assertEqual(res["fp"], 1)
        self.assertEqual(res["fn"], 1)
        # Cost per 1000 = (13 / 4) * 1000 = 3250.0
        self.assertEqual(res["expected_cost_per_1000"], 3250.0)

    # 6. Baseline Strategy Costs
    def test_baseline_cost_calculation(self) -> None:
        """Test approve_all and reject_all benchmark costs."""
        y_true = np.array([0, 0, 0, 1])  # 1 default out of 4 (25% default rate)

        # Approve all (predict 0): FN=1, FP=0 -> Cost = 12.0 -> Cost/1000 = 3000.0
        base_approve = compute_baseline_cost(y_true, strategy="approve_all", cost_fp=1.0, cost_fn=12.0)
        self.assertEqual(base_approve["total_cost"], 12.0)
        self.assertEqual(base_approve["expected_cost_per_1000"], 3000.0)

        # Reject all (predict 1): FN=0, FP=3 -> Cost = 3.0 -> Cost/1000 = 750.0
        base_reject = compute_baseline_cost(y_true, strategy="reject_all", cost_fp=1.0, cost_fn=12.0)
        self.assertEqual(base_reject["total_cost"], 3.0)
        self.assertEqual(base_reject["expected_cost_per_1000"], 750.0)

    # 7. Decision Threshold Grid Search
    def test_threshold_grid_search(self) -> None:
        """Verify that grid search selects cost-minimizing threshold reflecting asymmetric loss."""
        y_true = np.array([0, 0, 0, 0, 1])  # 20% default rate
        y_prob = np.array([0.02, 0.05, 0.06, 0.08, 0.15])

        # Under 12:1 cost, a 15% probability defaulter MUST be rejected (threshold <= 0.15)
        best_t, min_cost, _ = evaluate_threshold_grid(y_true, y_prob, cost_fp=1.0, cost_fn=12.0)
        self.assertLessEqual(best_t, 0.15)

    # 8. Pipeline Imputation & Execution Resilience
    def test_pipeline_imputation_handles_missing_values(self) -> None:
        """Verify pipeline trains and predicts on datasets containing NaNs without throwing exceptions."""
        X = self.sample_df.drop(columns=["serious_dlq_in_2yrs"])
        y = self.sample_df["serious_dlq_in_2yrs"]

        # Duplicate sample to allow small fit
        X_train = pd.concat([X] * 4, ignore_index=True)
        y_train = pd.concat([y] * 4, ignore_index=True)

        pipeline = create_model_pipeline(
            model_family="hist_gradient_boosting",
            imputation_strategy="median_with_indicator",
        )
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), len(X))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    # 9. Pipeline with Logistic Regression and Feature Scaling
    def test_pipeline_logistic_regression_scaling(self) -> None:
        """Verify Logistic Regression pipeline trains with StandardScaler."""
        X = self.sample_df.drop(columns=["serious_dlq_in_2yrs"])
        y = self.sample_df["serious_dlq_in_2yrs"]

        X_train = pd.concat([X] * 5, ignore_index=True)
        y_train = pd.concat([y] * 5, ignore_index=True)

        pipeline = create_model_pipeline(
            model_family="logistic_regression",
            imputation_strategy="median_with_indicator",
        )
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X)[:, 1]
        self.assertEqual(len(probs), len(X))

    # 10. Model Artifact Serialization & Inference
    def test_model_artifact_save_load_predict(self) -> None:
        """Test full save, load, and applicant scoring cycle."""
        X = self.sample_df.drop(columns=["serious_dlq_in_2yrs"])
        y = self.sample_df["serious_dlq_in_2yrs"]

        pipeline = create_model_pipeline("hist_gradient_boosting")
        pipeline.fit(pd.concat([X] * 4, ignore_index=True), pd.concat([y] * 4, ignore_index=True))

        artifact_path = self.test_root / "test_model.joblib"
        save_model_artifact(pipeline, optimal_threshold=0.08, metrics={"auc": 0.85}, filepath=artifact_path)
        self.assertTrue(artifact_path.exists())

        loaded = load_model_artifact(artifact_path)
        self.assertEqual(loaded["optimal_threshold"], 0.08)

        scored = predict_applicants(X, artifact=loaded)
        self.assertIn("credit_decision", scored.columns)
        self.assertIn("default_probability", scored.columns)
        self.assertEqual(len(scored), len(X))


if __name__ == "__main__":
    unittest.main()

