"""
Machine Learning Pipeline builder module.
Constructs scikit-learn Pipelines enforcing strict leak-free boundaries,
comparing missingness imputation strategies, and configuring two distinct model families.
"""

from typing import List, Literal, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CONFIG, CreditRiskConfig
from .features import CreditFeatureEngineer

# Feature column subsets
NUMERIC_FEATURES = [
    "revolving_utilization",
    "age",
    "debt_ratio",
    "monthly_income",
    "num_open_credit_lines",
    "num_real_estate_loans",
    "num_dependents",
    "clean_30_59_days_past_due",
    "clean_60_89_days_past_due",
    "clean_90_days_late",
    "total_past_due_count",
    "has_bureau_derogatory_flag",
    "monthly_income_is_missing",
    "num_dependents_is_missing",
    "debt_ratio_is_absolute_debt",
    # Engineered features
    "net_discretionary_income",
    "income_per_dependent",
    "severe_delinquency_ratio",
    "weighted_delinquency_score",
    "debt_to_credit_lines_ratio",
]


def build_preprocessor(
    imputation_strategy: Literal["median_simple", "median_with_indicator"] = "median_with_indicator",
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build leak-free preprocessing transformer.
    - median_simple: Fills missing values with training set median.
    - median_with_indicator: Fills with median and appends missing indicator columns.
    - scale_numeric: Applies StandardScaler (required for linear models, omitted for tree ensembles).
    """
    add_indicator = imputation_strategy == "median_with_indicator"

    steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=add_indicator))
    ]

    if scale_numeric:
        steps.append(("scaler", StandardScaler()))

    transformer = Pipeline(steps=steps)

    preprocessor = ColumnTransformer(
        transformers=[("num", transformer, NUMERIC_FEATURES)],
        remainder="drop",
    )
    return preprocessor


def create_model_pipeline(
    model_family: Literal["logistic_regression", "random_forest", "hist_gradient_boosting"] = "hist_gradient_boosting",
    imputation_strategy: Literal["median_simple", "median_with_indicator"] = "median_with_indicator",
    config: CreditRiskConfig = DEFAULT_CONFIG,
    include_engineered: bool = True,
) -> Pipeline:
    """
    Assemble end-to-end Pipeline combining Feature Engineering, Preprocessing, and Classifier.
    """
    feature_engineer = CreditFeatureEngineer(include_engineered=include_engineered)

    if model_family == "logistic_regression":
        # Logistic Regression strictly requires feature scaling
        preprocessor = build_preprocessor(
            imputation_strategy=imputation_strategy,
            scale_numeric=True,
        )
        classifier = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight=None,
            random_state=config.random_seed,
            solver="lbfgs",
        )

    elif model_family == "random_forest":
        # Tree ensembles do not require scaling
        preprocessor = build_preprocessor(
            imputation_strategy=imputation_strategy,
            scale_numeric=False,
        )
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=20,
            class_weight=None,
            random_state=config.random_seed,
            n_jobs=-1,
        )

    elif model_family == "hist_gradient_boosting":
        # Gradient boosting handles scale differences natively and excels at tabular data
        preprocessor = build_preprocessor(
            imputation_strategy=imputation_strategy,
            scale_numeric=False,
        )
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_depth=6,
            min_samples_leaf=30,
            class_weight=None,
            random_state=config.random_seed,
        )

    else:
        raise ValueError(f"Unsupported model family: {model_family}")

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", feature_engineer),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline

