"""
Data ingestion module for Stage 3 Credit Risk Modeling.
Strictly reads curated dataset from PostgreSQL via SQLAlchemy. Direct CSV reads are prohibited.
"""

import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

from .config import DEFAULT_CONFIG, CreditRiskConfig

logger = logging.getLogger(__name__)


def load_credit_dataset(config: CreditRiskConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Load applicant records directly from PostgreSQL In-Database preparation View.
    Ensures complete decoupling from raw CSV files.
    """
    logger.info(f"Connecting to database via SQLAlchemy: {config.db_url}")
    engine = create_engine(config.db_url)

    query = f"SELECT * FROM {config.source_view};"
    logger.info(f"Executing query: {query}")

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns from database.")
    return df


def split_data_leak_free(
    df: pd.DataFrame, config: CreditRiskConfig = DEFAULT_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute stratified train/test split BEFORE any statistical transformation or imputation.
    Strictly guarantees that no evaluation set distribution leaks into training estimators.
    """
    target_col = config.target_column
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in loaded dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    logger.info(
        f"Split complete: Train={len(X_train):,} ({y_train.mean():.2%} default rate), "
        f"Test={len(X_test):,} ({y_test.mean():.2%} default rate)"
    )

    return X_train, X_test, y_train, y_test

