"""
Inference and Model Serialization module.
Enables saving trained decision pipelines and scoring new applicant payloads.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd


def save_model_artifact(
    pipeline: Any,
    optimal_threshold: float,
    metrics: Dict[str, Any],
    filepath: Union[str, Path] = "data/curated/models/credit_decision_pipeline.joblib",
) -> Path:
    """Serialize the trained pipeline along with decision metadata and optimal threshold."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "optimal_threshold": float(optimal_threshold),
        "training_metrics": metrics,
    }
    joblib.dump(artifact, path)
    return path


def load_model_artifact(
    filepath: Union[str, Path] = "data/curated/models/credit_decision_pipeline.joblib",
) -> Dict[str, Any]:
    """Load serialized pipeline bundle."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)


def predict_applicants(
    df_new: pd.DataFrame,
    artifact: Union[Dict[str, Any], str, Path] = "data/curated/models/credit_decision_pipeline.joblib",
) -> pd.DataFrame:
    """
    Score incoming applicant data matching the input schema.
    Returns DataFrame with default probability and binary approval decision.
    """
    if isinstance(artifact, (str, Path)):
        bundle = load_model_artifact(artifact)
    else:
        bundle = artifact

    pipeline = bundle["pipeline"]
    threshold = bundle["optimal_threshold"]

    probabilities = pipeline.predict_proba(df_new)[:, 1]
    is_high_risk = (probabilities >= threshold).astype(int)
    decisions = np.where(is_high_risk == 1, "REJECT (High Risk)", "APPROVE (Healthy)")

    results = df_new.copy()
    results["default_probability"] = probabilities
    results["optimal_decision_threshold"] = threshold
    results["predicted_high_risk"] = is_high_risk
    results["credit_decision"] = decisions

    return results

