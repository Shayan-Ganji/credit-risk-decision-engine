"""
Command-Line Interface (CLI) for Credit Risk Decision Engine.
Provides entry points for model evaluation, threshold optimization, and live applicant scoring.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from .config import DEFAULT_CONFIG
from .cost import compute_baseline_cost, evaluate_threshold_grid
from .data import load_credit_dataset, split_data_leak_free
from .evaluation import audit_fairness_by_demographic_group, evaluate_predictions
from .pipeline import create_model_pipeline
from .predict import load_model_artifact, predict_applicants, save_model_artifact


def configure_logging(verbose: bool = False) -> None:
    """Setup structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main CLI execution loop."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Decision Engine - Cost-Sensitive Underwriting CLI"
    )
    parser.add_argument(
        "--train-and-evaluate",
        action="store_true",
        help="Train the full credit decision pipeline and report optimal cost metrics.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to CSV file containing new applicant records to score.",
    )
    parser.add_argument(
        "--model-artifact",
        type=str,
        default="data/curated/models/credit_decision_pipeline.joblib",
        help="Path to saved model artifact bundle.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        if args.train_and_evaluate:
            print("=" * 65)
            print("CREDIT RISK DECISION ENGINE: TRAINING & EVALUATION")
            print("=" * 65)
            df = load_credit_dataset()
            X_train, X_test, y_train, y_test = split_data_leak_free(df)

            pipeline = create_model_pipeline("hist_gradient_boosting")
            pipeline.fit(X_train, y_train)

            val_prob = pipeline.predict_proba(X_test)[:, 1]
            opt_t, min_cost, _ = evaluate_threshold_grid(y_test, val_prob)
            metrics = evaluate_predictions(y_test, val_prob, threshold=opt_t)

            base = compute_baseline_cost(y_test, strategy="approve_all")
            base_cost = base["expected_cost_per_1000"]
            reduction = (base_cost - min_cost) / base_cost

            print(f"Optimal Decision Threshold:  {opt_t:.4f}")
            print(f"Baseline Cost per 1000:      {base_cost:.2f}")
            print(f"Optimized Cost per 1000:     {min_cost:.2f}")
            print(f"Financial Cost Reduction:    {reduction:.2%}")
            print(f"ROC-AUC:                     {metrics['roc_auc']:.4f}")
            print(f"Recall (Defaults Caught):    {metrics['recall']:.2%}")
            print("=" * 65)

            save_model_artifact(pipeline, opt_t, metrics, filepath=args.model_artifact)
            print(f"Artifact persisted to: {args.model_artifact}\n")
            return 0

        if args.predict:
            csv_path = Path(args.predict)
            if not csv_path.exists():
                print(f"Error: Input file not found: {csv_path}", file=sys.stderr)
                return 1

            df_new = pd.read_csv(csv_path)
            scored = predict_applicants(df_new, artifact=args.model_artifact)
            output_path = csv_path.with_name(f"{csv_path.stem}_scored.csv")
            scored.to_csv(output_path, index=False)
            print(f"Scored {len(scored)} applicants successfully. Output saved to: {output_path}")
            return 0

        parser.print_help()
        return 0

    except Exception as exc:
        logging.error(f"Execution failed: {exc}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
