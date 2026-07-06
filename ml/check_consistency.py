"""CLI utility: verify model training is stable across repeated runs.

Not a pytest test — run it directly:
    python -m ml.check_consistency data/processed.parquet --trials 3
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from ml.train import train_diabetes_model

logger = logging.getLogger(__name__)


def check_model_consistency(data_path: Path, n_trials: int = 3) -> dict:
    """Train the model n_trials times and report AUC stability"""

    logger.info(f"Testing model consistency with {n_trials} trials")

    results = []
    for i in range(n_trials):
        logger.info(f"Training trial {i+1}/{n_trials}")

        # Use different random seeds for each trial
        seed = random.randint(1, 1000)
        random.seed(seed)
        np.random.seed(seed)

        try:
            model, metrics = train_diabetes_model(
                data_path,
                model_output_path=None,  # Don't save intermediate models
                use_mlflow=False,
            )
            results.append(metrics["auc"])
        except Exception as e:
            logger.error(f"Trial {i+1} failed: {e}")
            continue

    if not results:
        raise ValueError("All trials failed")

    mean_auc = sum(results) / len(results)
    std_auc = (sum((x - mean_auc) ** 2 for x in results) / len(results)) ** 0.5

    consistency_metrics = {
        "trials": len(results),
        "auc_scores": results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "is_consistent": std_auc < 0.01,  # Less than 1% standard deviation
    }

    logger.info(f"Consistency test results: {consistency_metrics}")
    return consistency_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data/processed.parquet"),
        help="Path to the processed training parquet",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of training runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = check_model_consistency(args.data_path, n_trials=args.trials)

    print("Consistency results:")
    print(f"  Trials:     {results['trials']}")
    print(f"  AUC scores: {[f'{x:.4f}' for x in results['auc_scores']]}")
    print(f"  Mean AUC:   {results['mean_auc']:.4f}")
    print(f"  Std AUC:    {results['std_auc']:.4f}")

    if results["is_consistent"]:
        print("Model training is consistent")
        return 0
    print("Model training shows high variance")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
