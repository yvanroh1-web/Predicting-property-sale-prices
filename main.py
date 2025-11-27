"""
Main Script for French Property Price Prediction Project.

Entry point for the ML pipeline:
1. Data preprocessing (preprocessing.py)
2. Feature engineering (feature_engineering.py)
3. Model training (complex_models.py)
4. Model evaluation (evaluation_complex.py)
"""

import argparse
from src.preprocessing import main as preprocess
from src.feature_engineering import main as feature_eng
from src.complex_models import main as train
from src.evaluation_complex import main as evaluate


def main():
    """
    Run the complete ML pipeline.
    """
    print("French property price prediction: \n")
    print("DVF Dataset - Best models \n")

    parser = argparse.ArgumentParser(
        description="French Property Price Prediction Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["preprocess", "features", "train", "evaluate", "all"],
        default="all",
        help="Pipeline step to run (default: all)"
    )
    args = parser.parse_args()

    if args.step in ["preprocess", "all"]:
        print("STEP 1: Data preprocessing")
        preprocess()

    if args.step in ["features", "all"]:
        print("STEP 2: feature engineering")
        feature_eng()

    if args.step in ["train", "all"]:
        print("STEP 3: Model training")
        train()

    if args.step in ["evaluate", "all"]:
        print("STEP 4: Models evaluation")
        evaluate()

    print("Pipeline complete!")


if __name__ == "__main__":
    main()