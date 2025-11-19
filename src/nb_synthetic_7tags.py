#src/nb_synthetic_7tags.py
'''Train and Test Naive Bayes Classifier for Film Genre Classification'''

import argparse
import os
from pathlib import Path

from src.utils.data_loader import(
    prepare_synthetic_data,
    TARGET_GENRES
)

from src.utils.evaluation import (
    evaluate_model, 
    evaluate_hit_at_k,
    misclassification_report
)

from src.models.naive_bayes import NaiveBayesClassifier

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Naive Bayes on synthetic data (7 tags)."
    )

    parser.add_argument(
        "--synthetic_csv",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "synthetic"
            / "nb_synthetic_mpst_7tags.csv"
        ),
        help="Path to the synthetic CSV (7 tags) generated from the MPST NB model.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter for Naive Bayes.",
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.7,
        help="Fraction of synthetic data to use for training.",
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
        help="Fraction of synthetic data to use for validation.",
    )

    return parser.parse_args()
# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    synthetic_csv_path = Path(args.synthetic_csv)

    #1) Load synthetic splits
    synthetic_splits = prepare_synthetic_data(
        synthetic_csv_path=str(synthetic_csv_path),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=42,
    )
    
    #2) Dataset Splits
    X_train, y_train = synthetic_splits["train"]
    X_val, y_val = synthetic_splits["val"]
    X_test, y_test = synthetic_splits["test"]

    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")
    print(f"Unique labels in train: {len(set(y_train))}")
    print("--------------------------------------------")

    #2) Train Naive Bayes classifier
    nb_model = NaiveBayesClassifier(alpha=args.alpha)
    print('Training Naive Bayes model...')
    nb_model.fit(X_train, y_train)
    print('Training complete.')

    #3) Evaluate on validation set
    print('Validating Naive Bayes model...')
    evaluate_model(
        model=nb_model,
        X=X_val,
        y=y_val,
        genre_names=TARGET_GENRES,
        average='macro',
        verbose=True
    )
    evaluate_hit_at_k(
        model=nb_model,
        X=X_val,
        y=y_val,
        k=3,
        verbose=True
    )
    

    #Save model
    model_path = "saved_models/nb_synthetic_7tags.pkl"
    os.makedirs("saved_models", exist_ok=True)
    nb_model.save(model_path)
    print(f"Saved Naive Bayes model to {model_path}")

    #4) Test set
    print('Testing Naive Bayes model...')
    evaluate_model(
        model=nb_model,
        X=X_test,
        y=y_test,
        genre_names=TARGET_GENRES,
        average='macro',
        verbose=True
    )
    evaluate_hit_at_k(
        model=nb_model,
        X=X_test,
        y=y_test,
        k=3,
        verbose=True
    )

# ---------------------------------------------------------------------------
# Run main function
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()