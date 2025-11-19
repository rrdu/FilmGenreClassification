#src/nb_mpst_7tags.py
'''Train and Test Naive Bayes Classifier for Film Genre Classification
MPST Dataset mapped to 7 tags only'''


import argparse
import os
from pathlib import Path

from src.utils.data_loader import (
    prepare_mpst_data_single_genre,
    TARGET_GENRES
    # ['action','comedy','drama','fantasy','horror','romance','sci-fi']
)
from src.utils.evaluation import (
    evaluate_model,
    evaluate_hit_at_k,
)
from src.models.naive_bayes import NaiveBayesClassifier

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Naive Bayes on MPST with 7 target genres."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "mpst"),
        help="Directory containing mpst_full_data.csv and partition.json",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter for Naive Bayes.",
    )
    parser.add_argument(
        "--drop_multi_genre",
        action="store_true",
        help="If set, drop movies that map to more than one of the 7 genres.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "mpst_full_data.csv"
    partition_json_path = data_dir / "partition.json"

    print("============================================")
    print("Naive Bayes on MPST (7 target genres)")
    print("============================================")
    print(f"MPST directory: {data_dir}")
    print(f"CSV path:       {csv_path}")
    print(f"Partition JSON: {partition_json_path}")
    print(f"Alpha:          {args.alpha}")
    print(f"Drop multi-genre examples: {args.drop_multi_genre}")
    print("--------------------------------------------")

    # 1) Load MPST splits mapped to 7 genres (single-label)
    splits = prepare_mpst_data_single_genre(
        mpst_csv_path=str(csv_path),
        partition_json_path=str(partition_json_path),
        target_genres=TARGET_GENRES,
        drop_multi_genre=args.drop_multi_genre,
    )

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")
    print("Unique labels in train:", sorted(set(y_train)))
    print("--------------------------------------------")

    # 2) Train Naive Bayes classifier
    nb_model = NaiveBayesClassifier(alpha=args.alpha)
    print("Training Naive Bayes model...")
    nb_model.fit(X_train, y_train)
    print("Training complete.")

    # Use the 7 target genres as label order for reporting
    genre_names = TARGET_GENRES

    # 3) Evaluate on validation set
    print("Validating Naive Bayes model...")
    evaluate_model(
        model=nb_model,
        X=X_val,
        y=y_val,
        genre_names=genre_names,
        average="macro",
        verbose=True,
    )
    evaluate_hit_at_k(
        model=nb_model,
        X=X_val,
        y=y_val,
        k=3,
        verbose=True,
    )

    # 4) Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/nb_mpst_7tags.pkl"
    nb_model.save(model_path)
    print(f"Saved Naive Bayes model to {model_path}")

    # 5) Test set evaluation
    print("Testing Naive Bayes model...")
    evaluate_model(
        model=nb_model,
        X=X_test,
        y=y_test,
        genre_names=genre_names,
        average="macro",
        verbose=True,
    )
    evaluate_hit_at_k(
        model=nb_model,
        X=X_test,
        y=y_test,
        k=3,
        verbose=True,
    )

# ---------------------------------------------------------------------------
# Run main function
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()