#src/train_test_naive_bayes_synthetic.py
'''Train and evaluate Naive Bayes Classifier on Synthetic Data'''

from src.utils.data_loader import prepare_synthetic_data

from src.utils.synthetic_data import GENRES
from src.utils.evaluation import evaluate_model
from src.models.naive_bayes import NaiveBayesClassifier
# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
CSV_PATH = 'data/synthetic/synthetic_movies.csv'

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    #1) Load synthetic data
    print("Loading synthetic dataset...")
    splits = prepare_synthetic_data(CSV_PATH)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print(f"Train size: {len(X_train)}")
    print(f"Val size:   {len(X_val)}")
    print(f"Test size:  {len(X_test)}")
    print("-" * 50)

    # 3) Train Naive Bayes on synthetic training data
    model = NaiveBayesGenreClassifier(alpha=1.0)
    print("Training Naive Bayes on synthetic data...")
    model.fit(X_train, y_train)
    print("Training complete.")
    print("-" * 50)

    # 4) Evaluate on validation set
    print("Validation performance (synthetic):")
    evaluate_model(
        model,
        X_val,
        y_val,
        genre_names=GENRES,
        average="macro",
        verbose=True,
    )

    # 5) Evaluate on test set
    print("Test performance (synthetic):")
    evaluate_model(
        model,
        X_test,
        y_test,
        genre_names=GENRES,
        average="macro",
        verbose=True,
    )

# ---------------------------------------------------------------------------
# Run main function
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()