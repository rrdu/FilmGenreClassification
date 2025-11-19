#src/nb_mpst.py
'''Train and Test Naive Bayes Classifier for Film Genre Classification'''

import argparse
import os
from pathlib import Path

from src.utils.data_loader import(
    prepare_mpst_data_raw
)

from src.utils.evaluation import (
    evaluate_model, 
    evaluate_multilabel_hit_at_k,
    misclassification_report
)

from src.models.naive_bayes import NaiveBayesClassifier

# ---------------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Naive Bayes on the MPST dataset."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data"),
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

    #1) Load MPST splits
    mpst_splits = prepare_mpst_data_raw(
        mpst_dir=str(data_dir),
        csv_filename='mpst_full_data.csv',
        partition_json_filename='partition.json',
        min_tag_freq=10,
        shuffle=True,
        random_seed=42
    )
    
    #2) Dataset Splits
    X_train = mpst_splits["train"]["X_flat"]
    y_train = mpst_splits["train"]["y_flat"]

    #For evaluation:
    X_val_flat = mpst_splits["val"]["X_flat"]
    y_val_flat = mpst_splits["val"]["y_flat"]
    X_val_raw  = mpst_splits["val"]["X_raw"]
    Y_val_raw  = mpst_splits["val"]["Y_raw"]    # list of lists/sets of tags

    X_test_flat = mpst_splits["test"]["X_flat"]
    y_test_flat = mpst_splits["test"]["y_flat"]
    X_test_raw  = mpst_splits["test"]["X_raw"]
    Y_test_raw  = mpst_splits["test"]["Y_raw"]
    print(f'Train size: {len(X_train)}')
    print("Unique labels in train:", len(set(y_train)))
    print(f'Val size:   {len(X_val_raw)}')
    print(f'Test size:  {len(X_test_raw)}')
    print('--------------------------------------------')

    #2) Train Naive Bayes classifier
    nb_model = NaiveBayesClassifier(alpha=args.alpha)
    print('Training Naive Bayes model...')
    nb_model.fit(X_train, y_train)
    print('Training complete.')

    all_genres = sorted(set(y_train) | set(y_val_flat) | set(y_test_flat))

    #3) Evaluate on validation set
    print('Validating Naive Bayes model...')
    evaluate_model(
        model=nb_model,
        X=X_val_flat,
        y=y_val_flat,
        genre_names=all_genres,
        average='macro',
        verbose=True
    )
    evaluate_multilabel_hit_at_k(
        model=nb_model,
        X_raw=X_val_raw,
        Y_true_sets=Y_val_raw,
        k=3,
        verbose=True
    )
    '''misclassification_report(
        model=nb_model,
        X_raw=X_val_raw,
        Y_true_sets=Y_val_raw,
        k=3,
        num_examples=3
    )'''

    #Save model
    model_path = "saved_models/nb_mpst_alltags.pkl"
    os.makedirs("saved_models", exist_ok=True)
    nb_model.save(model_path)
    print(f"Saved Naive Bayes model to {model_path}")


    #4) Test set
    print('Testing Naive Bayes model...')
    evaluate_model(
        model=nb_model,
        X=X_test_flat,
        y=y_test_flat,
        genre_names=all_genres,
        average='macro',
        verbose=True
    )
    evaluate_multilabel_hit_at_k(
        model=nb_model,
        X_raw=X_test_raw,
        Y_true_sets=Y_test_raw,
        k=3,
        verbose=True
    )
    '''misclassification_report(
        model=nb_model,
        X_raw=X_test_raw,
        Y_true_sets=Y_test_raw,
        k=3,
        num_examples=3
    )'''

# ---------------------------------------------------------------------------
# Run main function
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()