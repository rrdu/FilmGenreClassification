#src/train_test_naive_bayes.py
'''Train and Test Naive Bayes Classifier for Film Genre Classification'''

import argparse
from pathlib import Path

from src.utils.data_loader import(
    prepare_mpst_data_single_genre,
    TARGET_GENRES
)

from src.utils.evaluation import evaluate_model
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
    mpst_csv_path = data_dir / 'mpst_full_data.csv'
    partition_json_path = data_dir / 'partition.json'

    print('============================================')
    print('Naive Bayes on MPST')
    print('============================================')
    print(f'Data directory: {data_dir}')
    print(f'CSV path:       {mpst_csv_path}')
    print(f'Partition JSON: {partition_json_path}')
    print(f'Alpha (Laplace smoothing): {args.alpha}')
    print(f'Drop multi-genre examples: {args.drop_multi_genre}')
    print('--------------------------------------------')

    #1) Load MPST splits
    mpst_splits = prepare_mpst_data_single_genre(
        mpst_csv_path=str(mpst_csv_path),
        partition_json_path=str(partition_json_path),
        target_genres=TARGET_GENRES,
        drop_multi_genre=args.drop_multi_genre
    )

    X_train, y_train = mpst_splits['train']
    X_val, y_val = mpst_splits['val']
    X_test, y_test = mpst_splits['test']
    print(f'Train size: {len(X_train)}')
    print("Unique labels in train:", len(set(y_train)))
    print(f'Val size:   {len(X_val)}')
    print(f'Test size:  {len(X_test)}')
    print('--------------------------------------------')

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
# ---------------------------------------------------------------------------
# Run main function
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()