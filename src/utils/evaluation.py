#src/utils/evaluation.py
'''Evaluation functions'''

from typing import Dict, List, Sequence 
from sklearn.metrics import(
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    genre_names: List[str] | None=None,
    average: str='macro',
    verbose: bool=True
):
    '''
    Evaluate classification predictions with:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    '''

    if genre_names is None:
        genre_names= sorted(set(y_true))
    
    accuracy = accuracy_score(y_true, y_pred)

    #Metrics for each class
    precision_class, recall_class, f1_class, support_class = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=genre_names,
        average=None,
        zero_division=0
    )

    #Overall 
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0
    )

    metrics = {
        'accuracy': accuracy,
        f'precision_{average}': precision_avg,
        f'recall_{average}': recall_avg,
        f'f1_{average}': f1_avg,
        'per_class':{}
    }

    for genre, p, r, f, s in zip(genre_names, precision_class, recall_class, f1_class, support_class):
        metrics['per_class'][genre] = {
            'precision': p,
            'recall': r,
            'f1': f,
            'support': int(s)
        }
    
    if verbose:
        print("=" * 60)
        print("Overall performance")
        print("-" * 60)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Precision ({average}): {precision_avg:.4f}")
        print(f"Recall    ({average}): {recall_avg:.4f}")
        print(f"F1        ({average}): {f1_avg:.4f}")
        print()

        print("Per-class metrics")
        print("-" * 60)
        header = f"{'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}"
        print(header)
        print("-" * len(header))

        for lbl in genre_names:
            m = metrics["per_class"][lbl]
            print(
                f"{lbl:<12} "
                f"{m['precision']:>8.3f} "
                f"{m['recall']:>8.3f} "
                f"{m['f1']:>8.3f} "
                f"{m['support']:>8d}"
            )

        print("=" * 60)
        print()
    
    return metrics

# ---------------------------------------------------------------------------
# Evaluate the model
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    X: Sequence[str],
    y: Sequence[str],
    genre_names: List[str] | None=None,
    average: str='macro',
    verbose: bool=True
):
    '''
    For a model with a batch_predict method,
    calculate predictions and call evaluate_predictions

    Needs either:
    - model.batch_predict(synopses)
    - model.predict(synopsis)
    '''
    if hasattr(model, 'batch_predict'):
        y_pred = model.batch_predict(X)
    else:
        #Call predictions one-by-one
        y_pred = [model.predict(synopsis) for synopsis in X]

    return evaluate_predictions(
        y_true=y,
        y_pred=y_pred,
        genre_names=genre_names,
        average=average,
        verbose=verbose
    )