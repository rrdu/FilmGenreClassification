#src/utils/evaluation.py
'''Evaluation functions'''

from tabnanny import verbose
from typing import Dict, List, Sequence 
from sklearn.metrics import(
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# ---------------------------------------------------------------------------
# Evaluation function (label-agnostic)
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
    
    #Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    #Macro averaged precision/recall/f1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=genre_names, average=average, zero_division=0
    )

    if verbose:
        print("============================================================")
        print("Overall performance")
        print("------------------------------------------------------------")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Precision (macro): {prec_macro:.4f}")
        print(f"Recall    (macro): {rec_macro:.4f}")
        print(f"F1        (macro): {f1_macro:.4f}")
        print()
        print("Per-class metrics")
        print("------------------------------------------------------------")

        # Per-label metrics with support
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=genre_names,
            average=None,
            zero_division=0,
        )

        print(f"{'Label':15s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Support':>8s}")
        print("------------------------------------------------------------")
        for lbl, p, r, f, s in zip(genre_names, prec, rec, f1, support):
            print(f"{lbl:15s} {p:8.3f} {r:8.3f} {f:8.3f} {s:8d}")
        print("============================================================")
        print()

    return {
        "accuracy": acc,
        f"precision_{average}": prec_macro,
        f"recall_{average}": rec_macro,
        f"f1_{average}": f1_macro,
    }

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
# ---------------------------------------------------------------------------
# Show top-k predictions
# ---------------------------------------------------------------------------
def show_top_k_pred_exs(
    model,
    X,
    y,
    k:int=3,
    num_examples: int=3
):
    '''
    Show top-k predictions with probabilities for a few examples
    '''

    n = min(num_examples, len(X))
    print(f'\nTop-{k} predictions for {n} examples:\n' + '='*40)

    for i in range(n):
        synopsis = X[i]
        true_genre = y[i]
        top_k = model.predict_top_k_proba(synopsis, k=k)

        print(f'\nExample {i+1}:')
        print(f'True genre: {true_genre}')
        for rank, (genre, prob) in enumerate(top_k, start=1):
            print(f'  Rank {rank}: {genre:<10} (prob: {prob:.4f})')
# ---------------------------------------------------------------------------
# Evaluate hit @ k 
# fraction of examples where true label is in top-k predictions
# ---------------------------------------------------------------------------
def evaluate_hit_at_k(
    model,
    X: Sequence[str],
    y: Sequence[str],
    k:int=3,
    verbose: bool=True
):
    '''
    Evaluate hit @ k metric for the model

    Hit @ k = fraction of examples where true label is in top-k predictions
    '''
    #Get top-k predictions for examples
    if hasattr(model, 'batch_predict_top_k'):
        all_top_k = model.batch_predict_top_k(X, k=k)
    else:
        all_top_k = [model.predict_top_k(synopsis, k=k) for synopsis in X]

    assert len(all_top_k) == len(y), 'Number of predictions and genres should match'

    hits = 0
    for true_genre, top_k in zip(y, all_top_k):
        if true_genre in top_k:
            hits += 1
    
    hits_at_k = hits / len(y)

    if verbose:
        print(f'Top-{k} accuracy (Hit @ {k}): {hits_at_k:.4f}')
    
    return hits_at_k
# ---------------------------------------------------------------------------
# Evaluate multilabel hit @ k
# ---------------------------------------------------------------------------
def evaluate_multilabel_hit_at_k(
    model,
    X_raw: List[str],
    Y_true_sets: List[List[str]],
    k: int = 3,
    verbose: bool = True,
) -> float:
    """
    Multi-label hit@k:
    True label is a SET of tags.
    Prediction is top-k NB tags.
    Count hit if ANY true tag overlaps with predicted top-k.
    """
    assert len(X_raw) == len(Y_true_sets)

    hits = 0

    for text, true_tags in zip(X_raw, Y_true_sets):
        true_tags = set(true_tags)
        preds = model.predict_top_k(text, k=k)
        preds = set(preds)

        if len(true_tags & preds) > 0:
            hits += 1

    score = hits / len(X_raw)

    if verbose:
        print(f"Multi-label hit@{k}: {score:.4f}")

    return score

# ---------------------------------------------------------------------------
# Misclassification report
# ---------------------------------------------------------------------------
def misclassification_report(
    model,
    X_raw,
    Y_true_sets,
    k: int = 3,
    num_examples: int = 10,
    max_chars: int = 300,
):
    """
    Print misclassified examples for multi-label data.

    A sample is considered misclassified if NONE of its true tags
    appear in the model's top-k predictions.
    """
    assert len(X_raw) == len(Y_true_sets)

    mis_indices = []

    for i, (text, true_tags) in enumerate(zip(X_raw, Y_true_sets)):
        true_tags = set(true_tags)

        # get top-k predictions (as labels)
        topk = model.predict_top_k(text, k=k)
        topk_set = set(topk)

        # misclassified if there is NO overlap
        if not (true_tags & topk_set):
            mis_indices.append((i, text, true_tags, topk))

    if not mis_indices:
        print(f"\nNo misclassified examples under multi-label top-{k}.")
        return

    print(f"\nMisclassified examples (no true tag in top-{k}):")
    print("-" * 60)

    for idx, (i, text, true_tags, topk) in enumerate(mis_indices[:num_examples], start=1):
        if len(text) > max_chars:
            text_display = text[:max_chars] + "..."
        else:
            text_display = text

        print(f"\nExample {idx} (index {i})")
        print(f"True tags: {sorted(true_tags)}")
        print(f"Synopsis: {text_display}")
        print("Top predictions:")
        for rank, tag in enumerate(topk, start=1):
            # if you want probs too, use predict_topk_with_proba instead
            print(f"  {rank}. {tag}")
    print()
