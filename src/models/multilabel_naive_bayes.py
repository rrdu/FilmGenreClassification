#src/models/multilabel_naive_bayes.py

'''Multilabel Naive Bayes Classifier for Film Genre Classification'''

import math
import os
import numpy as np
from models.naive_bayes import NaiveBayesClassifier 
from utils.preprocessing import clean_text 

class MultilabelNaiveBayes:
    """
    One-vs-rest multilabel classifier using your existing NaiveBayesClassifier.
    For each label ℓ, train a binary model: 'pos' (ℓ present) vs 'neg' (ℓ absent).
    """
    def __init__(self, labels, alpha=1.0, remove_stopwords=False):
        self.labels = list(labels)              # e.g. ['action', 'horror', 'romance', ...]
        self.alpha = alpha
        self.remove_stopwords = remove_stopwords
        self.models = {lab: NaiveBayesClassifier(alpha=alpha) for lab in self.labels}
    # ---------------------------------------------------------------------------
    def _clean_batch(self, texts):
        return [clean_text(t, remove_stopwords=self.remove_stopwords) for t in texts]
    # ---------------------------------------------------------------------------
    def fit(self, X_texts, Y_label_sets):
        """
        X_texts: list[str]              – raw synopses
        Y_label_sets: list[set/iter]    – e.g. [{'action','horror'}, {'romance'}, ...]
        """
        X_clean = self._clean_batch(X_texts)

        for label in self.labels:
            y_binary = ["pos" if label in y else "neg" for y in Y_label_sets]
            self.models[label].fit(X_clean, y_binary)

        return self
    # ---------------------------------------------------------------------------
    def _pos_proba_for_label(self, model, text_clean):
        """Compute P(pos | text) from the NB model's log posteriors."""
        log_post = model._log_posterior(text_clean)   # {'pos': logP, 'neg': logP}
        max_log = max(log_post.values())
        exps = {k: math.exp(v - max_log) for k, v in log_post.items()}
        Z = sum(exps.values())
        return exps["pos"] / Z
    # ---------------------------------------------------------------------------
    def predict_proba(self, X_texts):
        """
        Returns array of shape (n_samples, n_labels) with P(label=1 | x).
        """
        X_clean = self._clean_batch(X_texts)
        n = len(X_clean)
        m = len(self.labels)
        P = np.zeros((n, m), dtype=float)

        for j, label in enumerate(self.labels):
            model = self.models[label]
            for i, text_clean in enumerate(X_clean):
                P[i, j] = self._pos_proba_for_label(model, text_clean)

        return P
    # ---------------------------------------------------------------------------
    def predict(self, X_texts, threshold=0.5):
        """
        Binary predictions (n_samples, n_labels) based on a threshold.
        """
        proba = self.predict_proba(X_texts)
        return (proba >= threshold).astype(int)
    # -------------------------------------------
    # SAVE FUNCTION
    # -------------------------------------------
    def save(self, model_dir):
        """
        Save all label-specific NB models into JSON files.
        Creates a directory structure like:
            model_dir/
                config.json
                action.json
                horror.json
                romance.json
                ...
        """
        os.makedirs(model_dir, exist_ok=True)

        # Save global config
        config = {
            "labels": self.labels,
            "alpha": self.alpha,
            "remove_stopwords": self.remove_stopwords
        }
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save each binary NB model
        for label, model in self.models.items():
            model_path = os.path.join(model_dir, f"{label}.json")

            data = {
                "classes": model.classes,                     # ['pos','neg']
                "class_log_prior_": model.class_log_prior_,   # dict
                "token_counts_": model.token_counts_,         # dict
                "total_token_counts_": model.total_token_counts_,
                "vocab": list(model.vocab)
            }

            with open(model_path, "w") as f:
                json.dump(data, f)

        print(f"[OK] Saved multilabel NB model to: {model_dir}")

    # -------------------------------------------
    # LOAD FUNCTION (CLASS METHOD)
    # -------------------------------------------
    @classmethod
    def load(cls, model_dir):
        """
        Load a saved multilabel NB model directory and reconstruct
        all binary NaiveBayesClassifier models.
        """
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        labels = config["labels"]
        alpha = config["alpha"]
        remove_stopwords = config["remove_stopwords"]

        # Create empty wrapper
        obj = cls(labels=labels, alpha=alpha, remove_stopwords=remove_stopwords)

        # Load each binary classifier
        for label in labels:
            file_path = os.path.join(model_dir, f"{label}.json")
            with open(file_path, "r") as f:
                data = json.load(f)

            nb = NaiveBayesClassifier(alpha=alpha)

            nb.classes = data["classes"]
            nb.class_log_prior_ = data["class_log_prior_"]
            nb.token_counts_ = {k: dict(v) for k, v in data["token_counts_"].items()}
            nb.total_token_counts_ = data["total_token_counts_"]
            nb.vocab = set(data["vocab"])
            nb.fitted = True

            obj.models[label] = nb

        print(f"[OK] Loaded multilabel NB model from: {model_dir}")
        return obj