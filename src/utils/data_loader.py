#src/utils/data_loader.py
'''Load and preprocess MPST and Synthetic Datasets'''

import json 
import re
import os
import random
import pandas as pd
from typing import List, Dict, Tuple, Literal

from src.utils.preprocessing import clean_text
from collections import Counter

# ---------------------------------------------------------------------------
# Train/Val/Test Split
# Default is 70/15/15
# ---------------------------------------------------------------------------
def train_val_test_split(
        synopses: List[str],
        genres: List[str],
        train_frac: float=0.7,
        val_frac: float=0.15,
        seed: int=42
):
    '''
    Split (synopses, genres) into train/val/test splits

    Returns:
    splits : dict
        {
          'train': (X_train, y_train),
          'val':   (X_val,   y_val),
          'test':  (X_test,  y_test),
        }
        - X_* are lists of plot synopses (str),
        - y_* are lists of genre labels (str).
    '''
    assert len(synopses) == len(genres), 'Synopses and genres should be the same length'
    n = len(synopses)

    indices = list(range(n))

    #Set seed
    random.seed(seed)
    random.shuffle(indices)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    def subset(idx_list):
        X = [synopses[i] for i in idx_list]
        y = [genres[i] for i in idx_list]
        return X, y

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'train': subset(train_idx),
        'val':   subset(val_idx),
        'test':  subset(test_idx),
    }
# ---------------------------------------------------------------------------
# Synthetic Data Loader
# ---------------------------------------------------------------------------
def prepare_synthetic_data(
        synthetic_csv_path,
        train_frac: float=0.7,
        val_frac: float=0.15,
        seed: int=42
):
    '''
    Given an existing synthetic csv path, create train/val/test splits
    
    Default split:
    - Train = 70%
    - Val = 15%
    - Test = 15%
    '''
    #1) Load synthetic CSV
    df = pd.read_csv(synthetic_csv_path)

    #2) Turn into train/val/test splits
    synopses = df["synopsis"].astype(str).tolist()
    labels = df["genre"].astype(str).tolist()

    #Clean text
    synopses = [clean_text(syn) for syn in synopses]

    split_dict = train_val_test_split(
        synopses,
        labels,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    return split_dict