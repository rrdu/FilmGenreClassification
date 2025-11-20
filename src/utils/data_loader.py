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