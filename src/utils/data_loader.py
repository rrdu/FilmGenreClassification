#src/utils/data_loader.py
'''Load and preprocess MPST and Synthetic Datasets'''

import json 
import random
import pandas as pd
from typing import List, Dict, Tuple

from .synthetic_data import(
    GENRES as SYNTH_GENRES,
    generate_synthetic_data
)

from src.utils.preprocessing import clean_text

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------
TARGET_GENRES = ['action', 'comedy', 'drama','fantasy', 'horror', 'romance', 'sci-fi']

#Map MPST tags to target genres
TAG_TO_GENRE = {
    'action': 'action',
    'comedy': 'comedy',
    'dramatic': 'drama',
    'melodrama': 'drama',
    'fantasy': 'fantasy',
    'horror': 'horror',
    'romantic': 'romance',
    'sci-fi': 'sci-fi'
}
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
# MPST Tags -> Target Genres
# ---------------------------------------------------------------------------
def _map_tags_to_genres(tags):
    '''
    Given MPST tags, map to set of target genres
    '''
    mapped = set()
    for t in str(tags).split(', '):
        if t in TAG_TO_GENRE:
            mapped.add(TAG_TO_GENRE[t])
    
    return mapped
# ---------------------------------------------------------------------------
def prepare_mpst_data_single_genre(
        mpst_csv_path,
        partition_json_path,
        target_genres,
        drop_multi_genre=True
):
    '''
    - Load MPST dataset
    - Map MPST tags to target genres
    - Only keep rows with at least one target genre
    - Use partition JSON to split into train/val/test
    - Return splits as (synopses, genres) lists

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
    if target_genres is not None:
        target_genres = TARGET_GENRES
    
    #1) Load MPST CSV
    df = pd.read_csv(mpst_csv_path)

    #2) Map tags to target genres
    df['mapped_genres'] = df['tags'].apply(_map_tags_to_genres)
    df['n_genres'] = df['mapped_genres'].apply(len)

    #3) Keep rows with at least 1 target genre
    df = df[df['n_genres'] > 0].copy()

    #4) If drop_multi_genre, keep only rows with single genre
    if drop_multi_genre:
        df = df[df['n_genres'] == 1].copy()
    
    #Single genre
    df["single_genre"] = df["mapped_genres"].apply(lambda s: next(iter(s)))
    
    #5) Load partition JSON to make splits
    with open(partition_json_path, 'r') as f:
        partitions = json.load(f)
    train_ids = set(partitions['train'])
    val_ids = set(partitions['val'])
    test_ids = set(partitions['test'])

    #Make sure imdb_id is string
    df['imdb_id'] = df['imdb_id'].astype(str)

    #Clean synopses
    df['plot_synopsis'] = df['plot_synopsis'].apply(clean_text)

    train_df = df[df['imdb_id'].isin(train_ids)]
    val_df = df[df['imdb_id'].isin(val_ids)]
    test_df = df[df['imdb_id'].isin(test_ids)]

    #Make (synopses, genres) lists for each split
    def to_xy(sub_df):
        synopses = sub_df['plot_synopsis'].astype(str).tolist()
        genres = sub_df['single_genre'].astype(str).tolist()
        return synopses, genres

    X_train, y_train = to_xy(train_df)
    X_val,   y_val   = to_xy(val_df)
    X_test,  y_test  = to_xy(test_df)

    return {
        'train': (X_train, y_train),
        'val':   (X_val,   y_val),
        'test':  (X_test,  y_test),
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

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    #Synthetic sanity check
    #synth_splits = prepare_synthetic_data(num_docs_per_genre=5)
    #X_train, y_train = synth_splits["train"]
    #print("Synthetic train size:", len(X_train))
    #print("Example synthetic sample:")
    #print("  text:", X_train[0])
    #print("  label:", y_train[0])
    print()