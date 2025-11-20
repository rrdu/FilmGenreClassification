#src/utils/nb_synthetic_data.py
'''Generate synthetic movie synopses from trained Naive Bayes model'''

import os
import math
import random
import pandas as pd

from typing import Dict, List, Tuple 
from models.naive_bayes import NaiveBayesClassifier

#---------------------------------------------------------------------------
#Turn log probs into normal probs
#---------------------------------------------------------------------------
def _normalize_log_probs(log_probs):
    '''
    Convert dict {item:log_p} into (items, probs)
    - sum(probs) = 1
    '''
    items = list(log_probs.keys())
    log_values = [log_probs[item] for item in items]

    #Subtract max log value for numerical stability
    max_log = max(log_values)
    exp_values = [math.exp(lv - max_log) for lv in log_values]
    total = sum(exp_values)
    probs = [ev / total for ev in exp_values]

    return items, probs

#---------------------------------------------------------------------------
# Generate synthetic synopses
#---------------------------------------------------------------------------
def generate_synthetic_from_nb(
    nb_model: NaiveBayesClassifier,
    samples_per_genre: int = 500,
    min_len: int = 40,
    max_len: int = 120,
    balanced_genres: bool = True,
    random_seed: int = 42
):
    '''
    Generate synthetic movie synopses using a trained Naive Bayes model.

    Args:
        nb_model: A trained NaiveBayesClassifier instance.
        samples_per_genre: Number of synthetic samples to generate per genre.
        min_len: Minimum length of generated synopsis.
        max_len: Maximum length of generated synopsis.
        balanced_genres: Whether to generate equal samples per genre.
        random_seed: Random seed for reproducibility.

    Returns:
        A list of synthetic movie synopses with their corresponding genres.
    '''
    rng = random.Random(random_seed)

    #1) Genres + class priors
    genre_log_priors = nb_model.class_priors_log #{label: log P(genre)}
    genres = sorted(genre_log_priors.keys())

    _, class_priors = _normalize_log_probs(genre_log_priors)

    #2) Per-genre word distributions
    #nb_model.word_log_probs[genre] = {word: log P(word | genre)}
    genre_to_vocab_probs = {}
    for genre in genres:
        log_probs = nb_model.word_log_probs[genre]
        vocab, probs = _normalize_log_probs(log_probs)
        genre_to_vocab_probs[genre] = (vocab, probs)
    
    synopses = []
    generated_genres = []

    #3) Generate synthetic samples
    if balanced_genres: #Same num per genre
        for genre in genres:
            vocab, probs = genre_to_vocab_probs[genre]
            for _ in range(samples_per_genre):
                length = rng.randint(min_len, max_len)
                words = rng.choices(vocab, weights=probs, k=length)
                synopsis = ' '.join(words)
                synopses.append(synopsis)
                generated_genres.append(genre)
    
    else: #Sample genres according to class priors
        total_samples = samples_per_genre
        for _ in range(total_samples):
            #Pick genre from priors
            genre = rng.choices(genres, weights=class_priors, k=1)[0]
            vocab, probs = genre_to_vocab_probs[genre]
            length = rng.randint(min_len, max_len)
            words = rng.choices(vocab, weights=probs, k=length)
            synopsis = ' '.join(words)
            synopses.append(synopsis)
            generated_genres.append(genre)
    
    df = pd.DataFrame({'synopsis': synopses, 'genre': generated_genres})
    return df

#---------------------------------------------------------------------------
# Save to CSV
#---------------------------------------------------------------------------
def generate_synthetic_from_nb_to_csv(
    model_path: str,
    output_csv_path: str,
    samples_per_genre: int = 1000,
    min_len: int = 40,
    max_len: int = 120,
    balanced_genres: bool = True,
    random_seed: int = 42
):
    '''Load trained NB model and generate synthetic data to CSV'''

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f'Loading Naive Bayes model from: {model_path}')
    nb_model = NaiveBayesClassifier.load(model_path)

    print('Generating synthetic data from Naive Bayes model...')
    df = generate_synthetic_from_nb(
        nb_model=nb_model,
        samples_per_genre=samples_per_genre,
        min_len=min_len,
        max_len=max_len,
        balanced_genres=balanced_genres,
        random_seed=random_seed
    )
    print(f'Generated {len(df)} synthetic data samples across {df['genre'].nunique()} genres.')

    df.to_csv(output_csv_path, index=False)
    print(f'Synthetic data saved to: {output_csv_path}')

#---------------------------------------------------------------------------
#Main function
#---------------------------------------------------------------------------
if __name__ == '__main__':
    generate_synthetic_from_nb_to_csv(
        model_path='saved_models/nb_mpst_7tags.pkl',
        output_csv_path='data/synthetic/nb_synthetic_mpst_7tags.csv',
        samples_per_genre=500,
        min_len=40,
        max_len=120,
        balanced_genres=True,
        random_seed=42
    )