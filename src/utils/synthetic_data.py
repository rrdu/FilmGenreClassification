#src/utils/synthetic_data.py
'''Generates synthetic movie synopsis data for Naive Bayes'''

import random
import pandas as pd
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
#Create genre vocab
# ---------------------------------------------------------------------------
GENRES = ['action', 'comedy', 'drama','fantasy', 'horror', 'romance', 'sci-fi']

COMMON_WORDS = [
    'movie', 'story', 'character', 
    'day', 'night', 'man', 'woman', 
    'group', 'journey', 'life', 'world', 'discover'
]

GENRE_WORDS = {
    'action': [
        'fight', 'battle', 'gun', 'explosion', 'chase',
        'mission', 'soldier', 'agent', 'attack', 'rescue'
        ],

    'comedy': [
        'funny', 'joke', 'laugh', 'party', 'prank',
        'awkward', 'clown', 'mischief', 'goofy', 'hilarious'
        ],

    'drama': [
        'family', 'struggle', 'life', 'emotion','conflict',
        'relationship', 'secrets', 'past', 'childhood', 'truth'
        ],

    'fantasy': [
        'magic', 'kingdom', 'dragon', 'quest', 'spell', 
        'sword', 'legend', 'wizard', 'creature', 'prophecy'
        ],

    'horror': [
        'ghost', 'monster', 'blood', 'haunted', 'killer', 
        'scream', 'nightmare', 'curse', 'shadow', 'evil'
        ],

    'romance': [
        'love', 'heart', 'kiss', 'date', 'passion', 
        'relationship', 'wedding', 'affair', 'feelings', 'romantic'
        ],

    'sci-fi': [
        'alien', 'space', 'future', 'robot', 'technology', 
        'galaxy', 'experiment', 'planet', 'dimension', 'cyber'
        ]
}
# ---------------------------------------------------------------------------
# Create Genre Weights
# ---------------------------------------------------------------------------

GENRE_WEIGHTS = {}

def make_genre_weights():
    '''Make probability distributions of each word given a genre'''
    global GENRE_WEIGHTS

    #Get full vocab
    all_words = set(COMMON_WORDS)
    for word_list in GENRE_WORDS.values():
        all_words.update(word_list)
    
    #Make distribution for each genre
    for genre in GENRES:
        weights = {}
        for word in all_words:
            if word in GENRE_WORDS[genre]: #Check if word is genre-specific for current genre
                weights[word] = 0.6     #Genre-specific word = high weight
            elif word in COMMON_WORDS:
                weights[word] = 0.25    #Common word = neutral weight
            else:
                weights[word] = 0.05    #Other genre-specific word = low weight
        
        GENRE_WEIGHTS[genre] = weights

#Call function to initialize weights
make_genre_weights()

# ---------------------------------------------------------------------------
# Make synthetic data
# ---------------------------------------------------------------------------
def generate_synthetic_data(
        num_docs_per_genre: int=200, 
        genres: List[str] | None=None,
        min_len: int=12,
        max_len: int=36,
        seed: int=42,
):
    '''Genereate synthetic movie synopsis data with genre labels
    
    Each synopsis:
    - Has a genre
    - Has length between [min_len, max_len]
    - Draws words i.i.d from P(word | genre)
    
    Returns:
    - texts: List[str] - list of synthetic synopses
    - labels: List[str] - matching genre labels

    '''
    if genres is None:
        genres = GENRES
    
    #Set seed
    random.seed(seed)

    synopses = []
    labels = []

    for genre in genres:
        #Get vocab and probabilities for current genre
        vocab = list(GENRE_WEIGHTS[genre].keys())
        probs = list(GENRE_WEIGHTS[genre][word] for word in vocab)

        for _ in range(num_docs_per_genre):
            #Sample length
            synopsis_len = random.randint(min_len, max_len)

            #Sample words i.i.d
            words = random.choices(vocab, weights=probs, k=synopsis_len)

            #Create synopsis text
            synopsis = ' '.join(words)

            #Add synopsis and label to lists
            synopses.append(synopsis)
            labels.append(genre)
    
    return synopses, labels
# ---------------------------------------------------------------------------
# Save synthetic data to CSV
# ---------------------------------------------------------------------------
def generate_synthetic_csv(
        filepath: str,
        num_docs_per_genre: int=200,
        genres: List[str] | None=None,
        min_len: int=12,
        max_len: int=36,
        seed: int=42,
        save_path: str | None=None,
):
    '''Generate synthetic data and save to CSV file'''
    synopses, genres = generate_synthetic_data(
        num_docs_per_genre=num_docs_per_genre,
        genres=genres,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
    )

    #Create DataFrame
    df = pd.DataFrame({
        'synopsis': synopses,
        'genre': genres
    })

    #Save to CSV
    if save_path is not None:
        df.to_csv(filepath, index=False)

    return df

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    #Test
    synopses, genres = generate_synthetic_data(num_docs_per_genre=2)
    for s, g in zip(synopses, genres):
        print(f"[{s}] {g}")

    #Generate and save CSV of synthetic data
    df = generate_synthetic_csv(
        filepath='synthetic_movie_synopses.csv',
        num_docs_per_genre=200,
        save_path='synthetic_movie_synopses.csv'
    )