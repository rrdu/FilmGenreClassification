#src/utils/preprocessing.py
'''Basic text cleaning'''

import re
from typing import List
from nltk.corpus import stopwords
# ---------------------------------------------------------------------------
# Basic text preprocessing
# ---------------------------------------------------------------------------
STOP_WORDS = set(stopwords.words('english'))

def clean_text(
        text,
        remove_stopwords: bool=False
):
    '''
    Basic text preprocessing:
    - Lowercase
    - Remove punctuation
    - Remove numbers
    - Normalize whitespace
    - Remove stopwords
    '''
    #Lowercase
    text = text.lower()

    #Remove punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    #Remove numbers/symbols
    text = re.sub(r"\b\d+\b", " ", text)

    #Normalize whitespace
    tokens = text.split()

    #Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    
    return ' '.join(tokens)