#src/models/naive_bayes.py
'''Naive Bayes Classifier for Film Genre Classification'''

import math

from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class NaiveBayesClassifier:
    '''
    Multinomial Naive Bayes Classifier for film genre classification from plot synopses

    - Each synopsis is a bag of words
    - Learns P(word | genre) and P(genre) from training data
    - Uses log probs 
    '''
    def __init__(self, alpha=1.0):
        #Laplace smoothing parameter
        self.alpha = alpha

        #Learned parameters
        self.class_priors_log = {}  # log P(genre)
        self.word_likelihoods_log = {}  # log P(word | genre)
        self.vocab = set()
        self.fitted = False
        
    # ---------------------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------------------
    def _tokenize(self, text) -> List[str]:
        '''Lower + split using whitespace'''
        return text.lower().split()
    
    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    def fit(self, synopses, genres):
        '''
        Fit Naive Bayes Classifier on training data (synopses + genre)
        '''
        assert len(synopses) == len(genres), "Synopses and genres should have the same length"

        #1) Count documents per genre
        class_counts = Counter(genres)
        n_docs = len(genres)

        #2) Calculate class priors P(genre)
        self.class_priors_log = {}
        for genre, count in class_counts.items():
            self.class_priors_log[genre] = math.log(count / n_docs)
        
        #Count word occurrences per genre
        # word_counts[genre][word] = count
        word_counts = defaultdict(Counter)
        for synopsis, genre in zip(synopses, genres):
            tokens = self._tokenize(synopsis)
            for token in tokens:
                word_counts[genre][token] += 1
                self.vocab.add(token)
        
        vocab_size = len(self.vocab)

        #3) For each genre, calculate P(word | genre) w/Laplace smoothing
        self.word_log_probs = {}

        for genre in class_counts.keys():
            genre_word_counts = word_counts[genre]
            total_count_for_genre = sum(genre_word_counts.values())

            #Laplace smoothing denominator
            # total_count + alpha * |V|
            denom = total_count_for_genre + self.alpha * vocab_size

            #Calculate log P(word | genre) for each word in vocab
            log_probs_for_genre = {}
            for w in self.vocab:
                count_word_in_genre = genre_word_counts[w]
                prob = (count_word_in_genre + self.alpha) / denom
                log_probs_for_genre[w] = math.log(prob)
            
            self.word_log_probs[genre] = log_probs_for_genre
        
        self.fitted = True

    # ---------------------------------------------------------------------------
    # Prediction helpers
    # ---------------------------------------------------------------------------
    def _log_posterior(self, synopses):
        '''
        Calculate log P(genre | synopsis) for each synopsis
        '''
        if not self.fitted:
            raise RuntimeError('Fit model before predicting')
        
        #Tokenize synopses
        tokens = self._tokenize(synopses)

        #For each genre, first calculate log P(genre)
        log_posteriors = {}
        for genre, prior_log in self.class_priors_log.items():
            log_posteriors[genre] = prior_log
        
        #Add sum of log P(word | genre) for each word in synopsis
        for token in tokens:
            for genre in log_posteriors:
                if token in self.word_log_probs[genre]:
                    log_posteriors[genre] += self.word_log_probs[genre][token]

        return log_posteriors
    # ---------------------------------------------------------------------------
    def predict_proba(self, synopses):
        '''
        Return normalized P(genre | synopsis) for one synopsis
        '''
        log_posteriors = self._log_posterior(synopses)

        #Log-sum-exp for normalization
        max_log = max(log_posteriors.values())
        exp_shifted = {}
        for genre, logp in log_posteriors.items():
            exp_shifted[genre] = math.exp(logp - max_log)
        sum_exp = sum(exp_shifted.values())

        norm_probs = {}
        for genre, val in exp_shifted.items():
            norm_probs[genre] = val / sum_exp
        
        return norm_probs
    # ---------------------------------------------------------------------------
    def predict(self, synopses):
        '''
        Predict genre for one synopsis
        '''
        log_posteriors = self._log_posterior(synopses)

        #Choose genre with highest log posterior
        predicted_genre = max(log_posteriors.items(), key=lambda x: x[1])[0]

        return predicted_genre
    
    # ---------------------------------------------------------------------------
    # Batch prediction + evaluation
    # ---------------------------------------------------------------------------
    def batch_predict(self, synopses):
        '''
        Predict genres for a list of synopses
        '''
        predictions = []
        for synopsis in synopses:
            predictions.append(self.predict(synopsis))

        return predictions 
    # ---------------------------------------------------------------------------
    def eval_accuracy(self, synopses, true_genres):
        '''
        Evaluate classification accuracy for list of synopses
        '''

        assert len(synopses) == len(true_genres), "Synopses and true genres should have the same length"

        preds = self.batch_predict(synopses)
        
        correct = sum(p == t for p, t in zip(preds, true_genres))
        accuracy = correct / len(true_genres)

        return accuracy
# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    #Test
    train_texts = [
        "explosion chase soldier mission",
        "kiss love heart relationship",
        "ghost blood haunted nightmare",
    ]
    train_labels = ["action", "romance", "horror"]

    model = NaiveBayesClassifier(alpha=1.0)
    model.fit(train_texts, train_labels)

    test_synopsis = "soldier in a dangerous mission with an explosion"
    pred_genre = model.predict(test_synopsis)
    print("Predicted genre:", pred_genre)

    print("Posterior distribution:", model.predict_proba(test_synopsis))