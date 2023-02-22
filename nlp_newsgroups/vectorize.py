from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np

class vectorize():

    def __init__(self):

        self.max_df = 0.95
        self.min_df = 2
        self.stop_words = list(ENGLISH_STOP_WORDS)
        self.num_features = 1000

    
    def get_ngram(self, text, ngram_range):
    
        vectorizer = CountVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
            stop_words=self.stop_words, ngram_range=ngram_range
        )

        self.features_ngram = vectorizer.fit_transform(text)

        terms = pd.Series(vectorizer.get_feature_names_out())

        self.summary_ngram = pd.DataFrame({
            'terms': terms,
            'term_count': np.asarray(self.features_ngram.sum(axis=0)).ravel(),
            'document_count': np.asarray((self.features_ngram>0).sum(axis=0)).ravel()
        })
        self.summary_ngram = self.summary_ngram.sort_values('term_count', ascending=False)


    def get_tfidf(self, text):

        tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
            stop_words=self.stop_words
        )
        self.tfidf = tfidf_vectorizer.fit_transform(text)


    def get_tf(self, text):

        tf_vectorizer = CountVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
            stop_words=self.stop_words
        )
        self.tf = tf_vectorizer.fit_transform(text)