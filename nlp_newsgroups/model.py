from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
import pandas as pd
import numpy as np

from nlp_newsgroups import performance

class model():

    def __init__(self):

        self.max_df = 0.95
        self.min_df = 2
        self.stop_words = list(ENGLISH_STOP_WORDS)
        self.num_features = 1000

        self.topic_num = 5
        self.nmf_init = "nndsvda"
        self.nmnmf_batch_size = 128


    @performance.timing
    def get_ngram(self, text, ngram_range):

        self.ngram_vectorizer = CountVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
            stop_words=self.stop_words, ngram_range=ngram_range
        )

        self.ngram_features = self.ngram_vectorizer.fit_transform(text)

        self.ngram_terms = pd.Series(self.ngram_vectorizer.get_feature_names_out())

        self.summary_ngram = pd.DataFrame({
            'terms': self.ngram_terms,
            'term_count': np.asarray(self.ngram_features.sum(axis=0)).ravel(),
            'document_count': np.asarray((self.ngram_features>0).sum(axis=0)).ravel()
        })
        self.summary_ngram = self.summary_ngram.sort_values('term_count', ascending=False)


    @performance.timing
    def get_topics(self, text, approach:Literal['nmf', 'mbnmf', 'lda'] = 'lda'):

        self.topic_approach = approach

        if self.topic_approach in ['nmf', 'mbnmf']:
            vectorizer = 'tfidf'
        else:
            vectorizer = 'tf'
        self.get_topic_vectorizer(text, vectorizer)

        if self.topic_approach == 'nmf':
            self.topic_model = NMF(
                n_components=self.topic_num, random_state=1, init=self.nmf_init, beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1
            ).fit(self.topic_features)

        elif self.topic_approach == 'mbnmf':
            self.topic_model = MiniBatchNMF(
                n_components=self.topic_num, random_state=1, init=self.nmf_init, beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=0.5, batch_size=self.nmnmf_batch_size
            ).fit(self.topic_features)

        elif self.topic_approach == 'lda':
            self.topic_model = LatentDirichletAllocation(
                n_components=self.topic_num, max_iter=5, learning_method="online",
                learning_offset=50.0, random_state=0,
            ).fit(self.topic_features)

        self.summary_topic = pd.DataFrame()
        for topic_num, topic_weight in enumerate(self.topic_model.components_):
            
            summary = pd.DataFrame({
                'Topic': [topic_num]*self.num_features,
                'Term': self.topic_terms,
                'Weight': topic_weight
            })
            summary = summary.sort_values('Weight', ascending=False)
            summary['Rank'] = range(0,len(summary))

            self.summary_topic = pd.concat([self.summary_topic, summary])


    @performance.timing
    def get_topic_vectorizer(self, text, vectorizer):

        if vectorizer == 'tfidf':
            self.topic_vectorizer = TfidfVectorizer(
                max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
                stop_words=self.stop_words
            )
        elif vectorizer == 'tf':
            self.topic_vectorizer = CountVectorizer(
                max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, 
                stop_words=self.stop_words
            )    

        self.topic_features = self.topic_vectorizer.fit_transform(text)

        self.topic_terms = pd.Series(self.topic_vectorizer.get_feature_names_out())
