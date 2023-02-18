from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np

class transform():

    def __init__(self):

        self.max_df = 0.95
        self.min_df = 2
        self.stop_words = 'english'
        self.num_features = 1000

    
    def get_bigram(self, text):
    
        vectorizer = CountVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, stop_words=self.stop_words,
            ngram_range=(2,2)
        )

        matrix = vectorizer.fit_transform(text)

        terms = np.array(vectorizer.get_feature_names_out())

        count = np.asarray(matrix.sum(axis=0)).ravel()

        bigram = pd.DataFrame({
            'terms': terms,
            'count': count
        })
        bigram = bigram.sort_values('count', ascending=False)

        np.where(terms=='ax ax')

        document, _ = matrix[:,159].nonzero()


    def get_tfidf(self, text):

        tfidf_vectorizer = TfidfVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, stop_words=self.stop_words
        )
        self.tfidf = tfidf_vectorizer.fit_transform(text)


    def get_tf(self, text):

        tf_vectorizer = CountVectorizer(
            max_df=self.max_df, min_df=self.min_df, max_features=self.num_features, stop_words=self.stop_words
        )
        self.tf = tf_vectorizer.fit_transform(text)