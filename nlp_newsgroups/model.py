from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
import pandas as pd
import numpy as np

from nlp_newsgroups import performance

class model():

    def __init__(self):

        self.model_params = {
            'token_pattern': '(?u)\\b\\w\\w+\\b',
            'max_df': 0.95,
            'min_df': 2,
            'stop_words': list(ENGLISH_STOP_WORDS),
            'num_features': 1000,
            'ngram_range': (1,2),
            'topic_num': 5,
            'topic_approach': 'lda',
            'nmf_init': "nndsvda",
            'nmnmf_batch_size': 128
        }


    @performance.timing
    def get_ngram(self, text):

        self.ngram = {}

        self.ngram['vectorizer'] = CountVectorizer(
            max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
            stop_words=self.model_params['stop_words'], ngram_range=self.model_params['ngram_range'],
            token_pattern=self.model_params['token_pattern']
        )

        self.ngram['features'] = self.ngram['vectorizer'].fit_transform(text)

        # TODO: is devectorized useful?
        self.ngram['devectorized'] = self.ngram['vectorizer'].inverse_transform(self.ngram['features'])

        self.ngram['terms'] = pd.Series(self.ngram['vectorizer'].get_feature_names_out())

        self.ngram['summary'] = pd.DataFrame({
            'terms': self.ngram['terms'],
            'term_count': np.asarray(self.ngram['features'].sum(axis=0)).ravel(),
            'document_count': np.asarray((self.ngram['features']>0).sum(axis=0)).ravel()
        })
        self.ngram['summary'] = self.ngram['summary'].sort_values('term_count', ascending=False)


    @performance.timing
    def get_topics(self, text):

        self.topic = {}

        if self.model_params['topic_approach'] in ['nmf', 'mbnmf']:
            vectorizer = 'tfidf'
        else:
            vectorizer = 'tf'
        self.get_topic_vectorizer(text, vectorizer)

        if self.model_params['topic_approach'] == 'nmf':
            self.topic['model'] = NMF(
                n_components=self.model_params['topic_num'], random_state=1, init=self.model_params['nmf_init'], beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1
            ).fit(self.topic['features'])

        elif self.model_params['topic_approach'] == 'mbnmf':
            self.topic['model'] = MiniBatchNMF(
                n_components=self.model_params['topic_num'], random_state=1, init=self.model_params['nmf_init'], beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=0.5, batch_size=self.model_params['nmnmf_batch_size']
            ).fit(self.topic['features'])

        elif self.model_params['topic_approach'] == 'lda':
            self.topic['model'] = LatentDirichletAllocation(
                n_components=self.model_params['topic_num'], max_iter=5, learning_method="online",
                learning_offset=50.0, random_state=0,
            ).fit(self.topic['features'])

        self.topic['summary'] = pd.DataFrame()
        for topic_num, topic_weight in enumerate(self.topic['model'].components_):
            
            summary = pd.DataFrame({
                'Topic': [topic_num]*self.model_params['num_features'],
                'Term': self.topic['terms'],
                'Weight': topic_weight
            })
            summary = summary.sort_values('Weight', ascending=False)
            summary['Rank'] = range(0,len(summary))
            summary['Topic'] = 'Topic '+summary['Topic'].astype('str')

            self.topic['summary'] = pd.concat([self.topic['summary'], summary])

        distribution = self.topic['model'].transform(self.topic['features'])
        distribution = pd.DataFrame.from_records(distribution)
        distribution.index.name = 'Document'
        distribution = distribution.melt(var_name='Topic', value_name='Confidence', ignore_index=False)
        distribution = distribution.sort_values(by=['Topic', 'Confidence'], ascending=[True, False])
        rank = distribution.groupby('Document')['Topic'].rank(method='max').astype('int64')
        distribution['Rank'] = rank
        distribution['Topic'] = 'Topic '+distribution['Topic'].astype('str')
        self.topic['Distribution'] = distribution


    @performance.timing
    def get_topic_vectorizer(self, text, vectorizer):

        if vectorizer == 'tfidf':
            self.topic['vectorizer'] = TfidfVectorizer(
                max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
                stop_words=self.model_params['stop_words'], token_pattern=self.model_params['token_pattern']
            )
        elif vectorizer == 'tf':
            self.topic['vectorizer'] = CountVectorizer(
                max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
                stop_words=self.model_params['stop_words'], token_pattern=self.model_params['token_pattern']
            )    

        self.topic['features'] = self.topic['vectorizer'].fit_transform(text)

        # TODO: is devectorized useful?
        self.topic['devectorized'] = self.topic['vectorizer'].inverse_transform(self.topic['features'])

        self.topic['terms'] = pd.Series(self.topic['vectorizer'].get_feature_names_out())
        