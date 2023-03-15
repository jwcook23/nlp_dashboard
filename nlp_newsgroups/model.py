import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np

from nlp_newsgroups import performance

class model():

    def __init__(self):

        self.model_topic_fname = 'model_topic.pkl'
        self.model_ner_fname = 'model_ner.pkl'

        self.load_model()


    def load_model(self):

        # TODO: build handling for NER
        with open(self.model_ner_fname, 'rb') as _fh:
            terms, summary = pickle.load(_fh)
            self.entity = {'terms': terms, 'summary': summary}

        # BUG: check if slider parameters changed
        if os.path.isfile(self.model_topic_fname):
            with open(self.model_topic_fname, 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)
        else:
            self.calculate_model()


    def calculate_model(self, input_params={}):

        params_changed = len(input_params)>0

        if not input_params:
            input_params = {key:val.value for key,val in self.model_inputs.items()}
            input_params['stop_words'] = pd.Series(list(ENGLISH_STOP_WORDS))
            self.model_params = input_params

        self.get_ngram(self.data_input)
        self.get_topics(self.data_input)

        if params_changed:
            self.default_figures(None)
        else:
            self.save_model(None)


    def save_model(self, event):

        with open(self.model_topic_fname, 'wb') as _fh:
            pickle.dump([self.model_params, self.ngram, self.topic], _fh)


    def recalculate_model(self, event):

        input_params = {key: val.value for key,val in self.model_inputs.items()}
        
        stopwords = pd.Series(input_params['stop_words'].split(','))
        stopwords = stopwords.str.lower()
        stopwords = stopwords.str.strip()
        input_params['stop_words'] = pd.concat([self.model_params['stop_words'], stopwords], ignore_index=True)

        compare = [key for key in input_params.keys() if key !='stop_words']
        change_params = [key for key in compare if input_params[key] != self.model_params[key]]
        if (input_params['stop_words'].isin(self.model_params['stop_words']) == False).any():
            change_params += ['stop_words']
        change_params = [self.model_inputs[key].title for key in change_params]
        change_params = ', '.join(change_params)

        if change_params:

            self.model_inputs['stop_words'].value = ""

            self.calculate_model(input_params)


    @performance.timing
    def get_ngram(self, text):

        self.ngram = {}

        self.ngram['vectorizer'] = CountVectorizer(
            max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
            stop_words=self.model_params['stop_words'].tolist(), ngram_range=self.model_params['ngram_range'],
            token_pattern=self.model_params['token_pattern']
        )

        self.ngram['features'] = self.ngram['vectorizer'].fit_transform(text)

        self.ngram['terms'] = pd.Series(self.ngram['vectorizer'].get_feature_names_out())

        self.ngram['summary'] = pd.DataFrame({
            'terms': self.ngram['terms'],
            'term_count': np.asarray(self.ngram['features'].sum(axis=0)).ravel(),
            'document_count': np.asarray((self.ngram['features']>0).sum(axis=0)).ravel()
        })
        self.ngram['summary'] = self.ngram['summary'].sort_values('term_count', ascending=False)


    def assign_topic(self, topic_model, features):

        distribution = topic_model.transform(features)
        distribution = pd.DataFrame.from_records(distribution, columns=self.topic['name'])
        distribution.index.name = 'Document'
        distribution = distribution.melt(var_name='Topic', value_name='Confidence', ignore_index=False)
        distribution = distribution.sort_values(by=['Topic', 'Confidence'], ascending=[True, False])
        rank = distribution.groupby('Document')['Confidence'].rank(ascending=False).astype('int64')
        distribution['Rank'] = rank

        return distribution


    @performance.timing
    def get_topics(self, text):

        self.topic = {}

        if self.model_params['topic_approach'] in ['Non-negative Matrix Factorization', 'MiniBatch Non-negative Matrix Factorization']:
            vectorizer = 'tfidf'
        else:
            vectorizer = 'tf'
        self.get_topic_vectorizer(text, vectorizer)


        if self.model_params['topic_approach'] == 'Non-negative Matrix Factorization':
            self.topic['model'] = NMF(
                n_components=self.model_params['topic_num'], random_state=1, init="nndsvda", beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1
            ).fit(self.topic['features'])

        elif self.model_params['topic_approach'] == 'MiniBatch Non-negative Matrix Factorization':
            self.topic['model'] = MiniBatchNMF(
                n_components=self.model_params['topic_num'], random_state=1, init="nndsvda", beta_loss="frobenius",
                alpha_W=0.00005, alpha_H=0.00005, l1_ratio=0.5, batch_size=128
            ).fit(self.topic['features'])

        elif self.model_params['topic_approach'] == 'Latent Dirichlet Allocation':
            self.topic['model'] = LatentDirichletAllocation(
                n_components=self.model_params['topic_num'], max_iter=5, learning_method="online",
                learning_offset=50.0, random_state=0,
            ).fit(self.topic['features'])

        self.topic['summary'] = pd.DataFrame()
        self.topic['name'] = []
        for topic_num, topic_weight in enumerate(self.topic['model'].components_):
            
            summary = pd.DataFrame({
                'Topic': [None]*self.model_params['num_features'],
                'Term': self.topic['terms'],
                'Weight': topic_weight
            })
            summary = summary.sort_values('Weight', ascending=False)
            summary['Rank'] = range(0,len(summary))
            name = f'Unnamed # {topic_num}'

            summary['Topic'] = name
            self.topic['name'] += [name]
            self.topic['summary'] = pd.concat([self.topic['summary'], summary])

        self.topic['Distribution'] = self.assign_topic(self.topic['model'], self.topic['features'])


    @performance.timing
    def get_topic_vectorizer(self, text, vectorizer):

        if vectorizer == 'tfidf':
            self.topic['vectorizer'] = TfidfVectorizer(
                max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
                stop_words=self.model_params['stop_words'].tolist(), token_pattern=self.model_params['token_pattern']
            )
        elif vectorizer == 'tf':
            self.topic['vectorizer'] = CountVectorizer(
                max_df=self.model_params['max_df'], min_df=self.model_params['min_df'], max_features=self.model_params['num_features'], 
                stop_words=self.model_params['stop_words'].tolist(), token_pattern=self.model_params['token_pattern']
            )    

        self.topic['features'] = self.topic['vectorizer'].fit_transform(text)

        self.topic['terms'] = pd.Series(self.topic['vectorizer'].get_feature_names_out())
        