import re
import os
import pickle
from math import floor, ceil

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bokeh.models import Legend

from nlp_newsgroups.model import model
from nlp_newsgroups.default import default

class actions(model, default):

    def __init__(self):

        default.__init__(self)

        self.model_topic_fname = 'model_topic.pkl'
        self.model_ner_fname = 'model_ner.pkl'

        # TODO: build handling for NER
        with open(self.model_ner_fname, 'rb') as _fh:
            terms, summary = pickle.load(_fh)

            self.entity = {'terms': terms, 'summary': summary}


    def model_cache(self, input_params={}):

        cache_exists = os.path.isfile(self.model_topic_fname)
        params_changed = len(input_params)>0

        if not cache_exists or input_params:

            if not input_params:
                input_params = {key:val.value for key,val in self.model_inputs.items()}
                input_params['stop_words'] = list(ENGLISH_STOP_WORDS)

            # BUG: check if slider parameters changed
            model.__init__(self, **input_params)

            self.get_ngram(self.data_input)
            self.get_topics(self.data_input)

            if params_changed:
                self.default_figures(None)

            if not params_changed:
                self.save_model(None)
    
        else:
            with open(self.model_topic_fname, 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)


    def save_model(self, event):

        with open(self.model_topic_fname, 'wb') as _fh:
            pickle.dump([self.model_params, self.ngram, self.topic], _fh)


    def set_status(self, message):

        # BUG: emit status message before callbacks complete
        self.status_message.text = message


    def recalculate_model(self, event):

        input_params = {key: val.value for key,val in self.model_inputs.items()}
        
        stopwords = input_params['stop_words'].split(',')
        stopwords = [word.strip().lower() for word in stopwords]
        input_params['stop_words'] = self.model_params['stop_words']+stopwords

        change_params = [key for key,val in input_params.items() if val!= self.model_params[key]]
        change_params = [self.model_inputs[key].title for key in change_params]
        change_params = ', '.join(change_params)

        if change_params:

            self.model_inputs['stop_words'].value = ""

            self.model_cache(input_params)


    def set_samples(self, sample_title, text, important_terms):

        self.title['sample'].text = f'Example Documents:<br>{sample_title}'
        self.sample_legend.text = '<u>Legend:</u><br><strong>Imporant Terms</strong><br><s>Stop Words</s>'
        self.sample_number.title = f'Document Sample #: {len(text)} total'
        self.sample_number.high = len(text)-1
        self.sample_text = text
        self.sample_important_terms = important_terms

        self.selected_sample(None, None, self.sample_number.value)


    def selected_sample(self, attr, old, new):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]

            pattern = self.model_params['token_pattern']
            pattern = '[^'+pattern+']'
            tokens = re.sub(pattern, ' ', text)

            pattern = self.sample_important_terms
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            important_terms = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            pattern = pd.Series(self.model_params['stop_words'])
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            stopword_terms = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            text = list(text)
            for match in important_terms:
                idx_start = match.start()
                idx_end = match.end()-1
                text[idx_start] = f'<text="3"><strong>{text[idx_start]}'
                text[idx_end] = f'{text[idx_end]}</text></strong>'
            for match in stopword_terms:
                idx_start = match.start()
                idx_end = match.end()-1
                text[idx_start] = f'<s>{text[idx_start]}'
                text[idx_end] = f'{text[idx_end]}</s>'
            text = ''.join(text)

            self.sample_document.text = text


    def selected_source(self, attr, old, row_source, fig_name):

        if fig_name=='ngram':
            self.selected_ngram(row_source)
        elif fig_name=='entity_label':
            self.selected_entity_label(row_source)
        elif fig_name=='entity':
            self.selected_entity(row_source)


    def selected_ngram(self, row_source):

        if len(row_source)==0:
            return
        
        self.default_selections(event='selected_ngram', ignore='ngram')

        sample_title = self.title['ngram'].text
        important_terms = self.ngram['summary'].iloc[row_source]

        document_idx = self.ngram['features'][:, important_terms.index].nonzero()[0]

        text = self.data_input[document_idx]

        # TODO: show distribution of term importance
        self.set_samples(sample_title, text, important_terms['terms'])


    def selected_entity_label(self, row_source):

        if len(row_source)==0:
            return    
        
        selected = self.source['entity_label'].data['Terms'].iloc[row_source]
        entity = self.entity['summary'][
            self.entity['summary']['entity_label'].isin(selected)
        ]

        self.set_entity(entity)

        # raise NotImplementedError('selected_entity_label not implemented')


    def selected_entity(self, row_source):

        if len(row_source)==0:
            return
        
        raise NotImplementedError('selected_entity not implemented')

        # self.default_selections(event='selected_ngram', ignore='ngram')

        # sample_title = self.title['ngram'].text
        # important_terms = self.ngram['summary'].iloc[new]

        # document_idx = self.ngram['features'][:, important_terms.index].nonzero()[0]

        # text = self.data_input[document_idx]

        # # TODO: show distribution of term importance
        # self.set_samples(sample_title, text, important_terms['terms'])


    def get_topic_prediction(self, event):

        self.default_selections(event='get_topic_prediction', ignore=None)

        text = pd.Series([self.predict['input'].value])

        features = self.topic['vectorizer'].transform(text)

        distribution = self.assign_topic(self.topic['model'], features)

        self.predict['renderer'].data_source.data = distribution.to_dict(orient='list')

        predicted_topic = distribution.loc[distribution['Confidence']>0, 'Topic']
        
        idx = features.nonzero()[1]
        important_terms = pd.DataFrame({
            'Topic': [predicted_topic]*len(idx),
            'Term': self.topic['terms'].loc[idx],
        })
        important_terms = important_terms.explode('Topic')
        important_terms = important_terms.merge(self.topic['summary'], on=['Topic','Term'])
        important_terms = important_terms[important_terms['Weight']>0]

        topics = important_terms['Topic'].drop_duplicates()
        topics = f"Predicted Topics {', '.join(topics)}"
        self.set_topics_distribution(topics, important_terms)
        
        self.set_samples('Topic Prediction', text, important_terms['Term'])


    def rename_topic(self, event):

        # TODO: save model with new topic name
        idx = self.topic['name'].index(self.topic_number)
        new_name = self.input['topic_name'].value
        self.topic['name'][idx] = new_name
        self.predict['figure'].y_range.factors = self.topic['name']
        
        self.topic['summary']['Topic'] = self.topic['summary']['Topic'].replace(self.topic_number, new_name)
        self.topic['Distribution']['Topic'] = self.topic['Distribution']['Topic'].replace(self.topic_number, new_name)
        self.glyph['topic_term'].glyph.fill_color = self.topic_color

        self.default_figures(None)

    
    def set_topics_distribution_range(self, attr, old, new):

        start = floor(new[0])-1
        end = ceil(new[1])

        self.figure['topic_distribution'].x_range.factors = self.topic_distribution_factors[start:end+1]
        self.figure['topic_distribution'].xaxis[0].axis_label = f'Terms {start+1}-{end}'


    def set_topics_distribution(self, title_text, important_terms):

        self.topic_distribution_factors = important_terms['Term'].drop_duplicates().tolist()
        self.figure['topic_distribution'].x_range.factors = self.topic_distribution_factors

        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].right = []

        topics = important_terms['Topic'].drop_duplicates()
        legend = []
        for topic_number in topics.values:
            
            source = important_terms.loc[
                important_terms['Topic']==topic_number,
                ['Term', 'Weight']
            ].to_dict(orient='list')
            
            idx = self.topic_color.transform.factors[
                self.topic_color.transform.factors==topic_number
            ].index[0]
            topic_color = self.topic_color.transform.palette[idx]
            
            render_line = self.figure['topic_distribution'].line(
                x='Term', y='Weight', source=source, line_width=4, line_color=topic_color
            )
            render_point = self.figure['topic_distribution'].square(
                x='Term', y='Weight', source=source, size=10, fill_color=topic_color, line_color=None
            )
            legend += [(topic_number, [render_line, render_point])]

        legend = Legend(items=legend, title='Topic')
        self.figure['topic_distribution'].add_layout(legend, 'right')

        self.input['topic_distribution_range'].end = len(self.topic_distribution_factors)
        self.input['topic_distribution_range'].value = (1, min(self.input['topic_distribution_range'].end, 25))
        self.set_topics_distribution_range(None, None, self.input['topic_distribution_range'].value)


    def selected_topic(self, attr, old, new):
        
        if len(new)==0:
            return

        self.default_selections(event='selected_topic', ignore='topic_number')

        self.topic_number = self.source['topic_number'].data['Topic'].iloc[new].values[0]

        topics = self.topic['Distribution'][
            (self.topic['Distribution']['Topic']==self.topic_number) & (self.topic['Distribution']['Rank']==1)
        ]

        document_idx = topics.index
        important_terms = self.topic['summary'].loc[
            (self.topic['summary']['Topic'].isin(topics['Topic'])) & (self.topic['summary']['Weight']>0)
        ]

        text = self.data_input[document_idx]

        self.set_topics_distribution(self.topic_number, important_terms)

        self.set_samples(f'Selected Topic = {self.topic_number}', text, important_terms['Term'])


    def set_yaxis_range(self, attr, old, new, fig_name, numfactors):
        
        start = floor(new)
        end = start+numfactors
        end = min(self.input['axis_range'][fig_name].end, end)

        self.figure[fig_name].y_range.factors = self.factors[fig_name][start-1:end]
        self.figure[fig_name].yaxis[0].axis_label = f'Terms {start}-{end-1}'
