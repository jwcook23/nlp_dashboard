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

        # TODO: differentiate these tags
        self.html = {
            'stopwords': ('<s>', '</s>'),
            'selected_terms': ('<strong><font color="red">', '</font></strong>'),
            'topic_terms': ('<strong>', '</strong>'),
            'labeled_entity': ('<strong><font color="blue">', '</font></strong>')
        }


    def model_cache(self, input_params={}):

        cache_exists = os.path.isfile(self.model_topic_fname)
        params_changed = len(input_params)>0

        if not cache_exists or input_params:

            if not input_params:
                input_params = {key:val.value for key,val in self.model_inputs.items()}
                input_params['stop_words'] = pd.Series(list(ENGLISH_STOP_WORDS))

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

            self.model_cache(input_params)


    def set_samples(self, sample_title, text, selected_terms, topic_terms, labeled_entity):

        self.title['sample'].text = f'Example Documents:<br>{sample_title}'
        self.sample_legend.text = f'''
        <u>Legend:</u><br>
        {self.html['selected_terms'][0]}Selected Term{self.html['selected_terms'][1]}<br>
        {self.html['stopwords'][0]}Stop Words{self.html['stopwords'][1]}<br>
        {self.html['topic_terms'][0]}Topic Terms{self.html['topic_terms'][1]}<br>
        {self.html['labeled_entity'][0]}Labeled Entity{self.html['labeled_entity'][1]}
        '''
        self.sample_number.title = f'Document Sample #: {len(text)} total'
        self.sample_number.high = len(text)-1
        self.sample_text = text
        self.sample_selected_terms = selected_terms
        self.sample_topic_terms = topic_terms
        self.sample_entity_labels = labeled_entity

        self.selected_sample(None, None, self.sample_number.value)


    def find_text(self, tokens, pattern):

        if pattern is None:
            return None

        pattern = pattern.str.replace(' +', ' ', regex=True)
        pattern = pattern.str.strip()
        pattern = pattern.apply(lambda x: re.escape(x))
        pattern = pattern.str.replace(r' ', r's+', regex=True)
        pattern = '|'.join(r'\b'+pattern+r'\b')
        indices = re.finditer(pattern, tokens, flags=re.IGNORECASE)

        return indices
    

    def apply_html(self, text, terms, formatter):
    
        if terms is None:
            return text

        for match in terms:
            idx_start = match.start()
            idx_end = match.end()-1
            text[idx_start] = f"{self.html[formatter][0]}{text[idx_start]}"
            text[idx_end] = f"{text[idx_end]}{self.html[formatter][1]}"

        return text


    def selected_sample(self, attr, old, new):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]

            pattern = self.model_params['token_pattern']
            pattern = '[^'+pattern+']'
            tokens = re.sub(pattern, ' ', text)

            # BUG: how to handle overlapping patterns?
            stopword_terms = self.find_text(tokens, self.model_params['stop_words'])
            selected_terms = self.find_text(tokens, self.sample_selected_terms)
            # TODO: highlight name of topic
            topic_terms = self.find_text(tokens, self.sample_topic_terms)
            # TODO: highlight label of entity
            if self.sample_entity_labels is None:
                labeled_entity = None
            else:
                labeled_entity = self.find_text(tokens, self.sample_entity_labels['entity_text'])

            text = list(text)

            text = self.apply_html(text, stopword_terms, 'stopwords')
            text = self.apply_html(text, selected_terms, 'selected_terms')
            text = self.apply_html(text, topic_terms, 'topic_terms')
            text = self.apply_html(text, labeled_entity, 'labeled_entity')

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
        
        self.default_selections(event='selected_ngram', ignore=pd.Series(['ngram']))

        sample_title = self.title['ngram'].text
        selected_terms = self.ngram['summary'].iloc[row_source]['terms']

        document_idx = self.ngram['features'][:, selected_terms.index].nonzero()[0]

        text = self.data_input[document_idx]

        # TODO: show distribution of term importance
        topic_terms = self.topic['terms']
        labeled_entity = self.entity['terms'].loc[document_idx,['entity_text','entity_label']]
        self.set_samples(sample_title, text, selected_terms, topic_terms, labeled_entity)


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
        
        selected_terms = self.source['entity'].data['Terms'].iloc[row_source]
        document_idx = self.entity['terms'].loc[
            self.entity['terms']['entity_clean'].isin(selected_terms), 'document'
        ].drop_duplicates()
        labeled_entity = self.entity['terms'][
            self.entity['terms']['document'].isin(document_idx)
        ]

        self.default_selections(event='selected_entity', ignore=pd.Series(['entity','entity_label']))

        sample_title = self.title['entity'].text

        text = self.data_input[document_idx]

        # TODO: show distribution of term importance
        topic_terms = self.topic['terms']
        self.set_samples(sample_title, text, selected_terms, topic_terms, labeled_entity)


    def get_topic_prediction(self, event):

        self.default_selections(event='get_topic_prediction', ignore=None)

        text = pd.Series([self.predict['input'].value])

        features = self.topic['vectorizer'].transform(text)

        distribution = self.assign_topic(self.topic['model'], features)

        self.predict['renderer'].data_source.data = distribution.to_dict(orient='list')

        predicted_topic = distribution.loc[distribution['Confidence']>0, 'Topic']
        
        idx = features.nonzero()[1]
        topic_terms = pd.DataFrame({
            'Topic': [predicted_topic]*len(idx),
            'Term': self.topic['terms'].loc[idx],
        })
        topic_terms = topic_terms.explode('Topic')
        topic_terms = topic_terms.merge(self.topic['summary'], on=['Topic','Term'])
        topic_terms = topic_terms[topic_terms['Weight']>0]

        topics = topic_terms['Topic'].drop_duplicates()
        topics = f"Predicted Topics {', '.join(topics)}"
        self.set_topics_distribution(topics, topic_terms)
        
        self.set_samples('Topic Prediction', text, None, topic_terms['Term'], None)


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


    def set_topics_distribution(self, title_text, topic_terms):

        self.topic_distribution_factors = topic_terms['Term'].drop_duplicates().tolist()
        self.figure['topic_distribution'].x_range.factors = self.topic_distribution_factors

        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].right = []

        topics = topic_terms['Topic'].drop_duplicates()
        legend = []
        for topic_number in topics.values:
            
            source = topic_terms.loc[
                topic_terms['Topic']==topic_number,
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

        self.default_selections(event='selected_topic', ignore=pd.Series(['topic_number']))

        self.topic_number = self.source['topic_number'].data['Topic'].iloc[new].values[0]

        topics = self.topic['Distribution'][
            (self.topic['Distribution']['Topic']==self.topic_number) & (self.topic['Distribution']['Rank']==1)
        ]

        document_idx = topics.index
        topic_terms = self.topic['summary'].loc[
            (self.topic['summary']['Topic'].isin(topics['Topic'])) & (self.topic['summary']['Weight']>0)
        ]

        text = self.data_input[document_idx]

        self.set_topics_distribution(self.topic_number, topic_terms)
        labeled_entity = self.entity['terms'].loc[document_idx,['entity_text','entity_label']]
        title = f'Selected Topic = {self.topic_number}'
        
        self.set_samples(title, text, None, topic_terms['Term'], labeled_entity)


    def set_yaxis_range(self, attr, old, new, fig_name, num_factors):
        
        start = floor(new)
        end = start+num_factors-1
        end = min(self.input['axis_range'][fig_name].end, end)

        self.figure[fig_name].y_range.factors = self.factors[fig_name][start-1:end]
        self.figure[fig_name].yaxis[0].axis_label = f'Terms {start}-{end}'
