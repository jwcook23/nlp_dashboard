import re
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bokeh.models import Legend

from nlp_newsgroups.model import model

class actions(model):

    def __init__(self):

        self.model_file_name = 'model.pkl'


    def model_cache(self, input_params={}):

        cache_exists = os.path.isfile(self.model_file_name)
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
                self.default_figures()

            # TODO: save new model changes
            if not params_changed:
                self.save_model(None)
    
        else:
            with open(self.model_file_name, 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)


    def save_model(self, event):

        with open(self.model_file_name, 'wb') as _fh:
            pickle.dump([self.model_params, self.ngram, self.topic], _fh)


    def set_status(self, message):

        # BUG: emit status message before callbacks complete
        self.status_message.text = message


    def recalculate_model(self, event):

        # message = "Recalculating Models! This may take a few minutes."
        # self.popup_alert(message)

        self.selected_reset(None)

        input_params = {key: val.value for key,val in self.model_inputs.items()}
        
        stopwords = input_params['stop_words'].split(',')
        stopwords = [word.strip().lower() for word in stopwords]
        input_params['stop_words'] = self.model_params['stop_words']+stopwords

        change_params = [key for key,val in input_params.items() if val!= self.model_params[key]]
        change_params = [self.model_inputs[key].title for key in change_params]
        change_params = ', '.join(change_params)

        if change_params:

            # message = f"Recalculating model with new parameters for: {change_params}"
            # self.set_status(message)

            self.model_inputs['stop_words'].value = ""

            self.model_cache(input_params)


    def selected_reset(self, event):

        self.default_samples()
        self.default_selections()
        self.default_topics_distribution()
        self.default_topic_assignment()


    def default_samples(self):

        self.title['sample'].text ='Example Documents'
        self.sample_number.title = 'Document Sample #: make selection'
        self.sample_legend.text = ''
        self.sample_document.text = ''
        self.sample_number.value = 0
        self.sample_number.high = 1
        self.sample_text = None


    def default_selections(self, ignore=None):

        self.sample_number.value = 0
        self.topic_number = None

        reset = list(self.source.keys())
        if ignore is not None:
            reset.remove(ignore)

        for source in reset:
            self.source[source].selected.indices = []


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


    def selected_ngram(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='ngram')

        sample_title = self.title['ngram'].text
        important_terms = self.ngram['summary'].iloc[new]

        document_idx = self.ngram['features'][:, important_terms.index].nonzero()[0]

        text = self.data_input[document_idx]

        # TODO: show distribution of term importance
        self.set_samples(sample_title, text, important_terms['terms'])


    def get_topic_prediction(self, event):

        self.default_selections()
        self.default_topics_distribution()

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
        new_name = self.input_topic_name.value
        self.topic['name'][idx] = new_name
        self.predict['figure'].y_range.factors = self.topic['name']
        
        self.topic['summary']['Topic'] = self.topic['summary']['Topic'].replace(self.topic_number, new_name)
        self.default_topics_terms()
        self.default_topics_distribution()
        self.glyph['topic_term'].glyph.fill_color = self.topic_color
        self.default_topic_assignment()
        self.default_selections()


    def set_topics_distribution(self, title_text, important_terms):

        # important_terms = important_terms.sort_values(by=['Topic','Weight'], ascending=[True, False])

        self.figure['topic_distribution'].title.text = f"Topic Term Importance (all terms): {title_text}"
        factors = important_terms['Term'].drop_duplicates().tolist()
        self.figure['topic_distribution'].x_range.factors = factors

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


    def selected_topic(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='topics')

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