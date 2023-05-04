import re
from math import floor, ceil

import pandas as pd

from bokeh.models import FactorRange

from nlp_newsgroups.default import default

class actions(default):

    def __init__(self):

        default.__init__(self)

        self.html_tag = {
            'stopwords': ('<s>', '</s>'),
            'selected_terms': ('<u>', '</u>')
            # 'topic_terms': ('<span style="background-color:coral">', '</span>'),
            # 'labeled_entity': ('<strong>', '<sup>Entity Label</sup>')
        }


    def set_status(self, message):

        # BUG: emit status message before callbacks complete
        self.status_message.text = message


    def set_samples(self, sample_title, text, selected_terms, labeled_entity):

        self.title['sample'].text = f'Example Documents:<br>{sample_title}'
        self.sample_legend.text = f'''
        <font size="2"><strong><u>Legend:</u></strong><br></font>
        {self.html_tag['selected_terms'][0]}Selected Term{self.html_tag['selected_terms'][1]}<br>
        {self.html_tag['stopwords'][0]}Stop Words{self.html_tag['stopwords'][1]}<br>
        Topic Terms (color = topic name)<br>
        <strong>Entity Name</strong><sup>Entity Label</sup>
        '''
        self.sample_number.title = f'Document Sample #: {len(text)} total'
        self.sample_number.high = len(text)-1
        self.sample_text = text
        self.sample_selected_terms = selected_terms
        self.sample_entity_labels = labeled_entity

        if self.sample_toggle.labels[self.sample_toggle.active] == "Document Samples":
            self.selected_sample(None, None, [0], None, None)
            

    def search_pattern(self, terms):

        pattern = terms.apply(lambda x: re.escape(x))
        pattern = pattern.replace(r'\\ ', '.+', regex=True)
        pattern = r'\b('+pattern+r')\b'
        pattern = '|'.join(pattern)

        return pattern


    def surround_html_tag(self, text, terms, formatter):

        if len(terms)==0:
            return text

        pattern = self.search_pattern(terms)

        replace = f'{self.html_tag[formatter][0]}\g<0>{self.html_tag[formatter][1]}'

        text = re.sub(pattern, replace, text, flags=re.IGNORECASE)

        return text
            

    def find_topic_terms(self, document_idx):

        topic_confidence = self.topic['confidence'].loc[
            (self.topic['confidence'].index==document_idx) & (self.topic['confidence']['Confidence']>0),
            ['Topic','Confidence']
        ]
        
        topic_terms = topic_confidence.merge(self.topic['summary'][['Topic','Term','Weight']], on='Topic')
        topic_terms = topic_terms[topic_terms['Weight']>0]

        topic_terms = topic_terms[
            topic_terms['Term'].isin(
                    self.topic['terms'][
                        self.topic['features'][document_idx,:].nonzero()[1]
                    ]
            )
        ]

        return topic_confidence, topic_terms


    def find_topic_colors(self, document):

        lookup = pd.DataFrame({
            'Topic': self.topic_color.transform.factors, 
            'Color': self.topic_color.transform.palette
        })
        lookup = lookup.merge(document[['Topic','Term']], on='Topic')
        lookup = lookup.drop(columns='Topic').set_index('Term')
        lookup = lookup['Color'].to_dict()

        return lookup


    def highlight_topics(self, text, document_idx, topic_confidence, topic_terms):

        if document_idx is not None:
            topic_confidence, topic_terms = self.find_topic_terms(document_idx)

        self.set_topic_confidence(topic_confidence)

        if len(topic_terms)==0:
            return text

        pattern = self.search_pattern(topic_terms['Term'])
        
        lookup = self.find_topic_colors(topic_terms)

        color = lambda m: f'<span style="background-color:{lookup[m.group().lower()]}">{m.group()}</span>'
        text = re.sub(pattern, color, text, flags=re.IGNORECASE)

        return text


    def highlight_entities(self, text, document_idx):

        text = list(text)

        document = self.sample_entity_labels[
            self.sample_entity_labels['Document Index']==document_idx
        ]
        for document_idx, row in document.iterrows():
            text[row['Start Character Index']] = f"<strong>{text[row['Start Character Index']]}"
            text[row['End Character Index']-1] = f"{text[row['End Character Index']-1]}</strong><sup>{row['Entity Label']}</sup>"

        text = ''.join(text) 

        return text


    def set_topic_confidence(self, distribution):

        self.figure['topic_confidence'].y_range.factors = distribution.loc[distribution['Confidence']>0, 'Topic'].values
        self.source['topic_confidence'].data = distribution.to_dict(orient='list')


    def get_topic_prediction(self, event):

        self.default_selections(event='get_topic_prediction', ignore=None)

        text = pd.Series([self.predict['input'].value])

        features = self.topic['vectorizer'].transform(text)

        topic_confidence = self.assign_topic(self.topic['model'], features)

        predicted_topic = topic_confidence.loc[topic_confidence['Confidence']>0, 'Topic']
        
        idx = features.nonzero()[1]
        topic_terms = pd.DataFrame({
            'Topic': [predicted_topic]*len(idx),
            'Term': self.topic['terms'].loc[idx],
        })
        topic_terms = topic_terms.explode('Topic')
        topic_terms = topic_terms.merge(self.topic['summary'], on=['Topic','Term'])
        topic_terms = topic_terms[topic_terms['Weight']>0]

        title = topic_terms['Topic'].drop_duplicates()
        title = f"Predicted Topics = {', '.join(title)}"
        self.set_topic_term_importance(title, topic_terms)
        
        self.set_samples(
            'Topic Prediction', text, selected_terms=None, labeled_entity=None
        )

        self.selected_sample(None, None, self.sample_number.value, topic_confidence, topic_terms)


    def rename_topic(self, event):

        # TODO: save model with new topic name
        idx = self.topic['name'].index(self.topic_number)
        new_name = self.input['topic_name'].value
        self.topic['name'][idx] = new_name
        self.figure['topic_confidence'].y_range.factors = self.topic['name']
        
        self.topic['summary']['Topic'] = self.topic['summary']['Topic'].replace(self.topic_number, new_name)
        self.topic['rollup'] = self.topic['rollup'].reset_index()
        self.topic['rollup']['Topic'] = self.topic['rollup']['Topic'].replace(self.topic_number, new_name)
        self.topic['rollup'] = self.topic['rollup'].set_index('Topic')
        self.topic['confidence']['Topic'] = self.topic['confidence']['Topic'].replace(self.topic_number, new_name)
        self.glyph['topic_term'].glyph.fill_color = self.topic_color

        self.default_figures(None)

    
    def set_topic_term_importance_range(self, attr, old, new):

        start = floor(new[0])-1
        end = ceil(new[1])

        self.figure['topic_distribution'].x_range.factors = self.topic_distribution_factors[start:end+1]
        self.figure['topic_distribution'].xaxis[0].axis_label = f'Terms {start+1}-{end}'


    def set_topic_term_importance(self, title_text, topic_terms):

        self.figure['topic_distribution'].title.text = title_text

        self.topic_distribution_factors = topic_terms['Term'].drop_duplicates().tolist()
        self.figure['topic_distribution'].x_range.factors = self.topic_distribution_factors

        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].right = []

        topics = topic_terms['Topic'].drop_duplicates()
        for topic_number in topics.values:
            
            source = topic_terms.loc[
                topic_terms['Topic']==topic_number,
                ['Term', 'Weight']
            ].to_dict(orient='list')
            
            idx = self.topic_color.transform.factors[
                self.topic_color.transform.factors==topic_number
            ].index[0]
            topic_color = self.topic_color.transform.palette[idx]
            
            self.figure['topic_distribution'].line(
                x='Term', y='Weight', source=source, line_width=4, line_color=topic_color
            )
            self.figure['topic_distribution'].square(
                x='Term', y='Weight', source=source, size=10, fill_color=topic_color, line_color=None
            )

        self.input['topic_distribution_range'].end = len(self.topic_distribution_factors)
        self.input['topic_distribution_range'].value = (1, min(self.input['topic_distribution_range'].end, 25))
        self.set_topic_term_importance_range(None, None, self.input['topic_distribution_range'].value)


    def set_axis_range(self, attr, old, new, fig_name, num_factors):
        
        start = floor(new)
        end = start+num_factors-1
        end = min(self.input['axis_range'][fig_name].end, end)

        if isinstance(self.figure[fig_name].y_range, FactorRange):
            self.figure[fig_name].y_range.factors = self.factors[fig_name][start-1:end]
            self.figure[fig_name].yaxis[0].axis_label = f'Terms {start}-{end}'
        else:
            self.figure[fig_name].x_range.factors = self.factors[fig_name][start-1:end]
            self.figure[fig_name].xaxis[0].axis_label = f'Terms {start}-{end}'
