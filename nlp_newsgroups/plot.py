import re
from operator import itemgetter
import os
import pickle

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    Div, ColumnDataSource, Spinner, ColorBar, Button, TextInput, CustomJS,
    Slider, RangeSlider, NumericInput, Select, TextAreaInput
)
from bokeh.transform import linear_cmap, factor_cmap
import pandas as pd
from squarify import normalize_sizes, squarify


from nlp_newsgroups.data import data
from nlp_newsgroups.model import model

class plot(data, model):

    def __init__(self):

        data.__init__(self)

        self.source = {}
        self.figure = {}
        self.text = {}

        self.user_inputs()

        self.model_cache()

        self.plot_titles()
        
        self.status_message = Div(text='')

        self.plot_ngram()
        self.plot_topics()
        self.predict_topics()
        self.plot_samples()


    def model_cache(self, input_params={}):

        file_name = 'model.pkl'

        if not os.path.isfile(file_name) or input_params:

            model.__init__(self, **input_params)

            self.get_ngram(self.data_all['text'])
            self.get_topics(self.data_all['text'])

            if input_params:
                self.default_figures()

            # TODO: save new model changes
            if not input_params:

                with open('model.pkl', 'wb') as _fh:
                    pickle.dump([self.model_params, self.ngram, self.topic], _fh)
    
        else:
            with open('model.pkl', 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)


    def plot_titles(self):

        self.title = {
            'main': Div(text='Newsgroups NLP Dashboard', styles={'font-size': '150%', 'font-weight': 'bold'}, width=275),
            'ngram': Div(text='Term Counts', styles={'font-size': '125%', 'font-weight': 'bold'}, width=200),
            'topics': Div(text='Document Topics', styles={'font-size': '125%', 'font-weight': 'bold'}, width=200),
            'sample': Div(text='', styles={'font-weight': 'bold', 'font-size': '125%'}, width=250)
        }


    def default_figures(self):

        self.default_ngram()
        self.default_topics()
        self.default_samples()


    def default_ngram(self, top_num=25):

        ngram = self.ngram['summary'].head(top_num)

        self.source['ngram'].data = {
            'Terms': ngram['terms'],
            'Term Count': ngram['term_count'],
            'Document Count': ngram['document_count']
        }

        self.figure['ngram'].y_range.factors = ngram['terms'].tolist()


    def default_topics(self, top_num=10):

        def treemap(df, col, x, y, dx, dy, *, N=100):
            sub_df = df.nlargest(N, col)
            normed = normalize_sizes(sub_df[col], dx, dy)
            blocks = squarify(normed, x, y, dx, dy)
            blocks_df = pd.DataFrame.from_dict(blocks).set_index(sub_df.index)
            return sub_df.join(blocks_df, how='left').reset_index()

        topics_combined = self.topic['summary'][self.topic['summary']['Rank']<top_num].copy()
        topics_combined = topics_combined.sort_values(by='Weight')

        topics_rollup = topics_combined.groupby('Topic').sum('Weight').sort_values(by='Weight')
        source_text = treemap(topics_rollup, "Weight", 0, 0, self.figure['topics'].width, self.figure['topics'].height)

        source_data = pd.DataFrame()
        for _, (Topic, _, _, x, y, dx, dy) in source_text.iterrows():
            df = topics_combined[topics_combined.Topic==Topic]
            source_data = pd.concat([
                source_data,
                treemap(df, "Weight", x, y, dx, dy, N=10)
            ], ignore_index=True)

        source_data["ytop"] = source_data['y'] + source_data['dy']
        source_data = source_data.to_dict(orient='series')

        self.source['topics'].data = source_data
        self.text['topic_num'].data = source_text.to_dict(orient='series')


    def set_status(self, message):

        # BUG: emit status message before callbacks complete
        self.status_message.text = message


    def recalculate_model(self, event):

        # message = "Recalculating Models! This may take a few minutes."
        # self.popup_alert(message)

        input_params = {key: val.value for key,val in self.inputs.items()}
        
        input_params['stop_words'] = self.model_params['stop_words']+[input_params['stop_words'].strip().lower()]

        change_params = [key for key,val in input_params.items() if val!= self.model_params[key]]
        change_params = [self.inputs[key].title for key in change_params]
        change_params = ', '.join(change_params)

        if change_params:

            # message = f"Recalculating model with new parameters for: {change_params}"
            # self.set_status(message)

            self.inputs['stop_words'].value = ""

            self.model_cache(input_params)


    def selected_reset(self, event):

        self.default_samples()
        self.default_selections()


    def user_inputs(self):

        self.input_reset = Button(label="Reset Selections", button_type="success")
        self.input_reset.on_event("button_click", self.selected_reset)

        self.input_recalculate = Button(label="Recalculate Models", button_type="danger")
        self.input_recalculate.on_event("button_click", self.recalculate_model)
        code = '{ alert("Recalculating Models! This may take a few minutes."); }'
        self.input_recalculate.js_on_click(CustomJS(code=code))

        # BUG: initialize model with these values, recalcuate if needed
        self.inputs = {
            'stop_words': TextInput(value="", title="Add Stopword", width=125),
            'max_df': Slider(start=0.75, end=1.0, value=0.95, step=0.05, title='Max Doc. Freq.', width=125),
            'min_df': Slider(start=1, end=len(self.data_all), value=2, step=1, title='Min Doc. #', width=125),
            'num_features': NumericInput(value=1000, low=1000, high=10000, title='# Features', width=75),
            'ngram_range': RangeSlider(start=1, end=3, value=(1,2), step=1, title='N-Gram Range', width=125),
            'topic_num': Slider(start=2, end=10, value=5, step=1, title='# Topics', width=125),
            'topic_approach': Select(
                value="Latent Dirichlet Allocation", 
                options=["Latent Dirichlet Allocation", "Non-negative Matrix Factorization", "MiniBatch Non-negative Matrix Factorization"],
                title='Topic Model',
                width=200
            )
        }


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

        reset = list(self.source.keys())
        if ignore is not None:
            reset.remove(ignore)

        for source in reset:
            self.source[source].selected.indices = []


    def plot_samples(self):

        self.sample_legend = Div(text='')

        self.sample_document = Div(
            text='', width=1400, height=100
        )

        self.sample_number = Spinner(low=0, high=1, value=0, step=1, width=100)
        self.sample_number.on_change('value', self.selected_sample)

        self.default_samples()


    def set_samples(self, sample_title, sample_legend, devectorized, document_idx, highlight_tokens):

        text = self.data_all.loc[document_idx,'text']
        devectorized = itemgetter(*document_idx)(devectorized)

        self.title['sample'].text = f'Example Documents: {sample_title}'
        self.sample_legend.text = f'<strong>Legend</strong><br>Bold: {sample_legend}<br> Underline: other feature terms'
        self.sample_number.title = f'Document Sample #: {len(text)} total'
        self.sample_number.high = len(text)-1
        self.sample_text = text
        self.sample_highlight = highlight_tokens
        self.sample_devectorized = devectorized
        self.selected_sample(None, None, self.sample_number.value)


    def selected_sample(self, attr, old, new):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]

            pattern = self.model_params['token_pattern']
            pattern = '[^'+pattern+']'
            tokens = re.sub(pattern, ' ', text)

            pattern = self.sample_highlight
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            matching_terms = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            pattern = pd.Series(self.sample_devectorized[new])
            pattern = pattern[~pattern.isin(self.sample_highlight)]
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            matching_features = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            text = list(text)
            for match in matching_terms:
                idx_start = match.start()
                idx_end = match.end()-1
                text[idx_start] = f'<text="2"><strong>{text[idx_start]}'
                text[idx_end] = f'{text[idx_end]}</text></strong>'
            for match in matching_features:
                idx_start = match.start()
                idx_end = match.end()-1
                text[idx_start] = f'<u>{text[idx_start]}'
                text[idx_end] = f'{text[idx_end]}</u>'
            text = ''.join(text)

            self.sample_document.text = text


    def selected_ngram(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='ngram')

        sample_title = self.figure['ngram'].title.text
        terms = self.source['ngram'].data['Terms'].iloc[new]
        sample_legend = f"Terms {','.join(terms.tolist())}"

        document_idx = self.ngram['features'][:, terms.index].nonzero()[0]
        highlight_tokens = terms

        self.set_samples(sample_title, sample_legend, self.ngram['devectorized'], document_idx, highlight_tokens)


    def plot_ngram(self):

        self.figure['ngram'] = figure(
            height=500, width=350, toolbar_location=None, tools="tap", tooltips="Document Count = @{Document Count}",
            x_axis_label='Term Count', y_axis_label='Term', y_range=[]
        )

        self.source['ngram'] = ColumnDataSource()
        self.default_ngram()

        cmap = linear_cmap(
            field_name='Document Count', palette='Turbo256', 
            low=min(self.source['ngram'].data['Document Count']), high=max(self.source['ngram'].data['Document Count'])
        )
        color_bar = ColorBar(color_mapper=cmap['transform'], title='Document Count')

        self.figure['ngram'].hbar(
            y='Terms', right='Term Count',
            source=self.source['ngram'], width=0.9, fill_color=cmap, line_color=None
        )
        self.figure['ngram'].add_layout(color_bar, 'right')   

        self.source['ngram'].selected.on_change('indices', self.selected_ngram)


    def selected_topic(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='topics')

        sample_title = self.figure['topics'].title.text

        topics_number = self.source['topics'].data['Topic'].iloc[new]

        topics = self.topic['Distribution'][
            (self.topic['Distribution']['Topic'].isin(topics_number)) & (self.topic['Distribution']['Confidence']>0.5)
        ]

        limit = 10
        
        document_idx = topics.index
        highlight_tokens = self.topic['summary'].loc[
            (self.topic['summary']['Topic'].isin(topics['Topic'])) & (self.topic['summary']['Rank']<limit),
            'Term'
        ]

        sample_legend = f"{','.join(topics_number.tolist())} top {limit} terms [{', '.join(highlight_tokens.tolist())}]"

        self.set_samples(sample_title, sample_legend, self.topic['devectorized'], document_idx, highlight_tokens)


    def plot_topics(self):

        self.figure['topics'] = figure(
            width=600, height=500, tooltips="@Term", toolbar_location=None, tools="tap",
            x_axis_location=None, y_axis_location=None
        )

        self.source['topics'] = ColumnDataSource()
        self.text['topic_num'] = ColumnDataSource()
        self.default_topics()

        self.figure['topics'].x_range.range_padding = self.figure['topics'].y_range.range_padding = 0
        self.figure['topics'].grid.grid_line_color = None

        self.topic_color = factor_cmap("Topic", "MediumContrast5", self.source['topics'].data['Topic'].drop_duplicates())
        self.figure['topics'].block(
            'x', 'y', 'dx', 'dy', source=self.source['topics'], line_width=1, line_color="white",
            fill_alpha=0.8, fill_color=self.topic_color
        )

        self.figure['topics'].text(
            'x', 'y', x_offset=2, text="Topic", source=self.text['topic_num'],
            text_font_size="18pt", text_color="white"
        )

        self.figure['topics'].text('x', 'ytop', x_offset=2, y_offset=2, text="Term", source=self.source['topics'],
            text_font_size="10pt", text_baseline="top",
        )

        self.source['topics'].selected.on_change('indices', self.selected_topic)


    def get_topic_prediction(self, event):

        text = [self.topic['predict']['input'].value]

        features = self.topic['vectorizer'].transform(text)

        self.topic['vectorizer'].inverse_transform(features)

        distribution = self.assign_topic(self.topic['model'], features)



    def predict_topics(self):

        self.topic['predict'] = {}

        self.topic['predict']['calculate'] = Button(label='Get Prediction', button_type='primary')
        # self.topic['predict']['calculate'].on_event("button_click", self.selected_reset)

        self.topic['predict']['input'] = TextAreaInput(
            value='', width=600, height=250, title='Predict topic for input text.'
        )

        self.topic['predict']['source'] = ColumnDataSource({'Topic':[], 'Confidence':[]})

        self.topic['predict']['result'] = figure(
            height=300, title='Topic Prediction'
        )

        self.topic['predict']['result'].vbar(
            x='Topic', top='Confidence', source=self.topic['predict']['source'], fill_color=self.topic_color
        )