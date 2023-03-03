import os
import pickle

from bokeh.plotting import figure
from bokeh.models import (
    Div, ColumnDataSource, Spinner, ColorBar, Button, TextInput, CustomJS,
    Slider, RangeSlider, NumericInput, Select, TextAreaInput
)
from bokeh.transform import linear_cmap, factor_cmap
from bokeh.palettes import Category10
import pandas as pd
from squarify import normalize_sizes, squarify
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nlp_newsgroups.data import data
from nlp_newsgroups.model import model
from nlp_newsgroups.actions import actions

class plot(data, model, actions):


    def __init__(self):

        data.__init__(self)
        actions.__init__(self)

        self.data_input = self.data_all['text']

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


    def user_inputs(self):

        self.input_reset = Button(label="Reset Selections", button_type="success")
        self.input_reset.on_event("button_click", self.selected_reset)

        self.input_recalculate = Button(label="Recalculate Models", button_type="danger")
        self.input_recalculate.on_event("button_click", self.recalculate_model)
        code = '{ alert("Recalculating Models! This may take a few minutes."); }'
        self.input_recalculate.js_on_click(CustomJS(code=code))

        # BUG: initialize model with these values, recalcuate if needed
        token_pattern = [('(?u)\\b\\w\\w+\\b', '2 or more alphanumeric characters')]
        self.model_inputs = {
            'token_pattern': Select(
                value=token_pattern[0][0], 
                options=token_pattern,
                title='Token Pattern',
                width=250
            ),
            'stop_words': TextInput(value="", title="Add Stopword", width=125),
            'max_df': Slider(start=0.75, end=1.0, value=0.95, step=0.05, title='Max Doc. Freq.', width=125),
            'min_df': Slider(start=1, end=len(self.data_all), value=2, step=1, title='Min Doc. #', width=125),
            'num_features': NumericInput(value=1000, low=1000, high=10000, title='# Features', width=75),
            'ngram_range': RangeSlider(start=1, end=3, value=(1,2), step=1, title='N-Gram Range', width=125),
            'topic_num': Slider(start=1, end=10, value=10, step=1, title='# Topics', width=125),
            'topic_approach': Select(
                value="Non-negative Matrix Factorization", 
                options=["Latent Dirichlet Allocation", "Non-negative Matrix Factorization", "MiniBatch Non-negative Matrix Factorization"],
                title='Topic Model',
                width=300
            )
        }


    def model_cache(self, input_params={}):

        file_name = 'model.pkl'
        cache_exists = os.path.isfile(file_name)
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
                with open('model.pkl', 'wb') as _fh:
                    pickle.dump([self.model_params, self.ngram, self.topic], _fh)
    
        else:
            with open('model.pkl', 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)


    def plot_titles(self):

        self.title = {
            'main': Div(text=f'NLP Dashboard<br>{len(self.data_input):,} Documents', styles={'font-size': '150%', 'font-weight': 'bold'}, width=275),
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
            df = topics_combined[(topics_combined.Topic==Topic) & (topics_combined.Weight>0)]
            source_data = pd.concat([
                source_data,
                treemap(df, "Weight", x, y, dx, dy, N=10)
            ], ignore_index=True)

        source_data["ytop"] = source_data['y'] + source_data['dy']
        source_data = source_data.to_dict(orient='series')

        self.source['topics'].data = source_data
        self.text['topic_num'].data = source_text.to_dict(orient='series')


    def plot_samples(self):

        self.sample_legend = Div(text='')

        self.sample_document = Div(
            text='', width=1400, height=100
        )

        self.sample_number = Spinner(low=0, high=1, value=0, step=1, width=100)
        self.sample_number.on_change('value', self.selected_sample)

        self.default_samples()


    def plot_ngram(self):

        self.figure['ngram'] = figure(
            height=500, width=400, toolbar_location=None, tools="tap", tooltips="Document Count = @{Document Count}",
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


    def plot_topics(self):

        self.figure['topics'] = figure(
            width=800, height=500, tooltips="@Term", toolbar_location=None, tools="tap",
            x_axis_location=None, y_axis_location=None, title='Topic Term Importance'
        )

        self.source['topics'] = ColumnDataSource()
        self.text['topic_num'] = ColumnDataSource()
        self.default_topics()

        self.figure['topics'].x_range.range_padding = self.figure['topics'].y_range.range_padding = 0
        self.figure['topics'].grid.grid_line_color = None

        factors = self.source['topics'].data['Topic'].drop_duplicates()
        self.topic_color = factor_cmap("Topic", palette=Category10[10], factors=factors)
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


    def predict_topics(self):

        self.predict = {}

        self.predict['calculate'] = Button(label='Get Prediction', button_type='primary')
        self.predict['calculate'].on_event("button_click", self.get_topic_prediction)

        self.predict['input'] = TextAreaInput(
            value="Football season is over. so I'll have more time put my new hard drive in.",
            width=300, height=250, title='Predict topic for input text.'
        )

        self.predict['source'] = ColumnDataSource({'Topic':[], 'Confidence':[]})

        self.predict['figure'] = figure(
            y_range=self.topic_color.transform.factors, width=300, height=250, title='Topic Prediction',
            x_axis_label='Confidence', toolbar_location=None
        )

        self.predict['renderer'] = self.predict['figure'].hbar(
            y='Topic', right='Confidence', source=self.predict['source'], fill_color=self.topic_color
        )