from math import pi
from typing import Literal
from functools import partial

from bokeh.plotting import figure
from bokeh.models import (
    Div, ColumnDataSource, Spinner, ColorBar, Button, TextInput, CustomJS,
    Slider, RangeSlider, NumericInput, Select, TextAreaInput, TapTool, HoverTool
)
from bokeh.transform import linear_cmap
import pandas as pd
from squarify import normalize_sizes, squarify

from nlp_newsgroups.data import data
from nlp_newsgroups.model import model
from nlp_newsgroups.selections import selections

class plot(data, model, selections):


    def __init__(self):

        self.figure = {}
        self.source = {}
        self.glyph = {}
        self.input = {'axis_range': {}}

        data.__init__(self)
        self.data_input = self.data_all['text']

        self.user_inputs()

        model.__init__(self)
        selections.__init__(self)

        self.plot_titles()
        
        self.plot_ybar('ngram')
        self.plot_ybar('entity', fig_height=350)
        self.plot_ybar('entity_label', fig_height=150)
        self.plot_topics_terms()
        self.plot_topics_confidence()
        self.plot_topics_distribution()
        self.plot_assignment()
        self.predict_topics()
        self.plot_samples()


    def user_inputs(self):

        self.input['reset'] = Button(label="Reset Selections", button_type="success", width=150)
        self.input['reset'].on_event("button_click", partial(self.default_selections, ignore=None))

        self.input['recalculate'] = Button(label="Recalculate Models", button_type="danger", width=150)
        self.input['recalculate'].on_event("button_click", self.recalculate_model)
        code = '{ alert("Recalculating Models! This may take a few minutes."); }'
        self.input['recalculate'].js_on_click(CustomJS(code=code))

        self.input['save'] = Button(label="Save Models", button_type="warning", width=150)
        self.input['save'].on_event("button_click", self.save_model)
        code = '{ alert("Models Saved!"); }'
        self.input['save'].js_on_click(CustomJS(code=code))

        # BUG: initialize model with these values, recalcuate if needed
        token_pattern = [('(?u)\\b\\w\\w+\\b', '2 or more alphanumeric characters')]
        self.model_inputs = {
            'token_pattern': Select(
                value=token_pattern[0][0], 
                options=token_pattern,
                title='Token Pattern',
                width=250
            ),
            'stop_words': TextInput(value="", title="Add Stopwords (comma seperated)", width=125),
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


    def plot_titles(self):

        self.title = {
            'main': Div(text=f'NLP Dashboard<br>{len(self.data_input):,} Documents', styles={'font-size': '150%', 'font-weight': 'bold'}, width=175),
            'terms_entity': Div(text='Entity or Term Summary', styles={'font-size': '125%', 'font-weight': 'bold'}, width=150),
            'ngram': Div(text='Term', styles={'font-weight': 'bold'}, width=75),
            'entity': Div(text='Entity Name', styles={'font-weight': 'bold'}, width=75),
            'entity_label': Div(text='Entity Label', styles={'font-weight': 'bold'}, width=75),
            'topics': Div(text='Document Topics', styles={'font-size': '125%', 'font-weight': 'bold'}, width=200),
            'sample': Div(text='', styles={'font-weight': 'bold', 'font-size': '125%'}, width=250)
        }


    def topic_treemap(self, top_num=10):

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

        return source_data, source_text
        

    def plot_samples(self):

        self.sample_legend = Div(text='')

        self.sample_document = Div(
            text='', width=1400, height=100
        )

        self.sample_number = Spinner(low=0, high=1, value=0, step=1, width=100)
        self.sample_number.on_change('value', self.selected_sample)

        self.default_samples()


    def plot_ybar(self, fig_name: Literal = ['ngram','entity','entity_label'], fig_height=550):

        self.figure[fig_name] = figure(
            height=fig_height, width=350, toolbar_location=None, tools="tap", 
            tooltips="Term Count = @{Term Count}<br>Document Count = @{Document Count}",
            x_axis_label='Term Count', y_range=[]
        )
        self.figure[fig_name].xaxis.major_label_orientation = pi/8

        self.source[fig_name] = ColumnDataSource()
        self.input['axis_range'][fig_name] = Slider(start=1, end=2, value=1, step=1, title='First Term Displayed', width=125)

        self.default_terms(fig_name)

        num_factors = len(self.figure[fig_name].y_range.factors)
        self.input['axis_range'][fig_name].on_change('value', partial(self.set_yaxis_range, fig_name=fig_name, num_factors=num_factors))

        cmap = linear_cmap(
            field_name='Document Count', palette='Turbo256', 
            low=min(self.source[fig_name].data['Document Count']), high=max(self.source[fig_name].data['Document Count'])
        )
        color_bar = ColorBar(color_mapper=cmap['transform'], title='Document Count')

        self.figure[fig_name].hbar(
            y='Terms', right='Term Count',
            source=self.source[fig_name], width=0.9, fill_color=cmap, line_color=None
        )
        self.figure[fig_name].add_layout(color_bar, 'right')   

        self.source[fig_name].selected.on_change('indices', partial(self.selected_source, fig_name=fig_name))


    def plot_topics_terms(self):

        self.figure['topics'] = figure(
            width=950, height=300, toolbar_location=None,
            x_axis_location=None, y_axis_location=None, title='Topic Term Importance (top 10 terms)'
        )
        self.figure['topics'].x_range.range_padding = self.figure['topics'].y_range.range_padding = 0
        self.figure['topics'].grid.grid_line_color = None
        self.source['topics'] = ColumnDataSource()
        self.source['topic_number'] = ColumnDataSource()
        
        self.default_topics_terms()

        self.glyph['topic_term'] = self.figure['topics'].block(
            'x', 'y', 'dx', 'dy', source=self.source['topics'], line_width=1, line_color="white",
            fill_alpha=0.8, fill_color=self.topic_color
        )
        hover_topic_term = HoverTool(renderers=[self.glyph['topic_term']], tooltips=[('Term', '@Term')])
        self.figure['topics'].add_tools(hover_topic_term)

        self.glyph['topic_number'] = self.figure['topics'].text(
            'x', 'y', x_offset=2, text="Topic", source=self.source['topic_number'],
            text_font_size="18pt", text_color="white"
        )
        hover_topic_number = HoverTool(renderers=[self.glyph['topic_number']], tooltips=[('Topic', '@Topic')])
        self.figure['topics'].add_tools(hover_topic_number)
        self.figure['topics'].add_tools(TapTool(renderers=[self.glyph['topic_number']]))
        self.source['topic_number'].selected.on_change('indices', self.selected_topic)

        self.figure['topics'].text('x', 'ytop', x_offset=2, y_offset=2, text="Term", source=self.source['topics'],
            text_font_size="10pt", text_baseline="top",
        )


    def plot_topics_confidence(self):

        self.source['topic_confidence'] = ColumnDataSource({'Topic':[], 'Confidence':[]})

        # TODO: rename title
        self.figure['topic_confidence'] = figure(
            y_range=self.topic_color.transform.factors, width=300, height=250, title='Topic Prediction',
            x_axis_label='Confidence', toolbar_location=None
        )

        self.figure['topic_confidence'].hbar(
            y='Topic', right='Confidence', source=self.source['topic_confidence'], fill_color=self.topic_color
        )


    def plot_topics_distribution(self):

        self.figure['topic_distribution'] = figure(
            width=950, height=200, toolbar_location=None, tools="tap", x_range=[], y_axis_label='Importance', title=''
        )
        self.figure['topic_distribution'].xaxis.major_label_orientation = pi/8
        self.input['topic_distribution_range'] = RangeSlider(start=1, end=2, value=(1,2), step=1, title='Term Range Displayed', width=125)
        self.input['topic_distribution_range'].on_change('value', self.set_topics_distribution_range)
        self.default_topics_distribution()


    def plot_assignment(self):

        self.input['topic_description'] = Div(text="*Select topic then assign new name.")
        self.input['topic_name'] = TextInput(value="", title="", width=125)
        self.set_topic_name = Button(label="*Rename Topic", button_type="default", width=125)
        self.set_topic_name.on_event("button_click", self.rename_topic)


    def predict_topics(self):

        self.predict = {}

        self.predict['calculate'] = Button(label='Get Prediction', button_type='primary')
        self.predict['calculate'].on_event("button_click", self.get_topic_prediction)

        self.predict['input'] = TextAreaInput(
            value="Baseball season is over. so I'll have more time put my new hard drive in.",
            width=300, height=250, title='Predict topic for input text.'
        )
