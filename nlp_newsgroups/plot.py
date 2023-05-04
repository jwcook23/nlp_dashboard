from math import pi
from typing import Literal
from functools import partial

from bokeh.plotting import figure
from bokeh.models import (
    Div, ColumnDataSource, Spinner, ColorBar, Button, TextInput, CustomJS,
    Slider, RangeSlider, NumericInput, Select, TextAreaInput, TapTool, HoverTool,
    RadioButtonGroup, FactorRange
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
        
        self.plot_bar(factor_axis='x', fig_name='Term Counts', fig_width=1300, fig_height=300)
        self.plot_bar(factor_axis='x', fig_name='Entity Name', fig_width=950, fig_height=300)
        self.plot_bar(factor_axis='y', fig_name='Entity Label', fig_width=350, fig_height=300)
        self.plot_topics_terms()
        self.plot_topics_weight()
        self.plot_topic_term_weight()
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
            'stop_words': TextInput(value="", title="Add Comma Seperated Stopword(s)", width=250),
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
            'main': Div(text=f'NLP Dashboard: {len(self.data_input):,} Documents', styles={'font-size': '150%', 'font-weight': 'bold'}, width=350),
            'terms_entity': Div(text='Entity or Term Summary', styles={'font-size': '125%', 'font-weight': 'bold'}, width=150),
            'Term Counts': Div(text='Term', styles={'font-weight': 'bold'}, width=75),
            'Entity Name': Div(text='Entity Name', styles={'font-weight': 'bold'}, width=75),
            'Entity Label': Div(text='Entity Label', styles={'font-weight': 'bold'}, width=75),
            'Topic Terms': Div(text='Document Topics', styles={'font-size': '125%', 'font-weight': 'bold'}, width=200),
            'sample': Div(text='', styles={'font-weight': 'bold', 'font-size': '125%'}, width=250, visible=False)
        }


    def topic_treemap(self, top_num=10):

        def treemap(df, col, x, y, dx, dy, *, N=100):
            sub_df = df.nlargest(N, col)
            normed = normalize_sizes(sub_df[col], dx, dy)
            blocks = squarify(normed, x, y, dx, dy)
            blocks_df = pd.DataFrame.from_dict(blocks).set_index(sub_df.index)
            return sub_df.join(blocks_df, how='left').reset_index()

        # TODO: can self.topic['rollup'] be used?
        topics_combined = self.topic['summary'][self.topic['summary']['Rank']<top_num].copy()
        topics_combined = topics_combined.sort_values(by='Weight')

        topics_rollup = topics_combined.groupby('Topic').sum('Weight').sort_values(by='Weight')
        source_text = treemap(topics_rollup, "Weight", 0, 0, self.figure['Topic Terms'].width, self.figure['Topic Terms'].height)

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

        self.sample_legend = Div(text='', visible=False)

        self.sample_document = Div(
            text='', width=1250, height=100
        )

        self.sample_toggle = RadioButtonGroup(labels=["All Documents", "Document Samples"], active=0)
        self.sample_toggle.on_event('button_click', self.activate_samples)
        
        self.sample_number = Spinner(low=0, high=1, value=0, step=1, width=100, visible=False)
        self.sample_number.on_change('value', partial(self.selected_sample,topic_weight=None,topic_terms=None))

        self.default_samples()


    def plot_bar(self, factor_axis, fig_name, fig_width, fig_height):

        tooltips = "Term Count = @{Term Count}<br>Document Count = @{Document Count}"

        if factor_axis=='y':
            self.figure[fig_name] = figure(
                height=fig_height, width=fig_width, toolbar_location=None, tools="tap", tooltips=tooltips,
                x_axis_label='Term Count', y_range = []
            )
            self.figure[fig_name].xaxis.major_label_orientation = pi/8
        else:
            self.figure[fig_name] = figure(
                height=fig_height, width=fig_width, toolbar_location=None, tools="tap", tooltips=tooltips,
                y_axis_label='Term Count', x_range = []
            )
            self.figure[fig_name].xaxis.major_label_orientation = pi/8

        self.source[fig_name] = ColumnDataSource()
        self.input['axis_range'][fig_name] = Slider(start=1, end=2, value=1, step=1, title='First Term Displayed', width=125)

        self.default_terms(fig_name)

        if factor_axis=='y':
            num_factors = len(self.figure[fig_name].y_range.factors)
        else:
            num_factors = len(self.figure[fig_name].x_range.factors)
        self.input['axis_range'][fig_name].on_change('value', partial(self.set_axis_range, fig_name=fig_name, num_factors=num_factors))

        cmap = linear_cmap(
            field_name='Document Count', palette='Cividis256', 
            low=min(self.source[fig_name].data['Document Count']), high=max(self.source[fig_name].data['Document Count'])
        )
        color_bar = ColorBar(color_mapper=cmap['transform'], title='Document Count')

        if factor_axis=='y':
            self.figure[fig_name].hbar(
                y='Terms', right='Term Count', source=self.source[fig_name],
                width=0.9, fill_color=cmap, line_color=None
            )
            self.figure[fig_name].add_layout(color_bar, 'right')
        else:
            self.figure[fig_name].vbar(
                x='Terms', top='Term Count', source=self.source[fig_name],
                width=0.9, fill_color=cmap, line_color=None
            )
            self.figure[fig_name].add_layout(color_bar, 'right')

        self.source[fig_name].selected.on_change('indices', partial(self.selected_source, fig_name=fig_name))


    def plot_topics_terms(self):

        self.figure['Topic Terms'] = figure(
            width=1250, height=300, toolbar_location=None,
            x_axis_location=None, y_axis_location=None, title='Topic Term Weight (top 10 terms)'
        )
        self.figure['Topic Terms'].x_range.range_padding = self.figure['Topic Terms'].y_range.range_padding = 0
        self.figure['Topic Terms'].grid.grid_line_color = None
        self.source['Topic Terms'] = ColumnDataSource()
        self.source['Topic Number'] = ColumnDataSource()
        
        self.default_topics_terms()

        self.glyph['Topic Terms'] = self.figure['Topic Terms'].block(
            'x', 'y', 'dx', 'dy', source=self.source['Topic Terms'], line_width=1, line_color="white",
            fill_alpha=0.8, fill_color=self.topic_color
        )
        hover_topic_term = HoverTool(renderers=[self.glyph['Topic Terms']], tooltips=[('Term', '@Term')])
        self.figure['Topic Terms'].add_tools(hover_topic_term)

        self.glyph['Topic Number'] = self.figure['Topic Terms'].text(
            'x', 'y', x_offset=2, text="Topic", source=self.source['Topic Number'],
            text_font_size="18pt", text_color="white"
        )
        hover_topic_number = HoverTool(renderers=[self.glyph['Topic Number']], tooltips=[('Topic', '@Topic')])
        self.figure['Topic Terms'].add_tools(hover_topic_number)
        self.figure['Topic Terms'].add_tools(TapTool(renderers=[self.glyph['Topic Number']]))
        self.source['Topic Number'].selected.on_change('indices', self.selected_topic)

        self.figure['Topic Terms'].text('x', 'ytop', x_offset=2, y_offset=2, text="Term", source=self.source['Topic Terms'],
            text_font_size="10pt", text_baseline="top",
        )


    def plot_topics_weight(self):

        self.source['Topic Weight'] = ColumnDataSource({'Topic':[], 'Weight':[]})

        self.figure['Topic Weight'] = figure(
            y_range=self.topic_color.transform.factors, width=300, height=200, title='Topic Distribution & Color Legend',
            x_axis_label='Weight', toolbar_location=None
        )

        self.figure['Topic Weight'].hbar(
            y='Topic', right='Weight', source=self.source['Topic Weight'], fill_color=self.topic_color
        )

        self.default_topic_weight()


    def plot_topic_term_weight(self):

        self.figure['Topic Distribution'] = figure(
            width=950, height=200, toolbar_location=None, tools="tap", x_range=[], y_axis_label='Weight', title=''
        )
        self.figure['Topic Distribution'].xaxis.major_label_orientation = pi/8
        self.input['topic_distribution_range'] = RangeSlider(start=1, end=2, value=(1,2), step=1, title='Term Range Displayed', width=125)
        self.input['topic_distribution_range'].on_change('value', self.set_topic_term_weight_range)
        self.default_topic_term_weight()


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
            width=1250, height=250, title='Predict topic for input text.'
        )
