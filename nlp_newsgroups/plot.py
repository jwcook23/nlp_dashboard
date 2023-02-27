import re
from operator import itemgetter
import os
import pickle

from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, Slider, ColorBar, Button
from bokeh.transform import linear_cmap, factor_cmap
import pandas as pd
from squarify import normalize_sizes, squarify


from nlp_newsgroups.data import data
from nlp_newsgroups.model import model

class plot(data, model):

    def __init__(self):

        data.__init__(self)

        self.model_cache()

        self.source = {}
        self.figure = {}

        self.plot_titles()
        self.reset_button()
        self.plot_samples()
        self.plot_ngram()
        self.plot_topics()


    def model_cache(self):

        # TODO: option to clear model cache
        file_name = 'model.pkl'
        if os.path.isfile(file_name):
            with open('model.pkl', 'rb') as _fh:
                self.model_params, self.ngram, self.topic = pickle.load(_fh)
        else:
            model.__init__(self)
            self.get_ngram(self.data_all['text'])
            self.get_topics(self.data_all['text'])
            
            with open('model.pkl', 'wb') as _fh:
                pickle.dump([self.model_params, self.ngram, self.topic], _fh)


    def plot_titles(self):

        self.title_main = Div(
            text='Newsgroups NLP Dashboard',
            styles={'font-size': '150%', 'font-weight': 'bold'}, width=400
        )


    def selected_reset(self, event):

        self.default_samples()
        self.default_selections()


    def reset_button(self):

        self.input_reset = Button(label="Reset Selections", button_type="success")
        self.input_reset.on_event("button_click", self.selected_reset)


    def default_samples(self):

        self.sample_title.text ='Example Documents'
        self.sample_subtitle.text = 'make a selection to display examples'
        self.sample_document.text = ''
        self.sample_number.value = 0
        self.sample_number.end = 1
        self.sample_text = None


    def default_selections(self, ignore=None):

        reset = list(self.source.keys())
        if ignore is not None:
            reset.remove(ignore)

        for source in reset:
            self.source[source].selected.indices = []


    def plot_samples(self):

        self.sample_title = Div(text='', styles={'font-weight': 'bold'})
        self.sample_subtitle = Div(text='')

        self.sample_document = Div(
            text='', width=1400, height=100
        )

        self.sample_number = Slider(start=0, end=1, value=0, step=1, title="Document Sample #", width=150)
        self.sample_number.on_change('value_throttled', self.selected_sample)

        self.default_samples()


    def set_samples(self, sample_title, sample_subtitle, terms, features, devectorized):

        idx = features[:, terms.index].nonzero()[0]
        text = self.data_all.loc[idx,'text']
        devectorized = itemgetter(*idx)(devectorized)

        self.sample_title.text = f'Example Documents: {sample_title}'
        self.sample_subtitle.text = f'Total Documents = {len(text)}<br>{sample_subtitle}'
        self.sample_number.end = len(text)-1
        self.sample_text = text
        self.sample_terms = terms
        self.sample_devectorized = devectorized
        self.selected_sample(None, None, 0)


    def selected_sample(self, attr, old, new):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]

            pattern = self.model_params['token_pattern']
            pattern = '[^'+pattern+']'
            tokens = re.sub(pattern, ' ', text)

            # matching terms: bold and underline
            pattern = self.sample_terms
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            highlight_terms = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            # feature terms: bold
            pattern = pd.Series(self.sample_devectorized[new])
            pattern = pattern[~pattern.isin(self.sample_terms)]
            pattern = pattern.str.replace(' ', r'\s+', regex=True)
            pattern = '|'.join(r'\b'+pattern+r'\b')
            bold_features = re.finditer(pattern, tokens, flags=re.IGNORECASE)

            text = list(text)
            for match in highlight_terms:
                text[match.start()] = f'<font size="4"><strong><u>{text[match.start()]}'
                text[match.end()] = f'{text[match.end()]}</font></u></strong>'
            for match in bold_features:
                text[match.start()] = f'<font size="4"><strong>{text[match.start()]}'
                text[match.end()] = f'{text[match.end()]}</font></strong>'
            text = ''.join(text)

            self.sample_document.text = text


    def selected_ngram(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='ngram')

        sample_title = self.figure['ngram'].title.text
        terms = self.source['ngram'].data['y'].iloc[new]
        sample_subtitle = 'terms: '+','.join(terms.tolist())

        self.set_samples(sample_title, sample_subtitle, terms, self.ngram['features'], self.ngram['devectorized'])


    def plot_ngram(self, top_num=25):

        ngram = self.ngram['summary'].head(top_num)
        # .sort_values(by='term_count')

        self.figure['ngram'] = figure(
            y_range=ngram['terms'], height=500, width=400, toolbar_location=None, tools="tap", 
            title="One & Two Word Term Counts", x_axis_label='Term Count', y_axis_label='Term'

        )

        self.source['ngram'] = ColumnDataSource(data=dict({
            'y': ngram['terms'],
            'right': ngram['term_count'],
            'color': ngram['document_count']
        }))

        cmap = linear_cmap(
            field_name='color', palette='Turbo256', 
            low=min(ngram['document_count']), high=max(ngram['document_count'])
        )
        color_bar = ColorBar(color_mapper=cmap['transform'], title='Document Count')

        self.figure['ngram'].hbar(
            source=self.source['ngram'], width=0.9, fill_color=cmap, line_color=None
        )
        self.figure['ngram'].add_layout(color_bar, 'above')   

        self.source['ngram'].selected.on_change('indices', self.selected_ngram)


    def selected_topic(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return
        
        self.default_selections(ignore='topics')

        sample_title = self.figure['topics'].title.text

        topics = self.source['topics'].data['Topic'].iloc[new]
        topics = self.topic['Distribution'][
            (self.topic['Distribution']['Topic'].isin(topics))
        ].index
        # 
        # self.topic['summary'][self.topic['summary']['Topic'].isin(topics)]


        terms = self.source['topics'].data['Term'].iloc[new]
        terms = self.topic['terms'][self.topic['terms'].isin(terms)]
        sample_subtitle = 'terms: '+','.join(terms.tolist())

        self.set_samples(sample_title, sample_subtitle, terms, self.topic['features'], self.topic['devectorized'])


    def plot_topics(self, top_num=10):
        
        topics_combined = self.topic['summary'][self.topic['summary']['Rank']<top_num].copy()

        topics_combined = topics_combined.sort_values(by='Weight')
        topics_rollup = topics_combined.groupby('Topic').sum('Weight').sort_values(by='Weight')

        def treemap(df, col, x, y, dx, dy, *, N=100):
            sub_df = df.nlargest(N, col)
            normed = normalize_sizes(sub_df[col], dx, dy)
            blocks = squarify(normed, x, y, dx, dy)
            blocks_df = pd.DataFrame.from_dict(blocks).set_index(sub_df.index)
            return sub_df.join(blocks_df, how='left').reset_index()

        width = 800
        height = 450
        topics_rollup = treemap(topics_rollup, "Weight", 0, 0, width, height)

        self.source['topics'] = pd.DataFrame()
        for _, (Topic, _, _, x, y, dx, dy) in topics_rollup.iterrows():
            df = topics_combined[topics_combined.Topic==Topic]
            self.source['topics'] = pd.concat([
                self.source['topics'],
                treemap(df, "Weight", x, y, dx, dy, N=10)
            ])

        self.source['topics']["ytop"] = self.source['topics']['y'] + self.source['topics']['dy']
        self.source['topics'] = self.source['topics'].to_dict(orient='series')
        self.source['topics'] = ColumnDataSource(data=self.source['topics'])

        self.figure['topics'] = figure(
            width=width, height=height, tooltips="@Term", toolbar_location=None, tools="tap",
            x_axis_location=None, y_axis_location=None, title='Document Topics'
        )
        self.figure['topics'].x_range.range_padding = self.figure['topics'].y_range.range_padding = 0
        self.figure['topics'].grid.grid_line_color = None

        fill_color = factor_cmap("Topic", "MediumContrast5", topics_combined['Topic'].drop_duplicates())
        self.figure['topics'].block(
            'x', 'y', 'dx', 'dy', source=self.source['topics'], line_width=1, line_color="white",
            fill_alpha=0.8, fill_color=fill_color
        )

        self.figure['topics'].text(
            'x', 'y', x_offset=2, text="Topic", source=topics_rollup,
            text_font_size="18pt", text_color="white"
        )

        self.figure['topics'].text('x', 'ytop', x_offset=2, y_offset=2, text="Term", source=self.source['topics'],
            text_font_size="6pt", text_baseline="top",
        )

        self.source['topics'].selected.on_change('indices', self.selected_topic)