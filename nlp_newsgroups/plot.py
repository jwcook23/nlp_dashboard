import re
from operator import itemgetter
import os
import pickle

from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, Slider, ColorBar
from bokeh.transform import linear_cmap, factor_cmap
import pandas as pd
from squarify import normalize_sizes, squarify


from nlp_newsgroups.data import data
from nlp_newsgroups.model import model

class plot(data, model):

    def __init__(self):

        data.__init__(self)

        self.model_cache()

        self.plot_titles()
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

    def default_samples(self):

        self.sample_title.text ='Example Documents'
        self.sample_subtitle.text = 'make a selection to display examples'
        self.sample_document.text = ''
        self.sample_number.value = 0
        self.sample_number.end = 1
        self.sample_text = None


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

        # highlight matching term in bold and underline
        pattern = '|'.join(terms)
        highlight = r'<u><strong>\g<0></strong></u>'
        text = text.str.replace(rf'(?i)(?:{pattern})', highlight, flags=re.IGNORECASE, regex=True)

        self.sample_title.text = f'Example Documents: {sample_title}'
        self.sample_subtitle.text = sample_subtitle
        self.sample_number.end = len(text)-1
        self.sample_text = text
        self.sample_devectorized = devectorized
        self.selected_sample(None, None, 0)


    def selected_sample(self, attr, old, new):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]

            # TODO: highlight features, or remove sample_devectorized
            pattern = '|'.join(self.sample_devectorized[new])
            text = re.sub(rf'(?i)(?:{pattern})', r'<strong>\g<0></strong>', text, re.IGNORECASE)

            self.sample_document.text = text


    def selected_ngram(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return

        sample_title = self.figure_ngram.title.text
        terms = self.source_ngram.data['y'].iloc[new]
        sample_subtitle = 'terms: '+','.join(terms.tolist())

        self.set_samples(sample_title, sample_subtitle, terms, self.ngram['features'], self.ngram['devectorized'])


    def plot_ngram(self, top_num=25):

        ngram = self.ngram['summary'].head(top_num).sort_values(by='term_count')

        self.figure_ngram = figure(
            y_range=ngram['terms'], height=500, width=400, toolbar_location=None, tools="tap", 
            title="One & Two Word Term Counts", x_axis_label='Term Count', y_axis_label='Term'

        )

        self.source_ngram = ColumnDataSource(data=dict({
            'y': ngram['terms'],
            'right': ngram['term_count'],
            'color': ngram['document_count']
        }))

        cmap = linear_cmap(
            field_name='color', palette='Turbo256', 
            low=min(ngram['document_count']), high=max(ngram['document_count'])
        )
        color_bar = ColorBar(color_mapper=cmap['transform'], title='Document Count')

        self.figure_ngram.hbar(
            source=self.source_ngram, width=0.9, fill_color=cmap, line_color=None
        )
        self.figure_ngram.add_layout(color_bar, 'above')   

        self.source_ngram.selected.on_change('indices', self.selected_ngram)


    def selected_topic(self, attr, old, new):

        if len(new) == 0:
            self.default_samples()
            return

        sample_title = self.figure_topics.title.text
        terms = self.source_topics.data['Term'].iloc[new]
        terms = self.topic['terms'][self.topic['terms'].isin(terms)]
        sample_subtitle = 'terms: '+','.join(terms.tolist())

        self.set_samples(sample_title, sample_subtitle, terms, self.topic['features'], self.topic['devectorized'])


    def plot_topics(self, top_num=10):
        
        topics_combined = self.topic['summary'][self.topic['summary']['Rank']<top_num].copy()
        topics_combined['Topic'] = 'Topic '+topics_combined['Topic'].astype('str')

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

        self.source_topics = pd.DataFrame()
        for _, (Topic, _, _, x, y, dx, dy) in topics_rollup.iterrows():
            df = topics_combined[topics_combined.Topic==Topic]
            self.source_topics = pd.concat([
                self.source_topics,
                treemap(df, "Weight", x, y, dx, dy, N=10)
            ])

        self.source_topics["ytop"] = self.source_topics['y'] + self.source_topics['dy']
        self.source_topics = self.source_topics.to_dict(orient='series')
        self.source_topics = ColumnDataSource(data=self.source_topics)

        self.figure_topics = figure(
            width=width, height=height, tooltips="@Term", toolbar_location=None, tools="tap",
            x_axis_location=None, y_axis_location=None, title='Document Topics'
        )
        self.figure_topics.x_range.range_padding = self.figure_topics.y_range.range_padding = 0
        self.figure_topics.grid.grid_line_color = None

        fill_color = factor_cmap("Topic", "MediumContrast5", topics_combined['Topic'].drop_duplicates())
        self.figure_topics.block(
            'x', 'y', 'dx', 'dy', source=self.source_topics, line_width=1, line_color="white",
            fill_alpha=0.8, fill_color=fill_color
        )

        self.figure_topics.text(
            'x', 'y', x_offset=2, text="Topic", source=topics_rollup,
            text_font_size="18pt", text_color="white"
        )

        self.figure_topics.text('x', 'ytop', x_offset=2, y_offset=2, text="Term", source=self.source_topics,
            text_font_size="6pt", text_baseline="top",
        )

        self.source_topics.selected.on_change('indices', self.selected_topic)