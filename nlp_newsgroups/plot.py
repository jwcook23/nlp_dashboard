from bokeh.plotting import figure
from bokeh.models import Div, ColumnDataSource, Slider

from nlp_newsgroups.data import data
from nlp_newsgroups.vectorize import vectorize

class plot(data, vectorize):

    def __init__(self):

        data.__init__(self)
        vectorize.__init__(self)

        self.get_titles()
        self.get_samples()
        self.get_ngrams()


    def get_titles(self):

        self.title_main = Div(
            text='Newsgroups NLP Dashboard',
            styles={'font-size': '150%', 'font-weight': 'bold'}, width=400
        )


    def get_samples(self):

        self.sample_title = Div(text='Example Documents', styles={'font-weight': 'bold'})
        self.sample_subtitle = Div(text='make a selection to display examples')
        self.sample_document = Div(
            text='', width=400, height=200
        )

        self.sample_number = Slider(start=0, end=1, value=0, step=1, title="Document Sample Number", width=150)
        self.sample_number.on_change('value_throttled', self.selected_sample)


    def set_samples(self, sample_title, sample_subtitle, text):

        self.sample_title.text = f'Example Documents: {sample_title}'
        self.sample_subtitle.text = sample_subtitle
        self.sample_number.end = len(text)-1
        self.sample_text = text
        self.selected_sample(None, None, 0)


    def selected_sample(self, attr, old, new):

        self.sample_document.text = self.sample_text.iloc[new]


    def selected_ngram(self, attr, old, new):

        sample_title = self.figure_ngram.title.text
        terms = self.source_ngram.data['y'].iloc[new]
        sample_subtitle = 'terms: '+','.join(terms.tolist())

        documents = self.features_ngram[:, terms.index].nonzero()[0]

        # TODO: highlight terms
        text = self.data_all.loc[documents,'text']

        self.set_samples(sample_title, sample_subtitle, text)


    def get_ngrams(self):

        self.get_ngram(self.data_all['text'], ngram_range=(1,2))
        ngram = self.summary_ngram.head(25).sort_values(by='term_count')

        self.figure_ngram = figure(
            y_range=ngram['terms'], height=300, title="One & Two Word Term Counts",
            toolbar_location=None, tools="tap", name='figure_ngram'
        )

        self.source_ngram = ColumnDataSource(data=dict({
            'y': ngram['terms'],
            'right': ngram['term_count']
        }))

        self.figure_ngram.hbar(source=self.source_ngram, width=0.9)

        self.source_ngram.selected.on_change('indices', self.selected_ngram)