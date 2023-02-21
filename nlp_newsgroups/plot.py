from bokeh.plotting import figure
from bokeh.models import Div

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

        self.sample_document = Div(
            text='make a selection to display', width=400, height=200
        )


    def selected_ngram(self):

        # TODO: display example documents when a term is selected
        # TODO: how to find all not just the first?
        pass
        # np.where(terms=='phone calls') 
        # document, _ = matrix[:,159].nonzero()


    def get_ngrams(self):

        self.get_ngram(self.data_all['text'], ngram_range=(1,2))
        ngram = self.summary_ngram.head(25).sort_values(by='term_count')

        self.figure_ngram = figure(
            y_range=ngram['terms'], height=300, title="One & Two Word Term Counts",
            toolbar_location=None, tools=""
        )

        self.figure_ngram.hbar(y=ngram['terms'], right=ngram['term_count'], width=0.9)