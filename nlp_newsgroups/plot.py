from bokeh.plotting import figure
from bokeh.models import Div

from nlp_newsgroups.data import data
from nlp_newsgroups.vectorize import vectorize

class plot(data, vectorize):

    def __init__(self):

        data.__init__(self)
        vectorize.__init__(self)

        self.get_titles()
        self.get_ngrams()


    def get_titles(self):

        self.title_main = Div(
            text='Newsgroups NLP Dashboard',
            styles={'font-size': '150%', 'font-weight': 'bold'}, width=400
        )


    def get_ngrams(self):

        ngram = self.get_ngram(self.data_all['text'], ngram_range=(1,2))
        ngram = ngram.head(50).sort_values(by='term_count')

        self.figure_ngram = figure(
            y_range=ngram['terms'], height=500, title="One & Two Word Term Counts",
            toolbar_location=None, tools=""
        )

        self.figure_ngram.hbar(y=ngram['terms'], right=ngram['term_count'], width=0.9)

        # TODO: display example documents when a term is selected
        # np.where(terms=='phone calls') TODO: how to find all not just the first?
        # document, _ = matrix[:,159].nonzero()