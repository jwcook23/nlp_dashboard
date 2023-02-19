from bokeh.plotting import figure
from bokeh.models import Div

class plot():

    def __init__(self):

        self.get_titles()
        self.get_ngrams()


    def get_titles(self):

        self.title_main = Div(
            text='Newsgroups NLP Dashboard',
            styles={'font-size': '150%', 'font-weight': 'bold'}, width=400
        )


    def get_ngrams(self):

        fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
        counts = [5, 3, 4, 2, 4, 6]

        self.figure_ngram = figure(
            y_range=fruits, height=350, title="Fruit Counts",
                toolbar_location=None, tools=""
        )

        self.figure_ngram.hbar(y=fruits, right=counts, width=0.9)