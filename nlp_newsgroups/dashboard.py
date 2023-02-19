# bokeh serve --show nlp_newsgroups/dashboard.py

from bokeh.plotting import curdoc
from bokeh.layouts import row, column

from nlp_newsgroups.plot import plot

class dashboard(plot):

    def __init__(self):

        plot.__init__(self)

        self.generate_layout()

        self.document = curdoc()
        self.document.add_root(self.layout)

    def generate_layout(self):
        
        self.layout = column(
            self.title_main,
            self.figure_ngram
        )


if __name__.startswith('bokeh_app'):
    page = dashboard()