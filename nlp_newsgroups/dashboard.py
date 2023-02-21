# bokeh serve --show nlp_newsgroups/dashboard.py

from bokeh.plotting import curdoc, output_file, show
from bokeh.layouts import row, column

from nlp_newsgroups.plot import plot

class dashboard(plot):

    def __init__(self, server=True, standalone=False):

        plot.__init__(self)

        self.generate_layout()

        if server:
            doc = curdoc()
            doc.add_root(self.layout)
        elif standalone:
            output_file('dashboard.html')
            show(self.layout)

        

    def generate_layout(self):
        
        self.layout = column(
            self.title_main,
            self.sample_document,
            self.figure_ngram
        )


print(__name__)


if __name__.startswith('bokeh_app'):
    page = dashboard()
elif __name__=='__main__':
    page = dashboard(server=False)