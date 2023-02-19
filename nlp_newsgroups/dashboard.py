from bokeh.plotting import curdoc, output_file, show
from bokeh.layouts import row, column

from nlp_newsgroups.plot import plot

class dashboard(plot):

    def __init__(self, server=True):

        plot.__init__(self)

        self.generate_layout()

        if server:
            document = curdoc()
            document.add_root(self.layout)
        else:
            output_file('dashboard.html')
            show(self.layout)

        

    def generate_layout(self):
        
        self.layout = column(
            self.title_main,
            self.figure_ngram
        )


if __name__.startswith('bokeh_app'):
    page = dashboard()
elif __name__=='__main__':
    page = dashboard(server=False)