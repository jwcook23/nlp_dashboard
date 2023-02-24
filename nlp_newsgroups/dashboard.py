# bokeh serve --show nlp_newsgroups/dashboard.py

# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

# TODO: save vectorizers and models (options to load different versions)

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
            output_file('tests/dashboard.html')
            show(self.layout)

        

    def generate_layout(self):

        self.layout = column(
            self.title_main,
            row(
                self.figure_ngram,
                self.figure_topics
            ),
            column(column(self.sample_title, row(self.sample_number, self.sample_subtitle)),self.sample_document)
        )

if __name__.startswith('bokeh_app'):
    page = dashboard()
elif __name__=='__main__':
    page = dashboard(server=False, standalone=True)