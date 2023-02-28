# bokeh serve --show nlp_newsgroups/dashboard.py

# https://medium.com/plotly/nlp-visualisations-for-clear-immediate-insights-into-text-data-and-outputs-9ebfab168d5b
# https://towardsdatascience.com/introduction-to-topic-modeling-using-scikit-learn-4c3f3290f5b9
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

# TODO: standard output to div to explain model steps and timing

# TODO: input custom stopwords and seperate figure like ngram

# TODO: named entity recognition

# TODO: topic plot diagnostics
# - distribution of assigned topic confidence
# - predict new topic
# - documents that don't fit any topic (add up confidence values)

# TODO: recalcuate based on parameter changes of slicers

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
            row(
                column(self.title['main'], row(self.input_recalculate, self.input_reset)), 
                row(*self.inputs.values())
            ),
            row(
                column(self.title['ngram'], self.figure['ngram']),
                column(self.title['topics'], self.figure['topics'])
            ),
            column(
                row(
                    column(self.title['sample'], self.sample_number),
                    self.sample_legend
                ),
                self.sample_document
            )
        )

if __name__.startswith('bokeh_app'):
    page = dashboard()
elif __name__=='__main__':
    page = dashboard(server=False, standalone=True)