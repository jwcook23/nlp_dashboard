# bokeh serve --show nlp_newsgroups/dashboard.py

# https://medium.com/plotly/nlp-visualisations-for-clear-immediate-insights-into-text-data-and-outputs-9ebfab168d5b
# https://towardsdatascience.com/introduction-to-topic-modeling-using-scikit-learn-4c3f3290f5b9
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py


# TODO: ability to name topics
# TODO: compare topic models
# TODO: named entity recognition
# TODO: ngram for stopwords

from bokeh.plotting import curdoc, output_file, show
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs

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

        general_hyperparameters = {
            key:val for key,val in self.model_inputs.items() 
            if key in ['token_pattern', 'max_df', 'min_df', 'stop_words', 'num_features', 'ngram_range']
        }
        topic_hyperparameters = {
            key:val for key,val in self.model_inputs.items() 
            if key in ['topic_num', 'topic_approach']
        }

        self.layout = column(
            row(
                column(self.title['main'], row(self.input_recalculate, self.input_reset)),
                Tabs(tabs=[
                    TabPanel(child=row(*general_hyperparameters.values()), title='General Hyperparameters'),
                    TabPanel(child=row(*topic_hyperparameters.values()), title='Topic Hyperparemeters')
                ])
            ),
            row(
                column(self.title['ngram'], self.figure['ngram']),
                column(
                    self.title['topics'],
                    Tabs(tabs=[
                        TabPanel(child=self.figure['topics'], title='Topic Summary'),
                        TabPanel(
                            child=row(
                                self.predict['calculate'],
                                self.predict['input'],
                                self.predict['figure']
                            ),
                            title='Topic Prediction'
                        )
                    ])
                )
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