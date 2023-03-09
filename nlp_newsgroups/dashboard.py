# bokeh serve --show nlp_newsgroups/dashboard.py

# TODO: complaints data source
# https://www.consumerfinance.gov/data-research/consumer-complaints/

# TODO: ability to lookup words (topics they are assigned to and their importance)
# TODO: n-gram topic count in addition to document count (option to fit the above todo?)

# TODO: compare topic models
# TODO: ngram for stopwords

from bokeh.plotting import curdoc, output_file, show
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs, Div

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
        space = Div(width=10)

        self.layout = column(
            row(
                row(self.title['main'], column(self.input['recalculate'], self.input['save'], self.input['reset']), space),
                Tabs(tabs=[
                    TabPanel(child=row(*general_hyperparameters.values()), title='General Hyperparameters'),
                    TabPanel(child=row(*topic_hyperparameters.values()), title='Topic Hyperparameters')
                ])
            ),
            row(
                column(
                    self.title['terms'],
                    Tabs(tabs=[
                        TabPanel(child=column(self.input['axis_range']['ngram'], self.figure['ngram']), title='Term Count'),
                        TabPanel(child=column(self.input['axis_range']['entity'], self.figure['entity']), title='Entity Count'),
                    ])
                ),
                column(
                    self.title['topics'],
                    Tabs(tabs=[
                        TabPanel(child=column(
                            row(self.set_topic_name, self.input['topic_name'], self.input['topic_description']),
                            self.figure['topics']
                        ), title='Topic Summary'),
                        TabPanel(
                            child=row(
                                self.predict['calculate'],
                                self.predict['input'],
                                self.predict['figure']
                            ),
                            title='Topic Prediction'
                        )
                    ]),
                    column(
                        row(self.title['topic_distribution'], self.input['topic_distribution_range']), 
                        self.figure['topic_distribution']
                    )
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