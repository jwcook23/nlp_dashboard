# bokeh serve --show nlp_newsgroups/dashboard.py

# TODO: complaints data source
# https://www.consumerfinance.gov/data-research/consumer-complaints/

# TODO: better sample highlighting for topic terms and labeled entities
# TODO: generalize cross-filtering for documents for all figures?

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

        tab_options = Tabs(tabs=[
            TabPanel(
                child=column(
                    self.title['topics'],
                    row(self.set_topic_name, self.input['topic_name'], self.input['topic_description']),
                    self.figure['topics']
                ), title='Topic Summary'
            ),
            TabPanel(
                child=column(
                    self.predict['calculate'],
                    self.predict['input']
                ), title='Topic Prediction'
            ),
            TabPanel(
                child=row(
                    column(
                        row(self.title['entity_label'], self.input['axis_range']['entity_label']),
                        self.figure['entity_label'],
                    ),
                    column(
                        row(self.title['entity'], self.input['axis_range']['entity']), 
                        self.figure['entity']
                    )
                ), title='Named Entities'
            ),
            TabPanel(
                child=column(
                    row(self.title['ngram'], self.input['axis_range']['ngram']), 
                    self.figure['ngram']
                ), title='Term Counts'
            ),
            TabPanel(
                child=row(
                    row(column(self.input['recalculate'], self.input['save']), space),
                    Tabs(tabs=[
                        TabPanel(child=row(*general_hyperparameters.values()), title='Topic & Term Count Hyperparameters'),
                        TabPanel(child=row(*topic_hyperparameters.values()), title='Topic Only Hyperparameters')
                    ])
                ), title='Model Training'
            )
        ])


        self.layout = column(
            column(
                row(self.title['main'], self.input['reset']),
                tab_options,
                row(
                    column(self.figure['topic_confidence'], self.sample_toggle, self.sample_number),
                    column(self.input['topic_distribution_range'], self.figure['topic_distribution'])
                )
            ),
            column(
                row(
                    self.title['sample'],
                    self.sample_legend
                ),
                self.sample_document
            )
        )

if __name__.startswith('bokeh_app'):
    page = dashboard()
elif __name__=='__main__':
    page = dashboard(server=False, standalone=True)