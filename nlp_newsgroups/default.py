import pandas as pd
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

class default():

    def __init__(self):

        self.factors = {}


    def default_figures(self, event):

        self.default_ngram()
        self.default_topics_terms()
        self.default_topic_term_weight()
        self.default_topic_assignment()
        self.default_samples()
        self.default_topic_weight()


    def default_selections(self, event, ignore):

        self.sample_number.value = 0
        self.topic_number = None

        reset = pd.Series(self.source.keys())
        if ignore is not None:
            reset = reset[~reset.isin(ignore)]

        for source in reset:
            self.source[source].selected.indices = []

        self.default_topic_term_weight()
        if ignore is not None and not (ignore=='Entity Name').any():
            self.default_entity()


    def default_samples(self):

        self.title['sample'].text ='Example Documents'
        self.sample_number.title = 'Document Sample #: make selection'
        self.sample_legend.text = ''
        self.sample_document.text = ''
        self.sample_number.value = 0
        self.sample_number.high = 1
        self.sample_text = None

        self.sample_document.text = ''


    def default_ngram(self):

        ngram = self.ngram['summary']

        self.source['Term Counts'].data = {
            'Terms': ngram['terms'],
            'Term Count': ngram['term_count'],
            'Document Count': ngram['Document Count']
        }

        self.factors['Term Counts'] = ngram['terms'].tolist()

        self.input['axis_range']['Term Counts'].end = len(self.factors['Term Counts'])
        self.set_axis_range(None, None, self.input['axis_range']['Term Counts'].value, 'Term Counts', 50)


    def set_entity(self, entity):

        entity = entity.groupby('Entity Clean Text')
        entity = entity.agg({'Entity Count': sum, 'Document Count': sum})
        entity = entity.reset_index()
        entity = entity.sort_values(by='Entity Count', ascending=False)
        entity = entity.reset_index(drop=True)

        self.source['Entity Name'].data = {
            'Terms': entity['Entity Clean Text'],
            'Term Count': entity['Entity Count'],
            'Document Count': entity['Document Count']
        }
        self.source['Entity Name'].selected.indices = []

        self.factors['Entity Name'] = entity['Entity Clean Text'].tolist()

        self.input['axis_range']['Entity Name'].end = len(self.factors['Entity Name'])
        self.set_axis_range(None, None, self.input['axis_range']['Entity Name'].value, 'Entity Name', 30)  


    def default_entity(self):

        entity = self.entity['summary']
        self.set_entity(entity)
    

    def default_entity_label(self):

        entity = self.entity['terms']

        entity = entity.groupby('Entity Label')
        entity = entity.agg({'Entity Clean Text': 'nunique', 'Document Index': 'nunique'})
        entity = entity.rename(columns={'Entity Clean Text': 'Entity Count', 'Document Index': 'Document Count'})
        entity = entity.reset_index()
        entity = entity.sort_values(by='Entity Count', ascending=False)
        entity = entity.reset_index(drop=True)
        # TODO: rename for clarity (plot_term function also needs adjustent)
        self.source['Entity Label'].data = {
            'Terms': entity['Entity Label'],
            'Term Count': entity['Entity Count'],
            'Document Count': entity['Document Count']
        }

        self.factors['Entity Label'] = entity['Entity Label'].tolist()

        self.input['axis_range']['Entity Label'].end = len(self.factors['Entity Label'])
        self.set_axis_range(None, None, self.input['axis_range']['Entity Label'].value, 'Entity Label', 10)


    def default_terms(self, fig_name):

        if fig_name == 'Term Counts':
            self.default_ngram()
        elif fig_name == 'Entity Name':
            self.default_entity()
        elif fig_name == 'Entity Label':
            self.default_entity_label()


    def default_topic_assignment(self):

        self.input['topic_name'].title = 'Select to Rename'
        self.input['topic_name'].value = ''


    def default_topics_terms(self):

        source_data, source_text = self.topic_treemap()

        self.source['Topic Terms'].data = source_data
        self.source['Topic Number'].data = source_text.to_dict(orient='series')

        factors = self.source['Topic Terms'].data['Topic'].drop_duplicates().reset_index(drop=True)
        self.topic_color = factor_cmap("Topic", palette=Category10[10], factors=factors)


    def default_topic_term_weight(self):

        self.figure['Topic Distribution'].title.text = 'Select in Topic Summary or Predict Topic to Display'
        self.figure['Topic Distribution'].x_range.factors = []
        if self.figure['Topic Distribution'].renderers:
            self.figure['Topic Distribution'].renderers = []


    def default_topic_weight(self):

        source = self.topic['rollup']
        source = source.reset_index().to_dict(orient='list')
        self.source['Topic Weight'].data = source
