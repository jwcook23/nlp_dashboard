import pandas as pd
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

class default():

    def __init__(self):

        self.factors = {}


    def default_figures(self, event):

        self.default_ngram()
        self.default_topics_terms()
        self.default_topic_term_importance()
        self.default_topic_assignment()
        self.default_samples()


    def default_selections(self, event, ignore):

        self.sample_number.value = 0
        self.topic_number = None

        reset = pd.Series(self.source.keys())
        if ignore is not None:
            reset = reset[~reset.isin(ignore)]

        for source in reset:
            self.source[source].selected.indices = []

        self.default_topic_term_importance()
        if ignore is not None and not (ignore=='entity').any():
            self.default_entity()


    def default_samples(self):

        self.title['sample'].text ='Example Documents'
        self.sample_number.title = 'Document Sample #: make selection'
        self.sample_legend.text = ''
        self.sample_document.text = ''
        self.sample_number.value = 0
        self.sample_number.high = 1
        self.sample_text = None


    def default_ngram(self):

        ngram = self.ngram['summary']

        self.source['ngram'].data = {
            'Terms': ngram['terms'],
            'Term Count': ngram['term_count'],
            'Document Count': ngram['document_count']
        }

        self.factors['ngram'] = ngram['terms'].tolist()

        self.input['axis_range']['ngram'].end = len(self.factors['ngram'])
        self.set_yaxis_range(None, None, self.input['axis_range']['ngram'].value, 'ngram', 25)


    def set_entity(self, entity):

        entity = entity.groupby('entity_clean')
        entity = entity.agg({'entity_count': sum, 'document_count': sum})
        entity = entity.reset_index()
        entity = entity.sort_values(by='entity_count', ascending=False)
        entity = entity.reset_index(drop=True)

        self.source['entity'].data = {
            'Terms': entity['entity_clean'],
            'Term Count': entity['entity_count'],
            'Document Count': entity['document_count']
        }
        self.source['entity'].selected.indices = []

        self.factors['entity'] = entity['entity_clean'].tolist()

        self.input['axis_range']['entity'].end = len(self.factors['entity'])
        self.set_yaxis_range(None, None, self.input['axis_range']['entity'].value, 'entity', 20)  


    def default_entity(self):

        entity = self.entity['summary']
        self.set_entity(entity)
    

    def default_entity_label(self):

        entity = self.entity['terms']

        entity = entity.groupby('entity_label')
        entity = entity.agg({'entity_clean': 'nunique', 'document_idx': 'nunique'})
        entity = entity.rename(columns={'entity_clean': 'entity_count', 'document_idx': 'document_count'})
        entity = entity.reset_index()
        entity = entity.sort_values(by='entity_count', ascending=False)
        entity = entity.reset_index(drop=True)
        # TODO: rename for clarity (plot_term function also needs adjustent)
        self.source['entity_label'].data = {
            'Terms': entity['entity_label'],
            'Term Count': entity['entity_count'],
            'Document Count': entity['document_count']
        }

        self.factors['entity_label'] = entity['entity_label'].tolist()

        self.input['axis_range']['entity_label'].end = len(self.factors['entity_label'])
        self.set_yaxis_range(None, None, self.input['axis_range']['entity_label'].value, 'entity_label', 5)


    def default_terms(self, fig_name):

        if fig_name == 'ngram':
            self.default_ngram()
        elif fig_name == 'entity':
            self.default_entity()
        elif fig_name == 'entity_label':
            self.default_entity_label()


    def default_topic_assignment(self):

        self.input['topic_name'].title = 'Select to Rename'
        self.input['topic_name'].value = ''


    def default_topics_terms(self):

        source_data, source_text = self.topic_treemap()

        self.source['topics'].data = source_data
        self.source['topic_number'].data = source_text.to_dict(orient='series')

        factors = self.source['topics'].data['Topic'].drop_duplicates().reset_index(drop=True)
        self.topic_color = factor_cmap("Topic", palette=Category10[10], factors=factors)


    def default_topic_term_importance(self):

        self.figure['topic_distribution'].title.text = 'Select in Topic Summary or Predict Topic to Display'
        self.figure['topic_distribution'].x_range.factors = []
        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].renderers = []


    def default_topic_confidence(self):

        source = self.topic['rollup'].rename(columns={'Weight': 'Confidence'})
        source = source.reset_index().to_dict(orient='list')
        self.source['topic_confidence'].data = source
