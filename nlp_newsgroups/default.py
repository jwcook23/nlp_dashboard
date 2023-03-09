from math import floor, ceil

from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

class default():

    def __init__(self):

        self.factors = {}


    def default_figures(self, event):

        self.default_ngram()
        self.default_topics_terms()
        self.default_topics_distribution()
        self.default_topic_assignment()
        self.default_samples()


    def default_selections(self, ignore=None):

        self.sample_number.value = 0
        self.topic_number = None

        reset = list(self.source.keys())
        if ignore is not None:
            reset.remove(ignore)

        for source in reset:
            self.source[source].selected.indices = []

        self.default_topics_distribution()


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
        self.set_yaxis_range(None, None, self.input['axis_range']['ngram'].value, 'ngram')


    def default_entity(self):

        entity = self.entity['summary']

        # TODO: graph for filtering by label type
        # self.entity['summary']['entity_label'].value_counts()

        entity = entity.groupby('entity_clean')
        entity = entity.agg({'entity_count': sum, 'document_count': sum})
        entity = entity.reset_index()
        entity = entity.sort_values(by='entity_count', ascending=False)

        self.source['entity'].data = {
            'Terms': entity['entity_clean'],
            'Term Count': entity['entity_count'],
            'Document Count': entity['document_count']
        }

        self.factors['entity'] = entity['entity_clean'].tolist()

        self.input['axis_range']['entity'].end = len(self.factors['entity'])
        self.set_yaxis_range(None, None, self.input['axis_range']['entity'].value, 'entity')


    def default_terms(self, figname):

        if figname == 'ngram':
            self.default_ngram()
        elif figname == 'entity':
            self.default_entity()


    def default_topic_assignment(self):

        self.input['topic_name'].title = 'Select to Rename'
        self.input['topic_name'].value = ''


    def default_topics_terms(self):

        source_data, source_text = self.topic_treemap()

        self.source['topics'].data = source_data
        self.source['topic_number'].data = source_text.to_dict(orient='series')

        factors = self.source['topics'].data['Topic'].drop_duplicates().reset_index(drop=True)
        self.topic_color = factor_cmap("Topic", palette=Category10[10], factors=factors)


    def default_topics_distribution(self):

        self.figure['topic_distribution'].x_range.factors = []
        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].renderers = []
            self.figure['topic_distribution'].right = []
