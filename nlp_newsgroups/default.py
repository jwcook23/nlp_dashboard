from math import floor, ceil

from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

class default():

    def __init__(self):

        pass


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


    def set_ngram_range(self, attr, old, new):

        start = floor(new)
        end = ceil(new)+min(self.input_ngram_range.end, 25)

        self.figure['ngram'].y_range.factors = self.ngram_factors[start:end]
        self.figure['ngram'].yaxis[0].axis_label = f'Terms {start}-{end-1}'


    def default_ngram(self):

        ngram = self.ngram['summary']

        self.source['ngram'].data = {
            'Terms': ngram['terms'],
            'Term Count': ngram['term_count'],
            'Document Count': ngram['document_count']
        }

        self.ngram_factors = ngram['terms'].tolist()

        self.input_ngram_range.end = len(self.ngram_factors)
        self.set_ngram_range(None, None, self.input_ngram_range.value)


    def set_entity_range(self, attr, old, new):

        start = floor(new)
        end = ceil(new)+min(self.input_entity_range.end, 25)

        self.figure['entity'].y_range.factors = self.entity_factors[start:end]
        self.figure['entity'].yaxis[0].axis_label = f'Terms {start}-{end-1}'


    def default_entity(self):

        entity = self.entity['summary']

        self.source['entity'].data = {
            'Terms': entity['entity_clean'],
            'Term Count': entity['entity_count'],
            'Document Count': entity['document_count']
        }

        self.entity_factors = entity['entity_clean'].tolist()

        self.input_entity_range.end = len(self.entity_factors)
        self.set_entity_range(None, None, self.input_entity_range.value)


    def default_topic_assignment(self):

        self.input_topic_name.title = 'Select to Rename'
        self.input_topic_name.value = ''


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
