from bokeh.transform import linear_cmap, factor_cmap
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


    def default_ngram(self, top_num=25):

        ngram = self.ngram['summary'].head(top_num)

        self.source['ngram'].data = {
            'Terms': ngram['terms'],
            'Term Count': ngram['term_count'],
            'Document Count': ngram['document_count']
        }

        self.figure['ngram'].y_range.factors = ngram['terms'].tolist()


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

        self.figure['topic_distribution'].title.text = 'Topic Term Importance (all terms): select topic to display'
        self.figure['topic_distribution'].x_range.factors = []
        if self.figure['topic_distribution'].renderers:
            self.figure['topic_distribution'].renderers = []
            self.figure['topic_distribution'].right = []
