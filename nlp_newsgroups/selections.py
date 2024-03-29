import pandas as pd

from nlp_newsgroups.actions import actions

class selections(actions):

    def __init__(self):
        
        actions.__init__(actions)

    def activate_samples(self, event):

        option = self.sample_toggle.labels[self.sample_toggle.active]

        if option == 'All Documents':
            self.title['sample'].visible = False
            self.sample_number.visible = False
            self.sample_legend.visible = False
            self.default_samples()
        elif option == 'Document Samples':
            self.title['sample'].visible = True
            self.sample_number.visible = True
            self.sample_legend.visible = True
            self.selected_sample(None, None, 0, None, None)


    def selected_sample(self, attr, old, new, topic_weight, topic_terms):

        if self.sample_text is not None:

            text = self.sample_text.iloc[new]
            if topic_weight is None:
                document_idx = self.sample_text.index[self.sample_number.value]
            else:
                document_idx = None

            if self.sample_entity_labels is not None:
                text = self.highlight_entities(text, document_idx)

            text = self.surround_html_tag(text, self.model_params['stop_words'], 'stopwords')

            if self.sample_selected_terms is not None:
                text = self.surround_html_tag(text, self.sample_selected_terms, 'selected_terms')

            text = self.highlight_topics(text, document_idx, topic_weight, topic_terms)
       
            self.sample_document.text = text


    def selected_source(self, attr, old, row_source, fig_name):

        if fig_name=='Term Counts':
            self.selected_ngram(row_source)
        elif fig_name=='Entity Label':
            self.selected_entity_label(row_source)
        elif fig_name=='Entity Name':
            self.selected_entity(row_source)


    def selected_ngram(self, row_source):

        if len(row_source)==0:
            return
        
        self.default_selections(event='selected_ngram', ignore=pd.Series(['Term Counts']))

        sample_title = self.title['Term Counts'].text
        selected_terms = self.ngram['summary'].iloc[row_source]['terms']

        document_idx = self.ngram['features'][:, selected_terms.index].nonzero()[0]

        text = self.data_input[document_idx]

        # TODO: show distribution of term weight
        topic_terms = self.topic['terms']
        labeled_entity = self.entity['terms'].loc[
            self.entity['terms']['Document Index'].isin(document_idx)
        ]
        self.set_samples(sample_title, text, selected_terms, topic_terms, labeled_entity)


    def selected_entity_label(self, row_source):

        if len(row_source)==0:
            return    
        
        selected = self.source['Entity Label'].data['Terms'].iloc[row_source]
        entity = self.entity['summary'][
            self.entity['summary']['Entity Label'].isin(selected)
        ]

        self.set_entity(entity)


    def selected_entity(self, row_source):

        if len(row_source)==0:
            return
        
        selected_terms = self.source['Entity Name'].data['Terms'].iloc[row_source]
        document_idx = self.entity['terms'].loc[
            self.entity['terms']['Entity Clean Text'].isin(selected_terms), 'Document Index'
        ].drop_duplicates()
        labeled_entity = self.entity['terms'][
            self.entity['terms']['Document Index'].isin(document_idx)
        ]

        self.default_selections(event='selected_entity', ignore=pd.Series(['Entity Name','Entity Label']))

        sample_title = self.title['Entity Name'].text

        text = self.data_input[document_idx]

        # TODO: show distribution of term weight
        topic_terms = self.topic['terms']
        self.set_samples(sample_title, text, selected_terms, topic_terms, labeled_entity)


    def selected_topic(self, attr, old, new):
        
        if len(new)==0:
            return

        self.default_selections(event='selected_topic', ignore=pd.Series(['Topic Number']))

        self.topic_number = self.source['Topic Number'].data['Topic'].iloc[new].values[0]

        topic_documents = self.topic['weight'][
            (self.topic['weight']['Topic']==self.topic_number) & (self.topic['weight']['Rank']==1)
        ]

        document_idx = topic_documents.index
        topic_terms = self.topic['summary'].loc[
            (self.topic['summary']['Topic'].isin(topic_documents['Topic'])) & (self.topic['summary']['Weight']>0)
        ]

        text = self.data_input[document_idx]

        title = f'Selected Topic = {self.topic_number}'


        self.set_topic_term_weight(title, topic_terms)

        labeled_entity = self.entity['terms'].loc[
            self.entity['terms']['Document Index'].isin(document_idx)
        ]

        self.set_samples(title, text, None, labeled_entity)
