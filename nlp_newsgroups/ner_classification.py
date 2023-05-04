import time
import pickle

import spacy
import pandas as pd

from nlp_newsgroups.data import data

news = data()

# load NER only
nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer"])


# Process whole documents
# texts = news.data_all.loc[0:10,'text'].tolist()
texts = news.data_all['text'].tolist()
docs = nlp.pipe(texts)

# # print(nlp.pipe_labels['ner'])


terms = pd.DataFrame()
time_start = time.time()
for idx,doc in enumerate(docs):
    document_idx = []
    entity_text = []
    entity_label = []
    start_char = []
    end_char = []
    for entity in doc.ents:
        document_idx += [idx]
        entity_text += [entity.text]
        entity_label += [entity.label_]
        start_char += [entity.start_char]
        end_char += [entity.end_char]
    df = pd.DataFrame({
        'Document Index': pd.Series(document_idx, dtype='int'),
        'Entity Raw Text': pd.Series(entity_text, dtype='object'),
        'Entity Label': pd.Series(entity_label, dtype='object'),
        'Start Character Index': pd.Series(start_char, dtype='int'),
        'End Character Index': pd.Series(end_char, dtype='int')
    })
    terms = pd.concat([terms, df], ignore_index=True)
time_end = time.time()
print(f'Finished in {time_end-time_start} seconds')

# BUG: multiple & duplicate entity_labels for entity_text per document
# multiple = terms.groupby(['Document Index','Entity Raw Text'])
# multiple = multiple.agg({'Entity Label': 'unique'})
# multiple = multiple[multiple['Entity Label'].str.len()>1]
# duplicate = self.entity['terms'].groupby(['Document Index','Entity Raw Text','Entity Label'])
# duplicate = duplicate.size()
# duplicate = duplicate[duplicate.size()>1]

terms['Entity Clean Text'] = terms['Entity Raw Text'].str.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True)
terms['Entity Clean Text'] = terms['Entity Clean Text'].str.replace(r'\s{2,}',' ', regex=True)
terms['Entity Clean Text'] = terms['Entity Clean Text'].str.strip()
terms['Entity Clean Text'] = terms['Entity Clean Text'].str.lower()
terms.loc[terms['Entity Clean Text'].str.len()==0, 'Entity Clean Text'] = pd.NA


summary = terms.groupby(['Entity Clean Text', 'Entity Label'])
summary = summary.agg({'Entity Raw Text': 'count', 'Document Index': 'nunique'})
summary = summary.rename(columns={'Entity Raw Text': 'Entity Count', 'Document Index': 'Document Count'})
summary = summary.reset_index()

file_name = 'model_ner.pkl'

with open(file_name, 'wb') as _fh:
    pickle.dump([terms, summary], _fh)

# with open(file_name, 'rb') as _fh:
#     terms, summary = pickle.load(_fh)