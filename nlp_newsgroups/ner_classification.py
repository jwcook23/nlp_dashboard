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
        'document_idx': pd.Series(document_idx, dtype='int'),
        'entity_text': pd.Series(entity_text, dtype='object'),
        'entity_label': pd.Series(entity_label, dtype='object'),
        'start_char': pd.Series(start_char, dtype='int'),
        'end_char': pd.Series(end_char, dtype='int')
    })
    terms = pd.concat([terms, df], ignore_index=True)
time_end = time.time()
print(f'Finished in {time_end-time_start} seconds')

# BUG: multiple & duplicate entity_labels for entity_text per document
# multiple = terms.groupby(['document_idx','entity_text'])
# multiple = multiple.agg({'entity_label': 'unique'})
# multiple = multiple[multiple['entity_label'].str.len()>1]
# duplicate = self.entity['terms'].groupby(['document_idx','entity_text','entity_label'])
# duplicate = duplicate.size()
# duplicate = duplicate[duplicate.size()>1]

terms['entity_clean'] = terms['entity_text'].str.replace(r'[^a-zA-Z0-9 ]', ' ', regex=True)
terms['entity_clean'] = terms['entity_clean'].str.replace(r'\s{2,}',' ', regex=True)
terms['entity_clean'] = terms['entity_clean'].str.strip()
terms['entity_clean'] = terms['entity_clean'].str.lower()
terms.loc[terms['entity_clean'].str.len()==0, 'entity_clean'] = pd.NA


summary = terms.groupby(['entity_clean', 'entity_label'])
summary = summary.agg({'entity_text': 'count', 'document_idx': 'nunique'})
summary = summary.rename(columns={'entity_text': 'entity_count', 'document_idx': 'document_count'})
summary = summary.reset_index()

file_name = 'model_ner.pkl'

with open(file_name, 'wb') as _fh:
    pickle.dump([terms, summary], _fh)

# with open(file_name, 'rb') as _fh:
#     terms, summary = pickle.load(_fh)