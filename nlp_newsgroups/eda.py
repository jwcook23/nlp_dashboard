import pandas as pd

import data, vectorize

news = data.load()

convert = vectorize.transform()

bigram = convert.get_ngram(news.data_all['text'], ngram_range=(2,2))