import pandas as pd

import data, vectorize

news = data.load()

convert = vectorize.transform()

convert.get_bigram(news.data_all['text'])