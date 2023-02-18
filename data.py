from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np


class load():

    def __init__(self):

        self.train_ratio = 0.6
        self.validate_test_ratio = 0.2

        self.load_raw()
        self.split_data()


    def load_raw(self):

        raw = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes')
        )

        self.data_all = pd.DataFrame({
            'text': pd.Series(raw['data'], dtype='string'),
            'category_label_id': pd.Series(raw['target'], dtype='int8')
        })

        label_name = pd.DataFrame({
            'category_label_id': pd.Series(range(0, len(raw['target_names'])), dtype='int8'),
            'category_label_name': pd.Series(raw['target_names'], dtype='category')
        })

        self.data_all = self.data_all.merge(label_name, on='category_label_id')


    def split_data(self):

        train_size = int(self.train_ratio*len(self.data_all))
        validate_test_size = int(1-self.validate_test_ratio*len(self.data_all))

        self.data_train, self.data_validate, self.data_test = np.split(
            self.data_all.sample(frac=1, random_state=1), 
            [train_size, validate_test_size]
        )


if __name__ == '__main__':

    news = load()