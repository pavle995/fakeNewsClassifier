import pandas as pd
from keras.preprocessing.text import Tokenizer


class Dataloader:

    @staticmethod
    def get_data(true_data_path, fake_data_path):
        # read date from csv, transform it to numpy and take only text
        true_data = pd.read_csv(true_data_path).to_numpy()[:100,1]
        fake_data = pd.read_csv(fake_data_path).to_numpy()[:100,1]

        return true_data, fake_data

    @staticmethod
    def preprocess_data(dataset):
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(dataset)

        one_hot_results = tokenizer.texts_to_matrix(dataset, mode='binary')
        #word_index = tokenizer.word_index
        print(one_hot_results)
        return one_hot_results
