import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


class Dataloader:

    @staticmethod
    def get_data(true_data_path, fake_data_path):
        # read date from csv, transform it to numpy and take only text
        true_data = pd.read_csv(true_data_path).to_numpy()[:100,1]
        fake_data = pd.read_csv(fake_data_path).to_numpy()[:100,1]

        return true_data, fake_data

    @staticmethod
    def preprocess_data(true_data, fake_data):
        true_indices = np.arange(true_data.shape[0])
        fake_indices = np.arange(start=(true_indices[-1]+1),stop=true_data.shape[0]+fake_data.shape[0])
        indices = np.concatenate([true_indices, fake_indices])
        true_labels = np.zeros(true_data.shape[0])
        fake_labels = np.ones(fake_data.shape[0])
        labels = np.concatenate([true_labels, fake_labels])
        true_data = Dataloader.tokenize_data(true_data)
        fake_data = Dataloader.tokenize_data(fake_data)
        dataset = np.concatenate([true_data, fake_data])
        np.random.shuffle(indices)
        dataset = dataset[indices]
        labels = labels[indices]
        return dataset, labels

    @staticmethod
    def add_label(dataset, label):
        num_of_texts = dataset.shape[0]
        num_of_words = dataset.shape[1]
        labels = np.full((num_of_texts,1, 1), label)
        dataset = np.reshape(dataset, (num_of_texts, 1, num_of_words))
        print(dataset)
        return np.append(dataset, labels, axis=1)

    @staticmethod
    def tokenize_data(dataset):
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(dataset)

        sequences = tokenizer.texts_to_sequences(dataset)
        data = pad_sequences(sequences, 110)
        return data
