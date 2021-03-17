from dataloader.Dataloader import Dataloader
import pandas as pd

FAKE_DATA_PATH = './data/Fake.csv'
TRUE_DATA_PATH = './data/True.csv'

pd.set_option('display.max_columns', 4)

if __name__ == '__main__':
    true_data, fake_data = Dataloader.get_data(TRUE_DATA_PATH, FAKE_DATA_PATH)
    true_data_tokenized = Dataloader.preprocess_data(true_data, fake_data)

