'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
import torch
import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        print("Loading Glove once...")
        self.tokenizer = get_tokenizer('basic_english')
        self.glove = GloVe(name='6B', dim=100)


    def load(self):
        print('loading data...')

        X_train, y_train, X_test, y_test = self.load_file()

        return X_train, y_train, X_test, y_test

    def load_file(self):
        X_train_neg_dir = '../../data/stage_4_data/text_classification/train/neg/'
        X_train_pos_dir = '../../data/stage_4_data/text_classification/train/pos/'

        X_test_neg_dir = '../../data/stage_4_data/text_classification/test/neg/'
        X_test_pos_dir = '../../data/stage_4_data/text_classification/test/pos/'

        X_train_neg, y_train_neg = self.retrieve_dataset(X_train_neg_dir, False)
        X_train_pos, y_train_pos = self.retrieve_dataset(X_train_pos_dir, True)

        X_test_neg, y_test_neg = self.retrieve_dataset(X_test_neg_dir, False)
        X_test_pos, y_test_pos = self.retrieve_dataset(X_test_pos_dir, True)

        X_train = X_train_pos + X_train_neg
        y_train = y_train_pos + y_train_neg

        X_test = X_test_pos + X_test_neg
        y_test = y_test_pos + y_test_neg

        X_train_padded = pad_sequence(X_train, batch_first=True)

        X_test_padded = pad_sequence(X_test, batch_first=True)

        torch.save({
            'X_train': X_train_padded,
            'y_train': torch.tensor(y_train, dtype=torch.long),
            'X_test': X_test_padded,
            'y_test': torch.tensor(y_test, dtype=torch.long)
        }, '../../data/stage_4_data/classification_dataset.pt')

        print(".pt file saved")
        return X_train, y_train, X_test, y_test

    def retrieve_dataset(self, file_dir, isPositive):
        y_value = 1 if isPositive else 0

        X = []
        y = []

        for file_name in os.listdir(file_dir):
            file_path = os.path.join(file_dir + file_name)

            embedding = self.retrieve_embedding(file_path)

            X.append(embedding)
            y.append(y_value)

        return X, y

    def retrieve_embedding(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = self.tokenizer(text)
        tokens = [word for word in tokens if word.isalpha()]

        indices = []
        for token in tokens:
            if token in self.glove.stoi:
                indices.append(self.glove.stoi[token])

        indices = indices[:256]

        embeddings = self.glove.vectors[torch.tensor(indices)]
        return embeddings


