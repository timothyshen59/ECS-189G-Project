'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from local_code.base_class.dataset import dataset
import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')

        X_train, y_train, X_test, y_test = self.load_file(self.dataset_source_folder_path + self.dataset_file_name)

        return X_train, y_train, X_test, y_test

    def load_file(self, file_path):
        f = open(file_path, 'rb')
        data = pickle.load(f)
        f.close()

        X_train = np.array([instance['image'] for instance in data['train']], dtype=np.float32)
        y_train = np.array([instance['label'] for instance in data['train']], dtype=np.int64)

        X_test =  np.array([instance['image'] for instance in data['test']], dtype=np.float32)
        y_test = np.array([instance['label'] for instance in data['test']], dtype=np.int64)


        return X_train, y_train, X_test, y_test



