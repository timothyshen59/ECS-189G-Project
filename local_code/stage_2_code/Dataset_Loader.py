'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_train_file_name = None
    dataset_source_test_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        X_train, y_train = self.load_file(self.dataset_source_train_file_name)
        X_test, y_test = self.load_file(self.dataset_source_test_file_name)
        return X_train, y_train, X_test, y_test

    def load_file(self, file_name):
        print('loading data...')
        X = []
        y = []
        f = open(self.dataset_source_folder_path + file_name, 'r')
        for line in f:
            line = line.strip('\n')
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return X, y