'''
Concrete SettingModule class for a specific experimental SettingModule
'''
from local_code.base_class.dataset import dataset
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from local_code.stage_4_code import Dataset_Loader_Gen
import torch

class Setting_Train_Test_Split(setting):
    # fold = 3

    def load_run_save_evaluate(self):
        padded_X, padded_y = self.dataset.load()

        self.method.data = {'train': {'X': padded_X, 'y': padded_y}}

        # run MethodModule
        print(self.method.run())

