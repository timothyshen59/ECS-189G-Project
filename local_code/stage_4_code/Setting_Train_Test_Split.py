'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np
from local_code.stage_1_code import Dataset_Loader
import torch

class Setting_Train_Test_Split(setting):
    # fold = 3

    def load_run_save_evaluate(self):
        data = torch.load('../../data/stage_4_data/classification_dataset.pt')

        # # load dataset

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}

        # run MethodModule
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None