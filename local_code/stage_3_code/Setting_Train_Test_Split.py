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

        # load dataset
        # loaded_data = self.dataset.load()
        X_train, y_train, X_test, y_test = self.dataset.load()
        # X_train, y_train, X_test, y_test = train_test_split(loaded_data['X'], loaded_data['y'], test_size = 0.33)

        if(self.dataset.dataset_file_name=='ORL'):
            X_single_channel = X_test[:, :, :, 0]
            y_test = np.array(y_test) - 1
            X_tensor_test = torch.tensor(X_single_channel, dtype=torch.float32).unsqueeze(1)
            y_tensor_test = torch.tensor(y_test, dtype=torch.long)
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_tensor_test, 'y': y_tensor_test}}
        elif(self.dataset.dataset_file_name == 'MNIST'):
            X_tensor_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
            y_tensor_test = torch.tensor(y_test, dtype=torch.long)
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_tensor_test, 'y': y_tensor_test}}
        elif(self.dataset.dataset_file_name == 'CIFAR'):
            X_tensor_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
            y_tensor_test = torch.tensor(y_test, dtype=torch.long)
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_tensor_test, 'y': y_tensor_test}}

        # run MethodModule
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None