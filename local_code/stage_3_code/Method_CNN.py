'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


SAVE_DIR = "../../result/stage_3_result/"
class Method_CNN(method, nn.Module):
    data = None
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, num_classes, dataset):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.dataset = dataset
        in_channels = 1
        if(dataset == 'CIFAR'): in_channels = 3
        # it defines the max rounds to train the model
        self.max_epoch = 10
        if(dataset == 'ORL'): self.max_epoch = 15
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html


        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=1),
            nn.Flatten(),
            nn.Dropout(0.2)  # Hyperparameter
        )
        if (dataset == 'ORL'):
            self.final_feature = nn.MaxPool2d(kernel_size=2, stride=2)
            self.final_fc = nn.Linear(32000, num_classes)
        elif (dataset == 'MNIST'):
            self.final_feature = nn.Identity()
            self.final_fc = nn.Linear(7744, num_classes)
        elif (dataset == 'CIFAR'):
            self.final_feature = nn.Identity()
            self.final_fc = nn.Linear(10816, num_classes)
    def forward(self, x):
        x = self.features(x)
        x= self.final_feature(x)
        x = self.classifier(x)
        y_pred = self.final_fc(x)
        return y_pred

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        train_loss_array = []
        train_accuracy_array = []
        train_f1_array = []
        train_precision_array = []
        train_recall_array = []

        if (self.dataset == 'ORL'):
            X_single_channel = X[:, :, :, 0]
            y = np.array(y) - 1
            X_tensor = torch.tensor(X_single_channel, dtype=torch.float32).unsqueeze(1)
            y_tensor = torch.tensor(y, dtype=torch.long)
        elif (self.dataset == 'MNIST'):
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            y_tensor = torch.tensor(y, dtype=torch.long)
        elif (self.dataset == 'CIFAR'):
            X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
            y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            epoch_loss = 0
            predictions = []
            labels = []

            print("Starting Epoch", epoch)

            for batch_X, batch_y in dataloader:
                y_pred = self.forward(torch.FloatTensor(np.array(batch_X)))
                train_loss = loss_function(y_pred, batch_y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)
                predictions.append(y_pred.detach())
                labels.append(batch_y.detach())



            y_true = torch.cat(labels, dim = 0)
            y_pred = torch.cat(predictions, dim = 0)

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
            accuracy, f1, precision, recall = accuracy_evaluator.evaluate()

            train_loss_array.append(epoch_loss/len(dataset))
            train_accuracy_array.append(accuracy)
            train_f1_array.append(f1)
            train_precision_array.append(precision)
            train_recall_array.append(recall)

            print('Epoch:', epoch, 'Accuracy:', accuracy, 'F1-score:', f1, 'Precision:', precision, 'Recall:', recall, 'Loss:', train_loss.item())


        plt.plot(train_loss_array, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Avg. Training Loss Over Time (Per Instance)')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR+'training_loss.png')
        plt.show()


        plt.plot(train_accuracy_array, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR + 'training_accuracy.png')
        plt.show()



        plt.plot(train_f1_array, label='Training F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training F1 Score Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR + 'training_f1_score.png')
        plt.show()



        plt.plot(train_precision_array, label='Training Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Training Precision Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR + 'training_precision.png')
        plt.show()



        plt.plot(train_recall_array, label='Training Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Training Recall Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR+'training_recall.png')
        plt.show()



    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}