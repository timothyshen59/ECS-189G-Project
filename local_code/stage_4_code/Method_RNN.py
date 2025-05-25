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
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# referenced https://www.geeksforgeeks.org/implementing-recurrent-neural-networks-in-pytorch/

class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4
    input_size = 100
    num_layers = 2
    dropout = 0.2
    hidden_size = 100

    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        batch_size = x.size(0)
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        num_directions = 2 if self.rnn.bidirectional else 1

        h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        y_pred = self.fc(out[:, -1, :])
        return y_pred.squeeze(1)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.BCEWithLogitsLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        print("Label balance:", y.sum().item() / len(y))

        print("Input mean:", X.mean())

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


        train_loss_array = []
        train_accuracy_array = []
        train_f1_array = []
        train_precision_array = []
        train_recall_array = []

        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            epoch_loss = 0
            predictions = []
            labels = []

            print("Starting Epoch", epoch)

            for batch_X, batch_y in dataloader:
                print("Batch X")
                y_pred = self.forward(batch_X.float().to(next(self.parameters()).device))
                train_loss = loss_function(y_pred, batch_y.float())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)
                predictions.append(y_pred.detach())
                labels.append(batch_y.detach())


            y_true = torch.cat(labels, dim = 0)
            logits = torch.cat(predictions, dim=0)

            probs = torch.sigmoid(logits)
            y_pred = (probs >= 0.5).long()

            accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred}
            accuracy, f1, precision, recall = accuracy_evaluator.evaluate()

            train_loss_array.append(epoch_loss/len(dataset))
            train_accuracy_array.append(accuracy)
            train_f1_array.append(f1)
            train_precision_array.append(precision)
            train_recall_array.append(recall)

            print('Epoch:', epoch, 'Accuracy:', accuracy, 'F1-score:', f1, 'Precision:', precision, 'Recall:', recall, 'Loss:', train_loss.item())
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        prob = torch.sigmoid(y_pred)

        y_pred =  (prob >= 0.5).long()
        # Threshold at 0.5 to get class labels (0 or 1)
   # Threshold at 0.5 to get class labels (0 or 1)

        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
