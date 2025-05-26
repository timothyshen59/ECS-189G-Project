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
from matplotlib import pyplot as plt

# referenced https://www.geeksforgeeks.org/implementing-recurrent-neural-networks-in-pytorch/
SAVE_DIR = '../../result/stage_4_result/'
CODE_DIR = '../../local_code/stage_4_code/'
DATA_DIR = '../../data/stage_4_data/text_generation/'

class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    input_size = 100
    num_layers = 2
    dropout = 0.2
    hidden_size = 256
    vocab_size = 0
    embedding_size = 256
    device = 'cpu'
    # it defines the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        vocab_data = torch.load(DATA_DIR + 'vocab_dict.pth')

        self.vocab_size = vocab_data['vocab_size']

        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.rnn = nn.GRU(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):

        embedded_x = self.embedding(x.long())

        out, (_,_) = self.rnn(embedded_x)
        output = self.fc(out)

        return output

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        train_loss_array = []


        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            self.train()

            epoch_loss = 0

            print("Starting Epoch", epoch)

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(next(self.parameters()).device)
                batch_y = batch_y.to(next(self.parameters()).device)

                y_pred = self.forward(batch_X.float())
                y_pred = y_pred[:, :-1, :]

                train_loss = loss_function(y_pred.reshape(-1, self.vocab_size), batch_y.view(-1))

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item() * batch_X.size(0)

            train_loss_array.append(epoch_loss/len(dataset))

        #Save Model
        torch.save(self.state_dict(), CODE_DIR + 'joke_rnn_weights.pth')


        plt.plot(train_loss_array, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Avg. Training Loss Over Time (Per Instance)')
        plt.legend()
        plt.grid(True)
        plt.savefig(SAVE_DIR + 'training_loss.png')
        plt.show()

    def test(self, starting_text, joke_length):
        self.eval()
        self.load_state_dict(torch.load(CODE_DIR + 'joke_rnn_weights.pth'))

        vocab_data = torch.load(DATA_DIR + 'vocab_dict.pth')
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']

        joke = []

        for start_text in starting_text:
            tokens = start_text.lower().split()
            input_ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]

            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            input = input_ids.copy()

            for _ in range(joke_length):
                with torch.no_grad():
                    output = self.forward(input_tensor)
                    logits = output[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()

                    if idx2word[next_id] == '<eos>':
                        break

                    input.append(next_id)
                    input_tensor = torch.tensor([[next_id]], dtype=torch.long).to(self.device)

            joke.append(' '.join([idx2word.get(tok, '<UNK>') for tok in input]))

        return joke

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')

        #Manually Change For Different Generated Jokes
        jokes = self.test(["What did the bartender say", "What gun do you use",  "What form of radiation bakes", "I like my slaves like", "What did the German air force"], 15)

        for i, joke in enumerate(jokes):
            print(f"Generated Joke {i + 1}: {joke}")

        return jokes









