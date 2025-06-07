from __future__ import division
from __future__ import print_function

import time
import argparse

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from torch import nn
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

SAVE_DIR = '../../result/stage_5_result/'
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=250,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# Modify Dataset Here
dataset = 'pubmed'
loader = Dataset_Loader(dName=dataset,dDescription="Citation Network")
loader.dataset_name = dataset
loader.dataset_source_folder_path = "../../data/stage_5_data/pubmed/"

adj, features, labels, idx_train, idx_val, idx_test = loader._load_data()
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    # global train_loss
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    predictions = output.max(1)[1]

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    precision_val = precision_score(labels[idx_val].cpu(), predictions[idx_val].cpu(),average='weighted')
    recall_val = recall_score(labels[idx_val].cpu(), predictions[idx_val].cpu(), average='weighted')
    f1_val = f1_score(labels[idx_val], predictions[idx_val], average='weighted')


    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'precision_val: {:.4f}'.format(precision_val),
          'recall_val: {:.4f}'.format(recall_val),
          'f1_val: {:.4f}'.format(f1_val),
          'time: {:.4f}s'.format(time.time() - t))

    # train_loss.append(loss_train.item())

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


#Train model and produce graph
# t_total = time.time()
# train_loss = []
# for epoch in range(args.epochs):
#     train(epoch)
#
# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# plt.plot(train_loss, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time (Per Epoch)')
# plt.legend()
# plt.grid(True)
# plt.savefig(SAVE_DIR + 'training_loss.png')
# plt.show()
#
# print(train_loss)
# # Testing
# test()

num_runs = 50
test_acc, test_loss = [], []

base_seed = 0

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}\n")

    run_seed = base_seed + run

    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)


    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)


    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        train(epoch)

    # Testing
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    test_acc.append(acc_test.item())
    test_loss.append(loss_test.item())

    print(f"Run {run + 1} Test Accuracy: {acc_test.item():.4f}")

avg_test_acc = np.mean(test_acc)

print("Average Test Accuracy Across 50 Runs", avg_test_acc * 100, "%")