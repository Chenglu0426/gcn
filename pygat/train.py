import os
import random
import argparse
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import load_data, accuracy
from models import GAT

#training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 72)
parser.add_argument('--epochs', type = int, default = 10000)
parser.add_argument('--lr', type = float, default = 0.005)
parser.add_argument('--weight_decay', type = float, default = 5e-4)
parser.add_argument('--hidden', type = int, default = 8)
parser.add_argument('--nb_heads', type = int, default = 8)
parser.add_argument('--dropout', type = float, default = 0.6)
parser.add_argument('--alpha', type = float, default = 0.2)
parser.add_argument('--patience', type = int, default = 100)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()
model = GAT(nfeat = features.shape[1], nhid = args.hidden, nclasses = int(labels.max()) + 1, dropout = args.dropout, nheads = args.nb_heads, alpha = args.alpha)
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t)
          )
    return loss_val.data.item()

def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('test set results:',
          'loss = {:.4f}'.format(loss_test.data.item()),
          'accuracy = {:.4f}'.format(acc_test.data.item())
          )

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb < best_epoch:
        os.remove(file)

print('optimization finished!')
print('total time elapsed: {:.4f}s'.format(time.time() - t_total))

print('loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

compute_test()


    




















