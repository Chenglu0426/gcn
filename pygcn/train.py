import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action = 'store_true', default = False)
parser.add_argument('--fastmode', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--weight_decay', type = float, default = 5e-4)
parser.add_argument('--hidden', type = int, default = 16)
parser.add_argument('--dropout', type = float, default = 0.5)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data()
model = GCN(nin = features.shape[1], nhid = args.hidden, nclass = labels.max().item() + 1, dropout = args.dropout)
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
if args.cuda():
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_val = idx_val.cuda()
    idx_train = idx_train.cuda()
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
    if not args.fastmode:
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t)
          )
def test():
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    print('test set results:', 'loss = {.4f}'.format(loss_test.item()), 'accuracy = {.4f}'.format(acc_test.item()))

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print('time used: {:.4f}'.format(time.time() - t_total))
test()

















