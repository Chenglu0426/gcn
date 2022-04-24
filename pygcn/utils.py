import numpy as np
import scipy.sparse as sp
import torch

def onehot(label):
    classes = set(label)
    class_dict = {c: np.eye(len(classes))[i, :] for i, c in enumerate(classes)}
    label_onehot = np.array(list(map(class_dict.get, label)))
    return label_onehot

def load_data(path = '../data/cora/', dataset = 'cora'):
    print(f'loading {dataset} dataset...')
    data = np.genfromtxt('{}{}.context'.format(path, dataset))
    features = sp.csr_matrix(data[:, 1: -1], dtype = np.float32)
    labels = onehot(data[:, -1])

    idx = np.array(data[:, 0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edge = np.genfromtxt('{}{}.cite'.format(path, dataset))
    edge_ordered = np.array(list(map(idx_map.get, edge.flatten()))).reshape(edge.shape)
    adj = sp.coo_matrix((np.ones(len(edge.shape[0])), (edge_ordered[:, 0], edge_ordered[:, 1])), shape = (labels.shape[0], labels.shape[0]))

    adj = adj + adj.T
    adj[adj > 1] = 1
    features =


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype = np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat = sp.diags(r_inv)
    mx = r_mat.dot(mx)
    return mx



