import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, bias = True):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.FloatTensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        a = torch.mm(input, self.weight)
        output = torch.mm(adj, a)
        if self.bias:
            return output + self.bias
        else:
            return output




