import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class GNNLayer(Module):
    """One graph propagation layer: y = A * act(x W), with the propagated
    output A * y returned as well."""

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = torch.tanh
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=False):
        support = torch.mm(features, self.weight)
        if active:
            support = self.act(support)
        output = torch.spmm(adj, support)
        az = torch.spmm(adj, output)
        return output, az
