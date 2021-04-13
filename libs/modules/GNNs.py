from overrides import overrides

import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax


class GNNEncoder(torch.nn.Module, Registrable):
    """The base class for all GNN encoder modules.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError


@GNNEncoder.register("gcn")
class GNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x.squeeze()

    def get_output_dim(self):
        return self.conv2.out_channels
