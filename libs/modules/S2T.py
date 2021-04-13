import torch
import torch.nn as nn
from overrides import overrides

import torch
from torch_geometric.nn import GCNConv, GATConv, ASAPooling
import torch.nn.functional as F

from allennlp.common.registrable import Registrable
from allennlp.nn.util import masked_softmax
from torch_geometric.utils import dense_to_sparse

from allennlp.nn import util
from allennlp.modules.attention import AdditiveAttention

# attentive_weights = self._attention(
#             state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
#         )
#         # shape: (group_size, encoder_output_dim)
#         attentive_read = util.weighted_sum(
#             state["encoder_outputs"], attentive_weights)


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
class GCN(torch.nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super(GCN, self).__init__()
        self.conv1 = GATConv(hidden_size, hidden_size)
        self.conv2 = GATConv(hidden_size, hidden_size)

        self.pooling = ASAPooling(hidden_size)
        self.attention = AdditiveAttention(hidden_size, hidden_size)

    def forward(self, x, edge_index):
        #         x, edge_index = data.x, data.edge_index
        x = x.squeeze(0)
        # v = v.squeeze(0)
        # print("x", x.size())
        # print("v", v.size())
#         print(x.size())
#         print(v.size())
        # v = torch.tensor([v], device=x.device)
#         attentive_weights = self.attention(
#             v, x
#         )
#         attentive_read = util.weighted_sum(
#             x, attentive_weights)
        # print(attentive_weights.size())
        # print("res", attentive_read.size())
#         return attentive_read
        edge_index = edge_index + \
            torch.eye(edge_index.size(0)).to(device=x.device)
        edge_index, values = dense_to_sparse(edge_index)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

#         print(x.size())
        return x[-1].unsqueeze(0)


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.gcn = nn.Linear(hidden_size, hidden_size)
        self.gcn1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_embed_mat, A_matrix):
        # t*1*H,t*t

        A_matrix = A_matrix + \
            torch.eye(A_matrix.size(0)).to(device=tree_embed_mat.device)
        d = A_matrix.sum(1)
        D = torch.diag(torch.pow(d, -1))
        A = D.mm(A_matrix)
        tree_embed_mat = self.em_dropout(tree_embed_mat.squeeze(0))  # 1*t*H
#         print(A.size())
#         print(tree_embed_mat.size())

        new_tree_embed_mat = nn.functional.relu(self.gcn(A.mm(tree_embed_mat)))
        new_tree_embed_mat = nn.functional.relu(
            self.gcn1(A.mm(new_tree_embed_mat)))

        # print(new_tree_embed_mat.size())
        # print(new_tree_embed_mat.size())

        return new_tree_embed_mat  # t*H
