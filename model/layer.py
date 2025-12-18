import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, AvgPooling, MaxPooling, GATConv,SumPooling,SAGEConv,ChebConv
from model.utils import topk, get_batch_id

class PLDDTWeightedGAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.attn_fc.data, gain=1.414)
        # self.W = nn.Linear(in_dim, out_dim, bias=False)
        # self.attn = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        #nn.init.xavier_uniform_(self.attn.data, gain=1.414) #Cuz why not

    # def edge_attention(self, edges):
    #     z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
    #     alpha = F.leaky_relu((z2 @ self.attn).squeeze(-1))
    #     weight = edges.data['weight'].squeeze(-1)
    #     alpha = alpha * weight.float()
    #     return{'e': alpha}
    
    # def message_func(self, edges):
    #     return {'z': edges.src['z'], 'e': edges.data['e']}
    
    # def reduce_func(self, nodes):
    #     attn = F.softmax(nodes.mailbox['e'], dim=1).unsqueeze(-1)
    #     h = torch.sum(attn * nodes.mailbox['z'], dim=1)
    #     return {'h': h}
    
    def forward(self, graph, features):
        
        graph = graph.local_var()
        h = self.fc(features)
        graph.ndata['h'] = h

        graph.apply_edges(lambda edges: {
            'e': F.leaky_relu(torch.matmul(torch.cat([edges.src['h'], edges.dst['h']], dim=1), self.attn_fc).squeeze(-1) * edges.data['weight'].squeeze(-1))
        })

        graph.edata['a'] = dgl.ops.edge_softmax(graph, graph.edata['e'])

        graph.update_all(
            fn.u_mul_e("h", "a", "m"), fn.sum("m", "h")
        )

        # graph = graph.local_var()
        # z = self.W(features)
        # graph.ndata['z'] = z
        # graph.apply_edges(self.edge_attention)
        # graph.update_all(self.message_func, self.reduce_func)

        return graph.ndata['h']

class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer1 = GraphConv(in_dim, 1)
        self.score_layer2 = GraphConv(in_dim, 1)
        self.non_linearity = non_linearity
        self.allow_zero_in_degree = True 
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):

        score1 = self.score_layer1(graph, feature).squeeze()
        score2 = self.score_layer2(graph, feature).squeeze()
        score  = (score1+score2)/2
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm
    
import torch.nn as nn

class ChunkedGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, chunk_size=4000, norm='both'):
        super().__init__()
        self.conv = dgl.nn.GraphConv(in_dim, out_dim, norm=norm)
        self.chunk_size = chunk_size

    def forward(self, graph, feat):
        # Output buffer (pre-allocated, prevents fragmentation)
        out = torch.zeros(
            graph.number_of_nodes(),
            self.conv._out_feats,
            device=feat.device,
            dtype=feat.dtype,
        )

        # compute in chunks — mathematically identical
        N = graph.number_of_nodes()
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            out[start:end] = self.conv(graph, feat)[start:end]

        return out


class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (mean||sum) readout operation.
    """
    def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5):
        super(ConvPoolBlock, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim)
        self.conv2 = GraphConv(out_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.sumpool = SumPooling()
        self.allow_zero_in_degree = True   
    
    def forward(self, graph, feature):
        out = F.relu(self.conv1(graph, feature))
        out = torch.reshape(out,(-1,512))
        out = F.relu(self.conv2(graph, out))
        out = torch.reshape(out,(-1,512))
        out = F.relu(self.conv2(graph, out))
        out = torch.reshape(out,(-1,512))
        graph, out, _ = self.pool(graph, out)
        g_out = torch.cat([self.maxpool(graph, out), self.maxpool(graph, out)], dim=-1)
        return graph, out, g_out 