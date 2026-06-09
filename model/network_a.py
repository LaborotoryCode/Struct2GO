import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv,GATConv, AvgPooling, MaxPooling
from model.layer import ConvPoolBlock, SAGPool, PLDDTWeightedGAT
from model.fusion import CrossAttentionFusion 


class GATGOControlNetwork(torch.nn.Module):

    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_layers:int=3, dropout: float = 0.5):

        super(GATGOControlNetwork, self).__init__()

        self.dropout = dropout
        self.gat_layers = torch.nn.ModuleList([
            GATConv(in_dim, hid_dim, num_heads=4, feat_drop=0.0,
                    attn_drop=0.0, residual=False, activation=None, allow_zero_in_degree=True),
            GATConv(hid_dim, hid_dim, num_heads=4, feat_drop=0.0,
                    attn_drop=0.0, residual=False, activation=None, allow_zero_in_degree=True),
            GATConv(hid_dim, hid_dim, num_heads=4, feat_drop=0.0,
                    attn_drop=0.0, residual=False, activation=None, allow_zero_in_degree=True),
        ])
        self.lin1 = torch.nn.Linear(hid_dim, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, graph: dgl.DGLGraph, sequence_feature=None, label_network=None):
        feat = graph.ndata["feature"]

        for gat in self.gat_layers:
            feat = gat(graph, feat).mean(dim=1)
            feat = F.relu(feat)
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        graph.ndata["h"] = feat
        graph_feat = dgl.mean_nodes(graph, "h")
        feat = F.relu(self.lin1(graph_feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        return self.lin2(feat)

class GATGONetwork(torch.nn.Module):

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int = 3, dropout: float = 0.5):

        super(GATGONetwork, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers

        self.gat_layers = torch.nn.ModuleList([
            GATConv(in_dim if i == 0 else hid_dim, hid_dim, num_heads=4, feat_drop=0.0,
                    attn_drop=0.0, residual=False, activation=None, allow_zero_in_degree=True)
            for i in range(num_layers)
        ])

        self.cross_attn = CrossAttentionFusion(d_model=hid_dim, dropout=dropout)
        self.seq_proj = torch.nn.Linear(1024, hid_dim)
        self.struct_proj = torch.nn.Linear(hid_dim, hid_dim)

        self.lin1 = torch.nn.Linear(hid_dim + 1024, hid_dim * 2)
        self.lin2 = torch.nn.Linear(hid_dim * 2, hid_dim * 2)
        self.lin3 = torch.nn.Linear(hid_dim * 2, out_dim)

    def forward(self, graph: dgl.DGLGraph, sequence_feature=None, label_network=None):
        feat = graph.ndata["feature"]

        for gat in self.gat_layers:
            feat = gat(graph, feat).mean(dim=1)
            feat = F.relu(feat)
            feat = F.dropout(feat, p=self.dropout, training=self.training)

        graph.ndata["h"] = feat
        struct_emb = dgl.mean_nodes(graph, "h")

        if sequence_feature is None:
            sequence_feature = torch.zeros(
                struct_emb.shape[0], 1024, device=struct_emb.device, dtype=struct_emb.dtype
            )

        structure_proj = self.struct_proj(struct_emb)
        sequence_proj = self.seq_proj(sequence_feature)

        fused = 0.5 * structure_proj + 0.5 * sequence_proj
        fused = fused.unsqueeze(1)
        sequence_proj = sequence_proj.unsqueeze(1)
        cross_attn_out = self.cross_attn(fused, sequence_proj)
        cross_attn_out = cross_attn_out.squeeze(1)

        final_readout = torch.cat((cross_attn_out, sequence_feature), dim=-1)

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        return self.lin3(feat)

class SAGNetworkHierarchical(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with hierarchical readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs:int=3,
                 pool_ratio:float=0.5, dropout:float=0.5):
        super(SAGNetworkHierarchical, self).__init__()

        #Feature fusion
        print("hid", hid_dim)

        self.cross_attn = CrossAttentionFusion(d_model=1024, dropout=dropout)#??
        self.seq_proj = torch.nn.Linear(1024, hid_dim)
        self.struct_proj = torch.nn.Linear(1024, hid_dim) #You can change the 64
        self.fusion_gate = torch.nn.Linear(1, hid_dim)

        convpools = []

        self.dropout = dropout
        self.num_convpools = num_convs+1 
        #Because of the new GAT, for the same number of convs this needs to be +1
       #self.classify = torch.nn.Linear(hid_dim, out_dim)

        #Apply pLDDT before pooling
        for i in range(self.num_convpools):
            if i == 0:
                convpools.append(PLDDTWeightedGAT(in_dim, hid_dim))
            else:
                convpools.append(ConvPoolBlock(hid_dim, hid_dim, pool_ratio=pool_ratio))

        #Normal
        # for i in range(num_convs):
        #     _i_dim = in_dim if i == 0 else hid_dim
        #     _o_dim = hid_dim
        #     convpools.append(ConvPoolBlock(_i_dim, _o_dim, pool_ratio=pool_ratio))

        self.convpools = torch.nn.ModuleList(convpools)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hid_dim * 2 + 1024, nhead=8), num_layers=6)

        self.lin1 = torch.nn.Linear(hid_dim * 2 + 1024, hid_dim*2)
        self.lin2 = torch.nn.Linear(hid_dim*2, hid_dim*2)
        self.lin3 = torch.nn.Linear(1024, out_dim)
        self.label_network1 = GATConv(128,1,num_heads=8,allow_zero_in_degree=True)

        self.line_new = torch.nn.Linear(hid_dim * 2 + 1024, out_dim)
    

    def forward(self, graph:dgl.DGLGraph, sequence_feature, label_network:dgl.DGLGraph):
        feat = graph.ndata["feature"]
        final_readout = None

        #Before pooling
        
        for i in range(self.num_convpools):
            if i == 0:
                src, dst = graph.edges()
                graph.edata['weight'] = (
                    (graph.ndata['plddt'][src] + graph.ndata["plddt"][dst]) / 2.0
                ).float()
                feat = self.convpools[i](graph, feat)
            else:
                graph, feat, readout = self.convpools[i](graph, feat)
                if final_readout is None:
                    final_readout = readout
                else:
                    final_readout = final_readout + readout
        

        #Normal
        
        final_readout = torch.cat((final_readout,sequence_feature), -1)
        #con_readout = self.transformer_encoder(final_readout)
        #final_readout = torch.cat((sequence_feature,con_readout), -1)
        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)
        #feat = feat.t()
        
        return feat



class SAGNetworkGlobal(torch.nn.Module):
    """The Self-Attention Graph Pooling Network with global readout in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The input node feature dimension.
        hid_dim (int): The hidden dimension for node feature.
        out_dim (int): The output dimension.
        num_convs (int, optional): The number of graph convolution layers.
            (default: 3)
        pool_ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        dropout (float, optional): The dropout ratio for each layer. (default: 0)
    """
    def __init__(self, in_dim:int, hid_dim:int, out_dim:int, num_convs=3,
                 pool_ratio:float=0.5, dropout:float=0.0):
        super(SAGNetworkGlobal, self).__init__()
        self.dropout = dropout
        self.num_convs = num_convs

        convs = []
        for i in range(num_convs):
            _i_dim = in_dim if i == 0 else hid_dim
            _o_dim = hid_dim
            convs.append(GraphConv(_i_dim, _o_dim))
        self.convs = torch.nn.ModuleList(convs)

        concat_dim = num_convs * hid_dim
        self.pool = SAGPool(concat_dim, ratio=pool_ratio)
        self.avg_readout = AvgPooling()
        self.max_readout = MaxPooling()

        self.lin1 = torch.nn.Linear(concat_dim * 2 + 1024, hid_dim)
        self.lin2 = torch.nn.Linear(hid_dim, hid_dim // 2)
        self.lin3 = torch.nn.Linear(hid_dim // 2, out_dim)
    
    def forward(self, graph:dgl.DGLGraph, sequence_feature):
        feat = graph.ndata["feature"]
        conv_res = []

        for i in range(self.num_convs):
            feat = self.convs[i](graph, feat)
            conv_res.append(feat)
        
        conv_res = torch.cat(conv_res, dim=-1)
        graph, feat, _ = self.pool(graph, conv_res)
        feat = torch.cat([self.avg_readout(graph, feat), self.max_readout(graph, feat)], dim=-1)
        feat = torch.cat((feat,sequence_feature),-1)
        feat = F.relu(self.lin1(feat))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)

        return feat



def get_sag_network(net_type:str="deepfri"):
    if net_type == "gatgo_control":
        return GATGOControlNetwork
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError("SAGNetwork type {} is not supported.".format(net_type))
