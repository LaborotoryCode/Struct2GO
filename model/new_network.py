import torch
import torch.nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv,GATConv, AvgPooling, MaxPooling
from model.layer import ConvPoolBlock, SAGPool, PLDDTWeightedGAT
from model.fusion import CrossAttentionFusion 


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


    def update_parent_features(self,label_network:dgl.DGLGraph, labels):
        # 获取图中的所有边
        edges = label_network.edges()

        second_dim_elements = labels[0,:]
        # 对于图中的每条边
        for child_idx, parent_idx in zip(edges[0], edges[1]):
            # 如果child节点的特征值大于parent节点的特征值
            if second_dim_elements[child_idx] > second_dim_elements[parent_idx]:
                # 更新parent节点的特征值为child节点的特征值
                second_dim_elements[parent_idx] = second_dim_elements[child_idx]
         # 更新labels的第二列为second_dim_elements
        labels[0, :] = second_dim_elements
        return labels
    

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
        """
        src, dst = graph.edges()
        graph.edata['weight'] = (
            (graph.ndata['plddt'][src] + graph.ndata["plddt"][dst]) / 2.0
        ).float()

        for i in range(self.num_convpools):
            graph, feat, readout = self.convpools[i](graph, feat)
            if final_readout is None:
                final_readout = readout
            else:
                final_readout = final_readout + readout
        """
        """
        seq = self.seq_proj(sequence_feature)
        struct = self.struct_proj(final_readout)

        seq_tok = seq.unsqueeze(1)                       # (B, 1, D)
        struct_tok = struct.unsqueeze(1)

        fused = self.cross_attn(seq_tok, struct_tok)
        fused = fused.squeeze(1)

        final_readout = torch.cat([seq, fused], dim=-1)
        """
        structure_proj = self.struct_proj(final_readout)
        sequence_proj = self.seq_proj(sequence_feature)
        
        graph.ndata["plddt"] = graph.ndata["plddt"].float()
        plddt = dgl.readout_nodes(graph, "plddt", op="mean")
        alpha = torch.sigmoid(self.fusion_gate(plddt))      # (B, 1)
        fused = alpha * structure_proj + (1 - alpha) * sequence_proj
        fused = fused.unsqueeze(1)          # (B, 1, hid_dim)
        sequence_proj = sequence_proj.unsqueeze(1)
        cross_attn_out = self.cross_attn(fused, sequence_proj)  # (B, 1, hid_dim)
        cross_attn_out = cross_attn_out.squeeze(1)
        #big_graph = torch.cat(all_readouts, dim=-1)
        final_readout = torch.cat((sequence_feature,cross_attn_out), -1)

        """
        # ---- Insert Transformer Encoder here ----
        # Transformer expects (seq_len, batch_size, feature_dim)
        tokens = final_readout.unsqueeze(1).transpose(0, 1)  # (1, B, hid_dim_total)
        tokens = self.transformer_encoder(tokens)           # (1, B, hid_dim_total)
        tokens = tokens.transpose(0, 1).squeeze(1)          # (B, hid_dim_total)
        final_readout = tokens
        # ----------------------------------------
        """

        feat = F.relu(self.lin1(final_readout))
        feat = F.dropout(feat, p=self.dropout, training=self.training)
        feat = F.relu(self.lin2(feat))
        #feat = F.log_softmax(self.lin3(feat), dim=-1)
        feat = self.lin3(feat)
        #feat = feat.t()

        """
        max_value,_ = torch.max(self.label_network1(label_network,feat),dim=1)
        feat -= F.relu(max_value)
        feat = feat.t()
        """

        # feat = self.update_parent_features(label_network, feat)
        #feat = self.line_new(final_readout)
        #feat = torch.sigmoid(feat)
        
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



def get_sag_network(net_type:str="hierarchical"):
    if net_type == "hierarchical":
        return SAGNetworkHierarchical
    elif net_type == "global":
        return SAGNetworkGlobal
    else:
        raise ValueError("SAGNetwork type {} is not supported.".format(net_type))