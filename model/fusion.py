import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # self.n_heads = n_heads
        # self.head_dim = d_model // n_heads

        print("d_model", d_model)

        self.query_proj = nn.Linear(512, d_model)
        self.key_proj = nn.Linear(512, d_model)
        self.value_proj = nn.Linear(512, d_model)

        self.out_proj = nn.Linear(1024, 1024) #hid_dim = 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_seq, X_struct):


        #try swapping below
        
        Q = self.query_proj(X_seq).unsqueeze(1)
        K = self.key_proj(X_struct).unsqueeze(1)
        V = self.value_proj(X_struct).unsqueeze(1)
        
        """
        Q = self.query_proj(X_struct).unsqueeze(1)
        K = self.key_proj(X_seq).unsqueeze(1)
        V = self.value_proj(X_seq).unsqueeze(1)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / (Q.size(-1) ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V).squeeze(1)
        return self.out_proj(output)
        # B, L, D = X_seq.size() #Batch size, Sequence length, Feature dimension

        # Q = self.query_proj(X_seq).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # K = self.key_proj(X_seq).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        # V = self.value_proj(X_seq).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn = torch.softmax(scores, dim=-1)
        # attn = self.dropout(attn)

        # out = torch.matmul(attn, V)
        # out = out.transpose(1,2).contiguous().view(B, L, D)
        # return self.out_proj(out)
    