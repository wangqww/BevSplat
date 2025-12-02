# SelfAttention code from "https://github.com/autonomousvision/transfuser/blob/main/transfuser/model.py"
# Prakash, Aditya, Kashyap Chitta, and Andreas Geiger.
# "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving."
# Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        '''
        d_model: feature channel number
        max_len: sequence length
        '''
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)
        self.pe = pe

    def forward(self, x, dropout):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)].detach().to(x.device)
        return nn.Dropout(p=dropout)(x)

class SelfAttentionConv(nn.Module):
    def __init__(self, seq, n_embd, n_head):
        super(SelfAttentionConv, self).__init__()

        self.seq = seq
        self.n_embd = n_embd

        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.query = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.value = nn.Sequential(
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        # regularization
        # self.attn_drop = nn.Dropout(attn_pdrop)
        # self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Conv2d(n_embd, n_embd, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.n_head = n_head

        self.pe = PositionalEncoding(n_embd, self.seq)
        # input shape: [seq_len, batch_size, embedding_dim]

    def forward(self, x, attn_pdrop, resid_pdrop, pe_pdrop):
        B, S, C, H, W = x.size()
        assert C == self.n_embd
        x0 = x.permute(1, 0, 3, 4, 2).reshape(S, B * H * W, C)
        x1 = self.pe(x0, pe_pdrop).reshape(S, B, H, W, C).permute(1, 0, 4, 2, 3).reshape(B * S, C, H, W)

        x2 = x0.reshape(S, B, H, W, C).permute(1, 0, 4, 2, 3).reshape(B * S, C, H, W).reshape(B, S, self.n_head, C // self.n_head, H, W). \
            permute(0, 4, 5, 1, 2, 3).reshape(B * H * W, S, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x1).reshape(B, S, self.n_head, C // self.n_head, H, W).\
            permute(0, 4, 5, 1, 2, 3).reshape(B * H * W, S, self.n_head, C // self.n_head).transpose(1, 2) # [bhw, nh, S, hs]
        q = self.query(x1).reshape(B, S, self.n_head, C // self.n_head, H, W). \
            permute(0, 4, 5, 1, 2, 3).reshape(B * H * W, S, self.n_head, C // self.n_head).transpose(1, 2) # [bhw, nh, S, hs]

        # self-attend: (bhw, nh, T, hs) x (bhw, nh, hs, T) -> (bhw, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        att = nn.Dropout(attn_pdrop)(att)
        y = att @ x2  # (bhw, nh, T, T) x (bhw, nh, T, hs) -> (bhw, nh, T, hs)
        y = y.transpose(1, 2).contiguous().reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2).reshape(B * S, C, H, W)
        # re-assemble all head outputs side by side
        # output projection
        # y = self.resid_drop(self.proj(y)).reshape(B, S, C, H, W)
        y = y.reshape(B, S, C, H, W)

        return y, att


class TransformerFusion(nn.Module):
    def __init__(self, seq, n_embd, n_head, n_layers):
        super(TransformerFusion, self).__init__()

        self.nn_layers = nn.ModuleList()

        for idx in range(n_layers):
            self.nn_layers.append(
                SelfAttentionConv(seq, n_embd, n_head)
            )

    def forward(self, x, attn_pdrop, resid_pdrop, pe_pdrop):
        # B, S, C, H, W = x.shape
        #
        # y = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, S, C)
        for layer in self.nn_layers:
            x, att = layer(x, attn_pdrop, resid_pdrop, pe_pdrop) # [B, S, C, H, W]

        # y = torch.mean(x, dim=1)  # [B, C, H, W]
        # y = torch.max(x, dim=1)[0]  # [B, C, H, W]

        # y = y.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)
        # y = torch.max(y, dim=1)[0]  # [B, C, H, W]
        return x, att

        # att.shape = (bhw, nh, T, T)


