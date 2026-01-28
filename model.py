import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization module.
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Attention(nn.Module):
    """
    Scaled Dot-Product Attention.
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            # Use -1e4 to avoid overflow/precision issues with -1e9
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # Apply attention
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of Multi-Head Attention and Feed-Forward network.
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class MXPT_Transformer(nn.Module): 
    """
    MXPT Transformer model for carbon estimation.
    """
    def __init__(self, seq_len=1500, xco2_channels=3, pos_channels=4, hidden=128, n_layers=4, attn_heads=8, out_channel=7, dropout=0.2, mode="Pre-train"):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.seq_len = seq_len
        self.feed_forward_hidden = hidden * 4

        # Encoding layers for XCO2 and positions
        self.encoding = nn.Sequential(
            nn.Linear(xco2_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.pos_encoding = nn.Sequential(
            nn.Linear(pos_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        # Transformer backbone
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)]
        )

        # Output heads
        self.recon_layer = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channel)
        )

        self.plume_identifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1) 
        )

        # Freeze backbone for downstream tasks (non-pre-train modes)
        if mode != "Pre-train":
            self.freeze(self.encoding)
            self.freeze(self.pos_encoding)
            self.freeze(self.transformer_blocks)
            self.freeze(self.recon_layer)

    def freeze(self, layer):
        """
        Freeze all parameters in a given layer.
        """
        for param in layer.parameters():
            param.requires_grad = False
    
    def forward(self, x, masked=None):
        # Split input into XCO2 features and position features
        x_xco2 = x[:, :, :3]
        x_p = x[:, :, 3:]

        if masked is None:
            masked = torch.zeros_like(x)

        # Create mask for attention mechanism (batch, 1, seq_len, seq_len)
        mask = (masked[:, :, 0] == 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # Embed features and add position encoding
        x = self.encoding(x_xco2)
        x = self.pos_encoding(x_p) + x    

        # Pass through Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)

        # Head 1: Reconstruction of XCO2 features
        x_rec = self.recon_layer(x)

        # Head 2: Plume regression
        plume_cls = self.plume_identifier(x)

        return x_rec, plume_cls
