import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, T, _ = query.size()
        Q = self.query(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.einsum("bhqk,bhvd->bhqd", attention_weights, V)
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out(attended_values)
