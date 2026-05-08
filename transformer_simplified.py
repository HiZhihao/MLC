import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, T, C)

        # split heads: (B, T, C) -> (B, H, T, D)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # attention scores: (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask: each position can only attend to itself and previous tokens
        mask = torch.tril(torch.ones(T, T, device=x.device))  # (T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)

        # weighted sum: (B, H, T, D)
        out = attn @ v

        # merge heads back: (B, H, T, D) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=256, num_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.max_len = max_len

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        assert T <= self.max_len

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)

        x = self.token_emb(input_ids) + self.pos_emb(pos)  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits


def training_step(model, batch, optimizer):
    """
    batch: (B, T+1)
    We shift by one token:
      input  = batch[:, :-1]
      target = batch[:, 1:]
    """
    model.train()

    x = batch[:, :-1]   # (B, T)
    y = batch[:, 1:]    # (B, T)

    logits = model(x)   # (B, T, V)
    B, T, V = logits.shape

    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        y.reshape(B * T)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()