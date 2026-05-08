import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # x: (B, T, C)
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # Split into heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        if mask is not None:
            # mask: (1, 1, T, T) or broadcastable to scores
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, H, T, T)

        # Weighted sum of V
        out = torch.matmul(attn, v)  # (B, H, T, D)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.out_proj(out)  # (B, T, C)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        # Pre-LN style
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
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

    def make_causal_mask(self, T, device):
        # Lower triangular mask
        # shape: (1, 1, T, T)
        mask = torch.tril(torch.ones(T, T, device=device))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        device = input_ids.device

        assert T <= self.max_len, "Sequence length exceeds max_len"

        positions = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)

        x = self.token_emb(input_ids) + self.pos_emb(positions)  # (B, T, C)

        mask = self.make_causal_mask(T, device)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)                # (B, T, C)
        logits = self.head(x)           # (B, T, vocab_size)
        return logits
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # batch shape: (B, T+1)
        batch = batch.to(device)

        # Input is all tokens except the last
        input_ids = batch[:, :-1]   # (B, T)

        # Target is all tokens except the first
        targets = batch[:, 1:]      # (B, T)

        logits = model(input_ids)   # (B, T, vocab_size)

        B, T, V = logits.shape

        # Flatten for cross entropy
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
import torch
from torch.utils.data import DataLoader, TensorDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    d_ff = 256
    num_layers = 2
    max_len = 32
    batch_size = 16
    num_epochs = 3
    lr = 1e-3

    # Dummy dataset
    # Each sample has length max_len + 1 so we can do input/target shift
    num_samples = 200
    data = torch.randint(0, vocab_size, (num_samples, max_len + 1))
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for (batch,) in dataloader:
            batch = batch.to(device)

            input_ids = batch[:, :-1]   # (B, T)
            targets = batch[:, 1:]      # (B, T)

            logits = model(input_ids)   # (B, T, vocab_size)
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()