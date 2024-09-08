import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from gpt.transformer import TransformerDecoder

to_np = lambda x: x.detach().cpu().numpy()

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.seq_len, config.embed_dim))

        self.dropout = nn.Dropout(config.drop)
        self.decoder = TransformerDecoder(config)

        self.seq_len = config.seq_len
        self.device = config.device

    def forward(self, idx):
        tok = self.tok_embed(idx)
        pos = self.pos_embed[:,:tok.shape[1],:]

        assert pos[0].shape == tok[0].shape
        x = self.dropout(tok + pos)
        logits = self.decoder(x)
        return logits

    def infer(self, input, batch=False):
        assert isinstance(input, np.ndarray)
        with torch.no_grad():
            x = torch.tensor(input, dtype=torch.long, device=self.device)
            if not batch:
                x = x.unsqueeze(0)
            y = self.forward(x)[:,-1] # last predict (B, N)
            if not batch:
                y = y.squeeze(0) # (N,)
            y = to_np(y)
        return y


class GPTConfig:
    def __init__(self):
        self.embed_dim = 512
        self.n_head = 8
        self.n_layer = 8
        self.drop = 0.1
        self.seq_len = 256
        self.n_vocab = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    config = GPTConfig()
    config.device = 'cpu'
    config.n_vocab = 10
    gpt = GPT(config)

    x = torch.randint(0, config.n_vocab, (64, config.seq_len))
    y = gpt(x)
    print('y.shape', y.shape)

    x = np.random.randint(0, config.n_vocab, (config.seq_len,))
    y = gpt.infer(x)
    print(y)

    x = np.random.randint(0, config.n_vocab, (64, config.seq_len))
    y = gpt.infer(x, batch=True)
    print(y.shape)
