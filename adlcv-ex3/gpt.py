import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class MaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.k_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        This function computes the multi-head self-attention of x with causal masking,
        so that each token only attends to previous tokens.
        """

        batch_size, seq_length, embed_dim = x.size()
        keys = self.k_projection(x)
        queries = self.q_projection(x)
        values = self.v_projection(x)

        # Rearrange keys, queries and values
        # from: batch_size x seq_length x embed_dim
        # to:   (batch_size x num_heads) x seq_length x head_dim
        keys = rearrange(
            keys, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )
        queries = rearrange(
            queries, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )
        values = rearrange(
            values, "b s (h d) -> (b h) s d", h=self.num_heads, d=self.head_dim
        )

        ####################### insert code here #########################################
        # Compute raw attention logits and scale them
        attention_logits = ...
        attention_logits = ...

        # Create a causal mask (upper-triangular with zeros on the diagonal)
        # so that each token can only attend to tokens at positions <= its own. HINT. torch.triu()
        mask = ...

        # Expand the mask to match attention_logits' shape: (batch_size * num_heads, seq_length, seq_length). HINT: torch.masked_fill()
        attention_logits = ...

        # Compute attention probabilities and weighted values
        attention = ...
        out = ...
        ###################################################################################

        # Rearrange output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(
            out, "(b h) s d -> b s (h d)", h=self.num_heads, d=self.head_dim
        )

        assert attention.size() == (batch_size * self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.attention = MaskedAttention(embed_dim=embed_dim, num_heads=num_heads)
        fc_hidden_dim = 4 * embed_dim if fc_dim is None else fc_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        residual = x
        x = self.layernorm1(x)  # LayerNorm BEFORE attention
        attention_out = self.attention(x)
        x = residual + attention_out  # residual AFTER
        x = self.dropout(x)

        # Feed-forward
        residual = x
        x = self.layernorm2(x)
        fc_out = self.fc(x)
        x = residual + fc_out
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0.0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        positions = self.pe(
            torch.arange(
                seq_length, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        positions = positions[None, :, :].expand(batch_size, seq_length, embed_dim)
        return x + positions


# -------------------- GPT Model --------------------
class AndersenGPT(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        max_seq_len,
        pos_enc="fixed",
        dropout=0.0,
        fc_dim=None,
        num_tokens=50_000,
    ):
        """
        The model outputs logits over the vocabulary for every token.
        """
        super().__init__()

        self.token_embedding = nn.Embedding(num_tokens, embed_dim)

        if pos_enc == "fixed":
            self.positional_encoding = PositionalEncoding(
                embed_dim=embed_dim, max_seq_len=max_seq_len
            )
        elif pos_enc == "learnable":
            self.positional_encoding = PositionalEmbedding(
                embed_dim=embed_dim, max_seq_len=max_seq_len
            )

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    fc_dim=fc_dim,
                    dropout=dropout,
                )
            )
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.ln_f = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, num_tokens, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of token indices of shape [batch_size, seq_length].
        Returns: Logits of shape [batch_size, seq_length, num_tokens] for next-token prediction.
        """
        tokens = self.token_embedding(x)
        tokens = self.positional_encoding(tokens)
        tokens = self.dropout(tokens)
        x = self.transformer_blocks(tokens)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
