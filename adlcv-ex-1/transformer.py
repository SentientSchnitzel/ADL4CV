import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

def to_device(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        This function computes the multi-head self-attention of x.
        """
        batch_size, seq_length, embed_dim = x.size()

        # Generate keys, queries, and values
        keys    = self.k_projection(x) # B x seq_len x embed_dim
        queries = self.q_projection(x) # B x seq_len x embed_dim
        values  = self.v_projeciton(x) # B x seq_len x embed_dim

        """
        Now you have to split the projected keys, queries, and values to multiple heads.
        """
        # First split the embed_dim to num_heads x head_dim
        keys = rearrange(keys, 'b s (numH Hdim) -> b s numH Hdim', numH=self.num_heads,Hdim=self.head_dim)
        # Secondly merge the batch_size with the num_heads
        keys = rearrange(keys, 'b s numH Hdim -> (b numH) s Hdim', numH=self.num_heads,Hdim=self.head_dim)
        
        # HINT repeat the same process for queries and values
        queries = rearrange(queries, 'b s (numH Hdim) -> b s numH Hdim', numH=self.num_heads,Hdim=self.head_dim)
        queries = rearrange(queries, 'b s numH Hdim -> (b numH) s Hdim', numH=self.num_heads,Hdim=self.head_dim)

        values = rearrange(values, 'b s (numH Hdim) -> b s numH Hdim', numH=self.num_heads,Hdim=self.head_dim)
        values = rearrange(values, 'b s numH Hdim -> (b numH) s Hdim', numH=self.num_heads,Hdim=self.head_dim)

        # Compute attetion logits
        attention_logits = torch.matmul(queries, rearrange(keys, 'b s d -> b d s')) # multiply queries and keys
        attention_logits = attention_logits * self.scale
        attention = F.softmax(attention_logits) # softmax on attention
        out = torch.matmul(attention, values) # multiply attention with values

        # Rearragne output
        # from (batch_size x num_head) x seq_length x head_dim to batch_size x seq_length x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_length, seq_length)
        assert out.size() == (batch_size, seq_length, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        
        # get half of the embedding indices
        div_term = torch.arange(0., embed_dim, 2)
        # miltiply each position with -(math.log(10000.0) / embed_dim)
        div_term = div_term * -(math.log(10000.0) / embed_dim)
        # compute the exp of div_term
        div_term = torch.exp(div_term)
        pe[:, ::2] = torch.sin(position/div_term) # HINT use torch.sin to assign to the even-positions the position * div_term
        pe[:, 1::2] = torch.cos(position/div_term) # HINT use torch.cos to assign to the odd-positions the position * div_term

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        return x + self.pe[:, :seq_length]

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):

        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(embedding_dim=embed_dim, num_embeddings=max_seq_len)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        positions = self.pe(torch.arange(seq_length, device=to_device()))
        positions = positions[None, :, :].expand(batch_size, seq_length, embed_dim)
        return x + positions
        

class TransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None, num_tokens=50_000, num_classes=2, 
                 
    ):
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']
        

        self.pool, self.pos_enc, = pool, pos_enc
        self.token_embedding = nn.Embedding(embedding_dim=embed_dim, num_embeddings=num_tokens)

        # Initialize cls token parameter
        if self.pool == 'cls':
            #self.cls_token = ... # Institate an nn.Parameter with size: 1,1,embed_dim
            max_seq_len +=1
        
        if self.pos_enc == 'fixed':
            self.positional_encoding = PositionalEncoding(embed_dim=embed_dim, max_seq_len=max_seq_len)
        elif self.pos_enc == 'learnable':
            self.positional_encoding = PositionalEmbedding(embed_dim=embed_dim, max_seq_len=max_seq_len)

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        tokens = self.token_embedding(x)
        batch_size, seq_length, embed_dim = tokens.size()

        # Include cls token in the input sequence
        ####################### insert code here #######################
        if self.pool == 'cls':
            # HINT: repeat the cls token of the batch dimension
            pass
        ################################################################

        x = self.positional_encoding(tokens)
        x = self.dropout(x)
        x = self.transformer_blocks(x)

        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
            
        # Get cls token
        ####################### insert code here #######################
        elif self.pool == 'cls':
            # HINT: get the first output token of the transfomer.
            pass
        ################################################################

        return self.classifier(x)