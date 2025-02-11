import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def positional_encoding_2d(nph, npw, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(nph), torch.arange(npw), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

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

        batch_size, seq_len, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projeciton(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        attention_logits = torch.matmul(queries, keys.transpose(1, 2))
        attention_logits = attention_logits * self.scale
        attention = F.softmax(attention_logits, dim=-1)
        out = torch.matmul(attention, values)

        # Rearragne output
        # from (batch_size x num_head) x seq_len x head_dim to batch_size x seq_len x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_len, seq_len)
        assert out.size() == (batch_size, seq_len, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
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

class ViT(nn.Module):
    def __init__(self, image_size, channels, patch_size, embed_dim, num_heads, num_layers,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None, num_classes=2, ):
        
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']

        self.pool, self.pos_enc, = pool, pos_enc

        H, W = image_size
        patch_h, patch_w = patch_size
        assert H % patch_h == 0 and W % patch_w == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (H // patch_h) * (W // patch_w)
        patch_dim = channels * patch_h * patch_w

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.rand(1,1,embed_dim))
            num_patches += 1
        
        # TASK: Implement patch embedding layer 
        #       Convert imaged to patches and project to the embedding dimension
        # HINT: 1) Use the Rearrange layer from einops.layers.torch 
        #          in the same way you used the rearrange function 
        #          in the image_to_patches function (playground.py)
        #       2) Stack Rearrange layer with a linear projection layer using nn.Sequential
        #          Consider including LayerNorm layers before and after the linear projection
        ######## insert code here ########
        #
        #
        #
        #
        #
        #################################

        if self.pos_enc == 'learnable':
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        elif self.pos_enc == 'fixed':
            self.positional_embedding = positional_encoding_2d(
                nph = H // patch_h, 
                npw = W // patch_w,
                dim = embed_dim,
            )  

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)


    def forward(self, img):

        tokens = self.to_patch_embedding(img)
        batch_size, num_patches, embed_dim = tokens.size()
        
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=batch_size)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            num_patches+=1
        
        positions =  self.positional_embedding.to(img.device, dtype=img.dtype)
        if self.pos_enc == 'fixed' and self.pool=='cls':
            positions = torch.cat([torch.zeros(1, embed_dim).to(img.device), positions], dim=0)
        x = tokens + positions
        
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        
        if self.pool =='max':
            x = x.max(dim=1)[0]
        elif self.pool =='mean':
            x = x.mean(dim=1)
        elif self.pool == 'cls':
            x = x[:, 0]

        return self.classifier(x)