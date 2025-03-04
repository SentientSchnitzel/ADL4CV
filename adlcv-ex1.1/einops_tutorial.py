import torch
from einops import repeat, rearrange

batch_size, heigh, width = 32, 224,224
a = torch.rand((batch_size, heigh, width))

# Increase the channels to 3
a1 = repeat(a, 'b h w -> b c h w', c=3)
print('Increase channels to 3',a1.shape)

# Swap batch with the channels
a2 = rearrange(a1, 'b c h w -> c b h w')
print('Swap batch with the channels', a2.shape)

# Merge batch with channels
a3 = rearrange(a1, 'b c h w -> (b c) h w')
print('Merge batch with channels', a3.shape)


# Split the batch to two dimensions
a4 = rearrange(a1, '(d1 d2) c h w -> d1 d2 c h w', d1=16,d2=2)
print('Split the batch to two dimensions', a4.shape)