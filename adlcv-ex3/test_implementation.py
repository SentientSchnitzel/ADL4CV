import torch
from gpt import AndersenGPT, MaskedAttention


# Test the modified Attention module (with causal masking)
def test_attention(batch_size=16, seq_len=512, embed_dim=128, device="cpu"):
    # Create a random token tensor of shape: (batch_size, seq_len, embed_dim)
    token = torch.rand(batch_size, seq_len, embed_dim).to(device)

    attention = MaskedAttention(embed_dim=embed_dim, num_heads=8)
    attention = attention.to(device)
    output = attention(token)

    # Check that output shape is as expected
    assert output.size() == (batch_size, seq_len, embed_dim)

    print(f"token shape: {token.shape}")
    print(f"output shape: {output.shape}")


# Test the GPT model for autoregressive next-token prediction.
def test_transformer_gpt(batch_size=16, seq_len=512, num_tokens=50_000, device="cpu"):
    # Generate a random input sequence of token indices: (batch_size, seq_len)
    input_seq = torch.randint(low=0, high=num_tokens, size=(batch_size, seq_len)).to(
        device
    )

    # Instantiate the GPT-style model. Note that we do not use any CLS token or pooling.
    transformer_gpt = AndersenGPT(
        embed_dim=128,
        num_heads=8,
        num_layers=2,
        max_seq_len=seq_len,
        pos_enc="learnable",  # 'fixed' or 'learnable'
        dropout=0.0,
        num_tokens=num_tokens,
    )
    transformer_gpt = transformer_gpt.to(device)
    output = transformer_gpt(input_seq)

    # For GPT, we expect logits for each token over the vocabulary:
    # shape: (batch_size, seq_len, num_tokens)
    assert output.size() == (batch_size, seq_len, num_tokens)

    print(f"input sequence shape: {input_seq.shape}")
    print(f"GPT output shape: {output.shape}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("--- Testing Masked Attention Module ---")
    test_attention(batch_size=8, seq_len=512, embed_dim=128, device=device)
    print("-" * 45)

    print("--- Testing TransformerGPT Model ---")
    test_transformer_gpt(batch_size=4, seq_len=128, num_tokens=10_000, device=device)
    print("-" * 45)
