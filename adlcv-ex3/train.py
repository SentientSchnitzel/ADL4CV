import os
import random

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from gpt import AndersenGPT
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# -----------------------
# Hyperparameters & Configs
# -----------------------
# Model
EMBED_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 12
MAX_SEQ_LEN = 1024  # Context length
PRETRAINED_TOKENIZER = "gpt2"
POS_ENC = "learnable"  # Options: learnable, fixed
# Training
START_FROM_PRETRAINED_GPT2_CHECKPOINT = True
BATCH_SIZE = 3
NUM_EPOCHS = 5
LR = 1e-4
WARMUP_STEPS = 625
WEIGHT_DECAY = 1e-4
GRADIENT_CLIPPING = 1.0
DATASET_NAME = "monurcan/andersen_fairy_tales"
MODEL_SAVE_PATH = "checkpoints"


# -----------------------
# Utility functions
# -----------------------
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------
# Data Preparation
# -----------------------
def prepare_data_iter(tokenizer, max_seq_len=512, batch_size=16):
    """
    Loads the dataset, tokenizes each story without truncation, and then
    splits the long token sequence into many training samples (chunks).
    Each chunk is of length max_seq_len+1 tokens so that when you shift,
    the inputs have length max_seq_len and the labels have length max_seq_len.
    """
    # Load the dataset
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    def tokenize_and_chunk(examples):
        # Tokenize without truncation/padding.
        outputs = tokenizer(examples["story"], add_special_tokens=True)
        all_chunks = {"input_ids": []}
        for tokens in outputs["input_ids"]:
            # You can use non-overlapping chunks (or add overlap if desired).
            # Here we use a non-overlapping strategy.
            # We want each chunk to have (max_seq_len+1) tokens to allow shifting.
            for i in range(0, len(tokens), max_seq_len + 1):
                chunk = tokens[i : i + max_seq_len + 1]
                # Only keep chunks that have at least 2 tokens (needed for input and label)
                if len(chunk) > 1:
                    all_chunks["input_ids"].append(chunk)
        return all_chunks

    # Tokenize and chunk the train and validation splits.
    # We remove all original columns so that each output is just our chunk.
    train_dataset = train_dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    test_dataset = test_dataset.map(
        tokenize_and_chunk,
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # The previous mapping produces a nested list (one list per original example).
    # Flatten the dataset so that each element is a single training sample (i.e. a chunk).
    train_dataset = train_dataset.flatten()
    test_dataset = test_dataset.flatten()

    def collate_fn(batch):
        # Each element of batch is a dict: {"input_ids": chunk}
        # Convert to tensors.
        input_ids = [
            torch.tensor(example["input_ids"], dtype=torch.long) for example in batch
        ]
        # Pad sequences in the batch to the maximum length (which might be less than max_seq_len+1)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        # For next-token prediction, the model input is everything except the last token
        # and the target (labels) is everything except the first token.
        inputs = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        return inputs, labels

    train_iter = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_iter = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_iter, test_iter


# -----------------------
# Main Training Loop
# -----------------------
def main(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_epochs=NUM_EPOCHS,
    pos_enc=POS_ENC,
    dropout=0.0,
    fc_dim=None,
    batch_size=BATCH_SIZE,
    lr=LR,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    gradient_clipping=GRADIENT_CLIPPING,
    max_seq_len=MAX_SEQ_LEN,
):
    # Use a pretrained tokenizer from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)
    # Ensure the tokenizer uses a padding token.
    # GPT2 has no pad token by default, so we assign the EOS token as pad_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get training and validation iterators.
    train_iter, test_iter = prepare_data_iter(
        tokenizer, max_seq_len=max_seq_len, batch_size=batch_size
    )
    vocab_size = tokenizer.vocab_size  # use the tokenizer's vocab size

    # Instantiate the GPT model.
    model = AndersenGPT(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        pos_enc=pos_enc,
        dropout=dropout,
        fc_dim=fc_dim,
        num_tokens=vocab_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if START_FROM_PRETRAINED_GPT2_CHECKPOINT:
        model.load_state_dict(torch.load("gpt2_pretrained.pt"))

    # Define the loss function for language modeling.
    # We ignore the padding token (which is now set to eos_token if needed)
    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Set up optimizer and learning rate scheduler.
    optimizer = torch.optim.AdamW(
        lr=lr, params=model.parameters(), weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(i / warmup_steps, 1.0)
    )

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        num_train_tokens = 0

        for batch in tqdm.tqdm(train_iter, desc="Training"):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass: outputs has shape (batch_size, seq_len, vocab_size)
            logits = model(inputs)
            # Flatten logits and labels for computing cross entropy loss.
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            loss = loss_function(logits, labels)
            loss.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.numel()
            num_train_tokens += labels.numel()

        avg_train_loss = train_loss / num_train_tokens
        print(
            f"  Training Loss: {avg_train_loss:.4f} | Perplexity: {np.exp(avg_train_loss):.4f}"
        )

        # Validation loop
        model.eval()
        val_loss = 0.0
        num_val_tokens = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(test_iter, desc="Validation"):
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                logits = logits.view(-1, vocab_size)
                labels = labels.view(-1)
                loss = loss_function(logits, labels)
                val_loss += loss.item() * labels.numel()
                num_val_tokens += labels.numel()

        avg_val_loss = val_loss / num_val_tokens
        print(
            f"  Validation Loss: {avg_val_loss:.4f} | Perplexity: {np.exp(avg_val_loss):.4f}"
        )

        # Save the model with the best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best.pt"))
            print(f"Model saved with best validation loss: {best_val_loss:.4f}")

    # Save the final model after training is complete
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "final.pt"))
    print(f"Final model saved to {MODEL_SAVE_PATH}/final.pt")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will run on {device}")
    set_seed(seed=1)

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    main()
