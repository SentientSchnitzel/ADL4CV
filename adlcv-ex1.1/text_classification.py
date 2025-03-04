import argparse
import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
import matplotlib.pyplot as plt  

from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, test_iter

def plot_loss(num_epochs, train_losses, val_losses, runname):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/loss_plot_{runname}.png")

def main(embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1
    ):
    
    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                            batch_size=batch_size
    )

    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS,
                                  )
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    # training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        train_bar = tqdm.tqdm(train_iter, desc="Training", leave=False)
        for batch in tqdm.tqdm(train_iter):
            # zero your gradients for every batch
            opt.zero_grad()
            input_seq = batch.text[0]
            if input_seq.size(1) > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            label = batch.label - 1 
            
            # forward pass
            out = model(input_seq)
            loss = loss_function(out, label)
            
            # backward pass
            loss.backward()
            
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            
            # update weights
            opt.step()
            sch.step()
            
            # Accumulate loss scaled by number of samples in this batch
            running_train_loss += loss.item() * input_seq.size(0)
            train_samples += input_seq.size(0)
            
            # Update tqdm display with current loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = running_train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Evaluation loop
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        tot, cor = 0.0, 0.0
        all_predictions = []
        val_bar = tqdm.tqdm(test_iter, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                input_seq = batch.text[0]
                if input_seq.size(1) > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                label = batch.label - 1
                out = model(input_seq)
                
                loss_val = loss_function(out, label)
                running_val_loss += loss_val.item() * input_seq.size(0)
                val_samples += input_seq.size(0)
                
                predictions = out.argmax(dim=1)
                all_predictions.append(predictions.cpu())
                tot += input_seq.size(0)
                cor += (label == predictions).sum().item()
                
                val_bar.set_postfix(loss=f"{loss_val.item():.4f}")
                
        avg_val_loss = running_val_loss / val_samples
        val_losses.append(avg_val_loss)
        acc = cor / tot
<<<<<<< HEAD:adlcv-ex-1/text_classification.py
        # After processing all batches, compute unique predicted classes
        unique_predictions = torch.unique(torch.cat(all_predictions))
        print("Unique predicted classes in validation:", unique_predictions.tolist())
    
        
=======
>>>>>>> cccfb02eaaa99bd4f3a365c591fc8adb4f5ec012:adlcv-ex1.1/text_classification.py
        print(f"Epoch {e+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Accuracy: {acc:.3f}")
        
        unique_predictions = torch.unique(torch.cat(all_predictions))
        print("Unique predicted classes in validation:", unique_predictions.tolist())

    return train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer-based Text Classification Training")
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension size')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--pos_enc', type=str, default='fixed', help='Positional encoding type')
    parser.add_argument('--pool', type=str, default='max', help='Pooling type (e.g., "max" or "avg")')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=None, help='Dimension for the fully connected layer (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=625, help='Number of warmup steps for the learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay factor')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Maximum norm for gradient clipping')
    parser.add_argument('--runname', type=str, default='test', help='Name of the run')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    
    set_seed(seed=1)
    
    train_losses, val_losses = main(embed_dim=args.embed_dim,
         num_heads=args.num_heads,
         num_layers=args.num_layers,
         num_epochs=args.num_epochs,
         pos_enc=args.pos_enc,
         pool=args.pool,
         dropout=args.dropout,
         fc_dim=args.fc_dim,
         batch_size=args.batch_size,
         lr=args.lr,
         warmup_steps=args.warmup_steps,
         weight_decay=args.weight_decay,
         gradient_clipping=args.gradient_clipping)
    
    plot_loss(args.num_epochs, train_losses, val_losses, args.runname)