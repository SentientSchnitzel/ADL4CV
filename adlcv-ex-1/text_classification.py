import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
from tqdm import tqdm
import json
from sklearn.metrics import classification_report


from transformer import TransformerClassifier, to_device
from visualize_experiments import compute_test_metrics, plot_train_val_loss, plot_train_val_accuracy

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
    # Create train, val and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    train, val = train.split(split_ratio=0.75)
    print('training: ', len(train), 'val: ', len(val), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, val_iter, test_iter


def main(embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1, save_model=False,
         train_iter=None, val_iter=None, test_iter=None
    ):

    loss_function = nn.CrossEntropyLoss()

    # train_iter, val_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO,
                                                        # batch_size=batch_size
    # )

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

    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    best_val_loss = float('inf')
    best_val_acc = 0.0

    epoch_bar = tqdm(range(num_epochs), desc="Epochs")
    # training loop
    for e in epoch_bar:
        # print(f'\n epoch {e}')
        model.train()
        running_loss = []
        tot, cor= 0.0, 0.0
        for batch in tqdm(train_iter):
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label) # compute loss
            loss.backward()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

            # metrics
            running_loss.append(loss.item())
            tot += float(input_seq.size(0))
            cor += float((label == out.argmax(dim=1)).sum().item())
        b_loss = np.mean(running_loss)
        acc = cor / tot
        training_loss.append(b_loss)
        training_acc.append(acc)
        
        # validation step
        with torch.no_grad():
            model.eval()
            running_loss = []
            tot, cor= 0.0, 0.0
            for batch in tqdm(val_iter):
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq)
                loss = loss_function(out, label)
                running_loss.append(loss.item())
                tot += float(input_seq.size(0))
                cor += float((label == out.argmax(dim=1)).sum().item())
            b_loss = np.mean(running_loss)
            acc = cor / tot
            validation_loss.append(b_loss)
            validation_acc.append(acc)

            if b_loss < best_val_loss:
                best_val_loss = b_loss
                best_val_acc = acc
                best_model = model
                torch.save(best_model.state_dict(), 'best_model.pth') if save_model else None
            

            # print(f'-- {"validation"} loss {b_loss:.3} - {"validation"} accuracy {acc:.3}')
            epoch_bar.set_postfix({"val_loss": f"{b_loss:.3f}", "val_acc": f"{acc:.3f}"})

    
    ### TEST THE BEST PERFORMING (val_loss) MODEL
    test_model = best_model if best_model else model
    test_model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = []
    
    with torch.no_grad():
        for batch in tqdm(test_iter):
            input_seq = batch.text[0]
            if input_seq.size(1) > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            
            label = batch.label - 1
            out = test_model(input_seq)
            preds = out.argmax(dim=1)
            probs = torch.softmax(out, dim=1)[:, 1]  # Probability of positive class
            
            loss = loss_function(out, label)
            running_loss.append(loss.item())
            
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics
    test_loss = np.mean(running_loss)
    test_acc = (all_labels == all_preds).mean()
    
    test_metrics = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'true_labels': all_labels.tolist(),
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
    }
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    return ({'train_loss': training_loss, 
            'train_acc': training_acc, 
            'val_loss': validation_loss, 
            'val_acc': validation_acc},
            test_metrics
            )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    
    train_iter, val_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, batch_size=16)

    history, test_metrics = main(
        embed_dim = 256,
        num_heads = 8,
        num_layers = 8,
        num_epochs = 16,
        pos_enc = 'learnable',
        pool = 'max',
        dropout = 0.4,
        lr = 1e-5,
        train_iter=train_iter,
        val_iter=val_iter,
        test_iter=test_iter
        )
    # print(history)

    # json.dump(history, open('history_ex1_single.json', 'w'))
    # json.dump(test_metrics, open('test_metrics_ex1_single.json', 'w'))

    # mini visualization of train and validation loss
    # also the test metrics, conf matrix etc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc


    # set global figure size to 10, 7
    plt.rcParams['figure.figsize'] = (10, 7)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True

    plt.figure()
    plot_train_val_loss(history)
    plt.savefig('ex1_single_train_val.png')
    
    plt.figure()
    plot_train_val_accuracy(history)
    plt.savefig('ex1_single_train_val_accuracy.png')
    
    more_test_metrics = compute_test_metrics(test_metrics)
    plt.figure()
    cm = more_test_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predictions')
    plt.ylabel('True Labels')
    plt.title('Test Confusion Matrix')
    plt.grid(False)
    plt.savefig('ex1_single_confusion_matrix.png')
    
    plt.figure()
    fpr, tpr, _ = roc_curve(test_metrics['true_labels'], test_metrics['probabilities'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC curve')
    plt.savefig('ex1_single_roc_curve.png')


