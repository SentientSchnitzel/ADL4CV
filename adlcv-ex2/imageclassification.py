import argparse
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

import matplotlib.pyplot as plt

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset

def plot_loss(num_epochs, train_loss_history, val_loss_history, val_acc_history):
    # Plot the training metrics
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, 'bo-', label='Train Loss')
    plt.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_history, 'go-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

def main(image_size=(32,32), patch_size=(4,4), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=2,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
         
    ):

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # For tracking training metrics
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    for e in range(num_epochs):
        print(f'\nEpoch {e+1}/{num_epochs}')
        model.train()
        running_train_loss = 0.0
        train_batches = 0

        # Training loop with tqdm progress bar
        train_bar = tqdm.tqdm(train_iter, desc="Training", leave=False)
        for image, label in train_bar:
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            running_train_loss += loss.item()
            train_batches += 1

            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

            # Update tqdm display with current loss
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / train_batches
        train_loss_history.append(avg_train_loss)
        
        # Validation loop: calculate loss and accuracy
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        total_samples = 0
        correct_predictions = 0

        val_bar = tqdm.tqdm(test_iter, desc="Validation", leave=False)
        with torch.no_grad():
            for image, label in val_bar:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image)
                loss = loss_function(out, label)
                running_val_loss += loss.item()
                val_batches += 1

                preds = out.argmax(dim=1)
                total_samples += image.size(0)
                correct_predictions += (preds == label).sum().item()

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = running_val_loss / val_batches
        val_accuracy = correct_predictions / total_samples

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        print(f"Epoch {e+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), 'model.pth')
            best_val_loss = avg_val_loss
        
    plot_loss(num_epochs, train_loss_history, val_loss_history, val_acc_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer Training")
    parser.add_argument('--image_size', type=int, nargs=2, default=[32, 32],
                        help="Image size as two ints: height width")
    parser.add_argument('--patch_size', type=int, nargs=2, default=[4, 4],
                        help="Patch size as two ints: height width")
    parser.add_argument('--channels', type=int, default=3,
                        help="Number of image channels")
    parser.add_argument('--embed_dim', type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument('--num_heads', type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of output classes")
    parser.add_argument('--pos_enc', type=str, default='learnable',
                        help="Type of positional encoding")
    parser.add_argument('--pool', type=str, default='cls',
                        help="Pooling method")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument('--fc_dim', type=int, default=None,
                        help="Dimension of fully connected layer")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=625,
                        help="Warmup steps for the scheduler")
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help="Weight decay")
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help="Gradient clipping value")
    
    args = parser.parse_args()

    # Convert list arguments to tuples
    image_size = tuple(args.image_size)
    patch_size = tuple(args.patch_size)

    print("Starting training...")
    set_seed(seed=1)
    main(image_size=image_size,
         patch_size=patch_size,
         channels=args.channels,
         embed_dim=args.embed_dim,
         num_heads=args.num_heads,
         num_layers=args.num_layers,
         num_classes=args.num_classes,
         pos_enc=args.pos_enc,
         pool=args.pool,
         dropout=args.dropout,
         fc_dim=args.fc_dim,
         num_epochs=args.num_epochs,
         batch_size=args.batch_size,
         lr=args.lr,
         warmup_steps=args.warmup_steps,
         weight_decay=args.weight_decay,
         gradient_clipping=args.gradient_clipping)
    print("Training done!")
