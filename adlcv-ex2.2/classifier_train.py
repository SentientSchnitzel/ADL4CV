import os
from tqdm import tqdm

import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from ddpm import Diffusion
from model import Classifier 
from util import set_seed, prepare_dataloaders

EPOCHS = 20

def create_result_folders(experiment_name):
    os.makedirs(os.path.join("weights", experiment_name), exist_ok=True)

def train(device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256):

    exp_name = 'classifier'
    create_result_folders(exp_name)
    train_loader, val_loader, _  = prepare_dataloaders()

    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    model = Classifier(img_size=img_size, c_in=input_channels, labels=5, 
        time_dim=time_dim,channels=channels, device=device
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params/1_000_000:.2f}M')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)
    pbar = tqdm(range(1, EPOCHS + 1), desc='Training')
    
    for epoch in pbar:
        # -------------------------
        #         TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1) Sample random timesteps in [0..T-1]
            t = torch.randint(0, T, (images.size(0),), device=device).long()

            # 2) Add noise to the images (noisy_images)
            x_t, _ = diffusion.q_sample(images, t)

            # 3) Forward pass through classifier
            logits = model(x_t, t)

            # 4) Compute loss and backprop
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate train metrics
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += images.size(0)

        # Average train loss/acc over all samples
        train_loss /= train_samples
        train_acc = train_correct / train_samples
    
    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            t = torch.randint(0, T, (images.size(0),), device=device).long()
            x_t, noise = diffusion.q_sample(images, t)
            logits = model(x_t, t)
            
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            
            # Accumulate train metrics
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += images.size(0)

        # Average train loss/acc over all samples
        train_loss /= train_samples
        train_acc = train_correct / train_samples
    
    # save your checkpoint in weights/classifier/model.pth
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed()
    train(device=device)

if __name__ == '__main__':
    main()
    

        
def train(device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256):
    exp_name = 'classifier'
    create_result_folders(exp_name)
    train_loader, val_loader, _  = prepare_dataloaders()

    diffusion = Diffusion(
        img_size=img_size,
        T=T,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    model = Classifier(
        img_size=img_size,
        c_in=input_channels,
        labels=5,
        time_dim=time_dim,
        channels=channels,
        device=device
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params/1_000_000:.2f}M')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # We'll track the progress of epochs with tqdm
    pbar = tqdm(range(1, EPOCHS + 1), desc='Training')

    for epoch in pbar:
        # -------------------------
        #         TRAIN
        # -------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 1) Sample random timesteps in [0..T-1]
            t = torch.randint(0, T, (images.size(0),), device=device).long()

            # 2) Add noise to the images (noisy_images)
            #    Adjust depending on your actual Diffusion interface
            noisy_images,_ = diffusion.q_sample(images, t)

            # 3) Forward pass through classifier
            logits = model(noisy_images, t)

            # 4) Compute loss and backprop
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate train metrics
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += images.size(0)
            
        # Average train loss/acc over all samples
        train_loss /= train_samples
        train_acc = train_correct / train_samples
        
        # -------------------------
        #        VALIDATION
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Sample random timesteps
                t = torch.randint(0, T, (images.size(0),), device=device).long()

                # Noisy images
                noisy_images = diffusion.q_sample(images, t)

                # Forward pass
                logits = model(noisy_images, t)
                loss = loss_fn(logits, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += images.size(0)

        val_loss /= val_samples
        val_acc = val_correct / val_samples

        # Update the tqdm progress bar with current metrics
        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
        })

    # Finally, save your checkpoint
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

if main == '__main__':
    main()