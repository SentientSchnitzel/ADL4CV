import os
from tqdm import tqdm

import numpy as np
import random

# torch imports
import torch
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

# custom imports
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
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Do not forget to noise your images !
            t = torch.randint(0, T, (images.size(0),)).to(device)
            x, noise = diffusion.q_sample(images, t)

            optimizer.zero_grad()
            logits = model(x, t)
            #logits = softmax(logits)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            # accumulate train metrics
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            train_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()

        train_loss /= train_samples
        train_acc = train_correct / train_samples

        pbar.set_description(f'Training: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

    # save your checkpoint in weights/classifier/model.pth
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed()
    train(device=device)

if __name__ == '__main__':
    main()
    

        