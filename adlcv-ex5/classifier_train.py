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
    l = len(train_loader)
    logger = SummaryWriter(os.path.join("runs", exp_name))

    min_train_loss = 1e10
    for epoch in pbar:
        model.train()
        batchbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        epoch_loss = 0
        for i, (images, labels) in enumerate(batchbar):
            images = images.to(device)
            labels = labels.to(device)

            # Do not forget to noise your images !

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t) # making noisy images at "t"
            logits = model(x_t, t)
            loss = loss_fn(logits, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        epoch_loss /= l
        if epoch_loss <= min_train_loss:
            torch.save(model.state_dict(), os.path.join("weights", exp_name, f"model{epoch}.pth"))
            min_train_loss = epoch_loss
    
    # save your checkpoint in weights/classifier/model.pth
    torch.save(model.state_dict(), os.path.join("weights", exp_name, 'model.pth'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed()
    train(device=device)

if __name__ == '__main__':
    main()
    

        