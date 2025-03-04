import os
import numpy as np
import matplotlib.pyplot as plt

# torch imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# custom imports
from ddpm import Diffusion
from model import UNet

from dataset.helpers import im_normalize, tens2image
from dataset import SpritesDataset

def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i])

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2929 # change it to any integer you want to see different results.
    torch.manual_seed(seed)

    os.makedirs('assets/', exist_ok=True)
    
    # dataset and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])

    batch_size = 8
    trainset = SpritesDataset(transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    images  = next(iter(trainloader))
    # visualize examples
    example_images = np.stack([im_normalize(tens2image(images[idx])) for idx in range(batch_size)], axis=0)
    show(example_images, 'Example sprites', save_path='assets/example.png')

    ################## Diffusion class ##################
    # TASK 1: Implement beta, alpha, and alpha_hat 
    diffusion = Diffusion(device=device)
    plt.figure()
    plt.plot(range(1,diffusion.T+1), diffusion.alphas.cpu().numpy(), label='alphas', linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.alphas_bar.cpu().numpy(), label='alphas_bar',linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.betas.cpu().numpy(), label='betas', linewidth=3)
    plt.title('Diffusion parameters')
    plt.legend()
    plt.savefig('assets/diffusion_params.png', bbox_inches='tight')
    plt.show()
    #####################################################
    

    # timesteps for forward
    t = torch.Tensor([0, 50, 100, 150, 200, 300, 499]).long().to(device)
    fig_titles = [f'Step {ti.item()}' for ti in t]
    x0 = images[0].unsqueeze(0).to(device) # add batch dimenstion

    ################## Forward process ##################
    # TASK 2: Implement it in the diffusion class
    xt, noise = diffusion.q_sample(x0, t)
    #####################################################

    noised_images = np.stack([im_normalize(tens2image(xt[idx].cpu())) for idx in range(t.shape[0])], axis=0)
    show(noised_images, title='Forward process', fig_titles=fig_titles, save_path='assets/forward.png')

    ################## Inverse process ##################
    model = UNet(device=device)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load('models/weights-59epochs-full-dataset.pt', map_location=device)) # load the given model
    torch.manual_seed(seed)

    # TASK 3: Implement it in the diffusion class
    x_new, intermediate_images = diffusion.p_sample_loop(model, 1, timesteps_to_save=t)
    intermediate_images = [tens2image(img.cpu()) for img in intermediate_images]
    show(intermediate_images, title='Reverse process', fig_titles=fig_titles, save_path='assets/reverse.png')
    #####################################################

