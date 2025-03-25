import os
from einops import repeat
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch
import torch
import torch.nn as nn

# custom imports
from ddpm import Diffusion
from model import Classifier
from dataset.helpers import *
from util import set_seed, prepare_dataloaders

def show_n_forward(imgs, title=None, fig_titles=None, save_path=None): 
    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    num_rows = len(imgs) // 7
    fig_width = 20  # Adjust the figure width as needed
    fig_height = 2 * num_rows  # Adjust the figure height as needed
    fig, axs = plt.subplots(num_rows, ncols=7, figsize=(fig_width, fig_height))

    for i in range(num_rows):
        for j in range(7):
            idx = i * 7 + j
            axs[i, j].imshow(imgs[idx])
            axs[i, j].axis('off')
            if fig_titles is not None:
                axs[i, j].set_title(fig_titles[idx])

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed()
    os.makedirs('assets', exist_ok=True)
    
    # Initialize diffusion class
    diffusion = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, device=device)

    # Load classifier
    model = Classifier(
        img_size=16, c_in=3, labels=5,
        time_dim=256, channels=32, device=device
    )
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))

    # sample time steps
    t = torch.Tensor([0, 50, 100, 150, 200, 300, 499]).long().to(device)

    # softmax
    softmax = nn.Softmax(dim=-1)

    ########################################## Per Timestep Accuracy ##########################################
    _, val_loader, _ = prepare_dataloaders(100, val_batch_size=1)
    per_t_acc = torch.zeros_like(t, dtype=torch.float32)
    for img, label in tqdm(val_loader, desc='Eval'):
        img = img.to(device)
        label = label.to(device)
        label = repeat(label, '1 -> (b 1)', b=t.shape[0])
        x_t, noise = diffusion.q_sample(img, t)

        with torch.no_grad():
            logits = model(x_t, t)

        out = softmax(logits)
        prediction = torch.argmax(out, dim=-1)
        
        assert prediction.shape == label.shape
        per_t_acc += prediction == label
        
    per_t_acc /= len(val_loader)

    plt.figure()
    plt.plot(t.cpu().numpy(), per_t_acc.cpu().numpy(), linewidth=3)
    plt.xlabel('Timestep', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.savefig('./assets/per_t_accuracy.png', bbox_inches='tight')


    ########################################## Qualitative Results ############################################
    _, val_loader,_ = prepare_dataloaders(100, val_batch_size=10)
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)

    vis_images = []
    vis_labels = []
    for img, label in zip(images, labels):
        img = img.unsqueeze(0)
        label = label.unsqueeze(0)
        label = repeat(label, '1 -> (b 1)', b=t.shape[0])
        x_t, noise = diffusion.q_sample(img, t)

        with torch.no_grad():
            logits = model(x_t, t)

        out = softmax(logits)
        prediction = torch.argmax(out, dim=-1)

        norm_images = [im_normalize(tens2image(im.cpu().detach())) for im in x_t]
        vis_images.extend(norm_images)
        text_labels = [f'Pred: {pred.item()}, Target: {l.item()}, t:{ti.item()}' for pred, l,ti in zip(prediction, label, t)]
        vis_labels.extend(text_labels)

    show_n_forward(vis_images, fig_titles=vis_labels, save_path='assets/clf_forward.png')
