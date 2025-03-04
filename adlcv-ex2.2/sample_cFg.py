import matplotlib.pyplot as plt

# torch
import torch
import torch.nn.functional as F

# custom imports
from ddpm import Diffusion
from model import UNet
from dataset.helpers import *
from util import show, set_seed, CLASS_LABELS
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed()

# Load model
ddpm_cFg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cFg', device=device)

unet_ddpm_cFg = UNet(num_classes=5, device=device)
unet_ddpm_cFg.eval()
unet_ddpm_cFg.to(device)
unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))

# Sample
y = torch.tensor([0,1,2,3,4], device=device) 
y = F.one_hot(y, num_classes=5).float()
x_new = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, 5, y=y)
imgs = [im_normalize(tens2image(x_gen.cpu())) for x_gen in x_new]
show(imgs, fig_titles=CLASS_LABELS, title='classifier FREE guidance', save_path='assets/cFg_samples.png')

