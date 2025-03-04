import os
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# custom imports
from ddpm import Diffusion
from model import Classifier, UNet
from dataset.helpers import *
from util import set_seed, prepare_dataloaders
set_seed()

class VGG(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg11.html
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, features=False):
        feat = self.features(x)
        feat = self.avg_pool(feat)
        x = self.dropout(self.flatten(feat))
        x = self.fc(x)
        if features:
            return feat
        else:
            return x
        
def get_features(model, images):
    model.eval()  
    with torch.no_grad():
        features = model(images, features=True)
    features = features.squeeze(3).squeeze(2).cpu().numpy()
    return features

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    # https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance
    # HINT: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html
    # Implement FID score

    fid = ...

    return fid

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########################################### classifier guidance ##########################################
    ddpm_cg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cg', device=device)
    classifier = Classifier(
        img_size=16, c_in=3, labels=5,
        time_dim=256,channels=32, device=device
    )
    classifier.to(device)
    classifier.eval()
    classifier.load_state_dict(torch.load('weights/classifier/model.pth', map_location=device))

    unet_ddpm = UNet(device=device)
    unet_ddpm.eval()
    unet_ddpm.to(device)
    unet_ddpm.load_state_dict(torch.load('weights/DDPM/model.pth', map_location=device))
    ddpm_cg.classifier = classifier

    ######################################### classifier-free guidance #########################################
    ddpm_cFg = Diffusion(img_size=16, T=500, beta_start=1e-4, beta_end=0.02, diff_type='DDPM-cFg', device=device)
    unet_ddpm_cFg = UNet(num_classes=5, device=device)
    unet_ddpm_cFg.eval()
    unet_ddpm_cFg.to(device)
    unet_ddpm_cFg.load_state_dict(torch.load('weights/DDPM-cfg/model.pth', map_location=device))

    model = VGG()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load('weights/vgg-sprites/model.pth', map_location=device))
    dims = 256 # vgg feature dim

    _ ,_, test_loader = prepare_dataloaders(val_batch_size=100)

    vgg_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    original_feat = np.empty((len(test_loader.dataset), dims))
    generated_feat_cg = np.empty((len(test_loader.dataset), dims))
    generated_feat_cFg = np.empty((len(test_loader.dataset), dims))

    start_idx = 0

    for images, _ in tqdm(test_loader):

        images = images.to(device)
        original = get_features(model, images)
        
        # classifier guidance
        y = torch.randint(0, 5, (images.shape[0],), device=device)
        cg_images = ddpm_cg.p_sample_loop(unet_ddpm, images.shape[0], y=y, verbose=False)
        cg_images = vgg_transform(cg_images/255.0)
        cg_features = get_features(model, cg_images)

        # classifier-free guidance
        y = F.one_hot(y, num_classes=5).float()
        cFg_images = ddpm_cFg.p_sample_loop(unet_ddpm_cFg, images.shape[0], y=y, verbose=False)
        cFg_images = vgg_transform(cFg_images/255.0)
        cFg_features = get_features(model, cFg_images)

        # store features
        original_feat[start_idx:start_idx + original.shape[0]] = original
        generated_feat_cg[start_idx:start_idx + original.shape[0]] = cg_features
        generated_feat_cFg[start_idx:start_idx + original.shape[0]] = cFg_features

        start_idx = start_idx + original.shape[0]
    

    mu_original, sigma_original = feature_statistics(original_feat)
    mu_cg, sigma_cg = feature_statistics(generated_feat_cg)
    mu_cFg, sigma_cFg = feature_statistics(generated_feat_cFg)

    fid_cg = frechet_distance(mu_original, sigma_original, mu_cg, sigma_cg)
    fid_cFg = frechet_distance(mu_original, sigma_original, mu_cFg, sigma_cFg)
    print(f'[FID classifier guidance] {fid_cg:.3f}')
    print(f'[classifier-free guidance] {fid_cFg:.3f}')