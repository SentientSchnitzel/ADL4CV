import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from vit import ViT  
from imageclassification import prepare_dataloaders, select_two_classes_from_cifar10

def denormalize(image_tensor):
    """
    Undo the normalization (mean=0.5, std=0.5) so that the image is in [0,1].
    """
    image = image_tensor * 0.5 + 0.5
    return image.clamp(0, 1)

def get_attention_rollout(att_maps):
    """
    att_maps: a Python list of length L (number of Transformer blocks).
              Each element is a tensor of shape (num_heads, seq_len, seq_len).
    
    Returns:
        mask (H x W) as a NumPy array, where H x W = (seq_len - 1), typically
        because one token is the [CLS] token and the rest are patches.
    """
    # 1) Stack into shape (L, num_heads, seq_len, seq_len)
    att_mats = torch.stack(att_maps, dim=0)  

    # 2) Average over heads => shape (L, seq_len, seq_len)
    att_mats = att_mats.mean(dim=1)

    # 3) Add identity and normalize each matrix
    L, N, _ = att_mats.shape  # L = #layers, N = seq_len
    I = torch.eye(N, device=att_mats.device)
    att_mats = att_mats + I[None, :, :]                # shape: (L, N, N)
    att_mats = att_mats / att_mats.sum(dim=-1, keepdim=True)  # normalize row-wise

    # 4) Recursively multiply the attention maps: shape => (N, N)
    joint_att = att_mats[0]  # first block
    for i in range(1, L):
        joint_att = att_mats[i] @ joint_att  # (N, N)

    # 5) The final matrix joint_att is (N, N).
    #    Often we consider row 0 as the "CLS -> patches" attention distribution.
    #    The first token is CLS, so we skip index 0 from the final distribution
    #    and reshape the remainder into a patch grid.
    v = joint_att[0]                 # shape (N,)
    grid_size = int(np.sqrt(N - 1))  # skip the CLS token => N-1 patch tokens
    mask = v[1:].reshape(grid_size, grid_size)  # shape: (grid_size, grid_size)

    # Move to CPU and NumPy
    mask = mask.detach().cpu().numpy()
    return mask

def visualize_attention(runname):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize the Model (with the same hyperparameters as in training) ---
    image_size = (32, 32)
    patch_size = (8, 8)
    channels = 3
    embed_dim = 256
    num_heads = 4
    num_layers = 4
    num_classes = 2
    pos_enc = 'learnable'
    pool = 'cls'
    dropout = 0.3
    fc_dim = None

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels,
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim,
                num_classes=num_classes)
    
    # Load your trained model weights.
    model.load_state_dict(torch.load(f'models/{runname}_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # --- Prepare Data and Select Images ---
    batch_size = 16
    _, _, _, testset = prepare_dataloaders(batch_size=batch_size)
    
    # Randomly select 5 images from the test set.
    indices = random.sample(range(len(testset)), 5)
    selected_images = [testset[i][0] for i in indices]  # each is a tensor of shape (C,H,W)

    # --- Create a Figure to Display the Results ---
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
    fig2, axes2 = plt.subplots(nrows=5, ncols=2, figsize=(8, 20))
    
    for i, img_tensor in enumerate(selected_images):
        # Denormalize and convert image to numpy (H, W, C) for display.
        img_denorm = denormalize(img_tensor)
        img_np = np.transpose(img_denorm.cpu().numpy(), (1, 2, 0))
        
        # Forward pass through the model to update attention weights.
        x = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)
            att_maps = model.get_attention_maps()
        
        # Compute the attention rollout mask.
        mask = get_attention_rollout(att_maps)  
        H, W, _ = img_np.shape
        mask_resized = cv2.resize(mask / mask.max(), (W, H))[..., np.newaxis]

        # Multiply the resized mask with the original image.
        # Note: img_np is assumed to be in [0, 1], so we scale it to [0, 255]
        overlay = (mask_resized * (img_np * 255)).astype("uint8")
        original_display = (img_np * 255).astype("uint8")
        
        axes[i, 0].set_title('Original')
        axes[i, 1].set_title('Attention Map')

        axes[i, 0].imshow(original_display)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay)
        axes[i, 1].axis("off")
        
        
        # Create a heatmap (using OpenCV’s COLORMAP_JET) from the mask.
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        
        # Overlay the heatmap on the original image.
        # Here we blend 50% of the original image with 50% of the heatmap.
        overlay = 0.5 * (img_np * 255) + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Plot the original image.
        axes2[i, 0].imshow(img_np)
        axes2[i, 0].axis('off')
        axes2[i, 0].set_title("Original Image")
        
        # Plot the attention overlay.
        axes2[i, 1].imshow(overlay)
        axes2[i, 1].axis('off')
        axes2[i, 1].set_title("Attention Overlay")
    
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(f"plots/{runname}_attention_visualization_1.png")
    fig2.savefig(f"plots/{runname}_attention_visualization_2.png")

if __name__ == "__main__":
    runname = "aug_e20_ed256_p8"
    visualize_attention(runname)
