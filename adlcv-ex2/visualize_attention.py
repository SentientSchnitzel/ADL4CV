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

def get_attention_rollout(model):
    """
    Computes the attention rollout given the attention weights stored in
    each transformer block. Assumes that a forward pass on a single image
    (batch size 1) has been performed so that each attention module's
    `last_attention` attribute is populated.
    Returns a 2D numpy array (mask) of shape (grid_size, grid_size).
    """
    att_mats = []
    # Loop through each transformer block and grab its attention weights.
    for block in model.transformer_blocks:
        # Each block’s attention.last_attention is of shape:
        #   (batch_size * num_heads, seq_len, seq_len)
        attn = block.attention.last_attention  # (num_heads, seq_len, seq_len) since batch=1
        # Reshape to add an explicit batch dimension: (1, num_heads, seq_len, seq_len)
        attn = attn.unsqueeze(0)
        att_mats.append(attn)
    
    # Stack to shape (num_layers, 1, num_heads, seq_len, seq_len) then squeeze batch dim:
    att_mats = torch.cat(att_mats, dim=0).squeeze(1)  # shape: (L, num_heads, seq_len, seq_len)
    # Average across heads → (L, seq_len, seq_len)
    att_mats = att_mats.mean(dim=1)
    
    # Add identity (for residual connections) and re-normalize for each layer.
    L, N, _ = att_mats.shape  # N = seq_len
    identity = torch.eye(N).to(att_mats.device)
    aug_att = att_mats + identity
    aug_att = aug_att / aug_att.sum(dim=-1, keepdim=True)
    
    # Recursively multiply the attention matrices.
    joint_att = torch.zeros(aug_att.size())
    joint_att[0] = aug_att[0]
    
    for n in range(1, aug_att.size(0)):
        joint_att[n] = torch.matmul(aug_att[n], joint_att[n - 1])
    
    v = joint_att[-1]
    grid_size = int(np.sqrt(aug_att.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    
    return mask

def visualize_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Initialize the Model (with the same hyperparameters as in training) ---
    image_size = (32, 32)
    patch_size = (4, 4)
    channels = 3
    embed_dim = 128
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
    model.load_state_dict(torch.load('model.pth', map_location=device))
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
        
        # Compute the attention rollout mask.
        mask = get_attention_rollout(model)  # shape: (grid_size, grid_size)
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
    fig.savefig("attention_visualization_1.png")
    fig2.savefig("attention_visualization_2.png")

if __name__ == "__main__":
    visualize_attention()
