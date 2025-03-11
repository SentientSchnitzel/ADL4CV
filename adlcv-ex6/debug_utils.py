import os
from PIL import Image
import numpy as np
import torch
from typing import List, Tuple
from textual_inversion_dataset import TextualInversionDataset

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image."""
    array = ((tensor.permute(1, 2, 0) + 1.0) * 127.5).numpy().astype(np.uint8)
    return Image.fromarray(array)

def debug_dataloader(tokenizer, config):
    """Debug function to check image processing in the dataloader."""
    print("\n=== Starting Dataloader Debug ===")
    
    # Create debug datasets with different combinations
    debug_datasets = [
        ("No Crop, No Flip", TextualInversionDataset(
            data_root=config["concept_folder"],
            tokenizer=tokenizer,
            size=512,
            placeholder_token=config["placeholder_token"],
            repeats=1,
            learnable_property=config["what_to_teach"],
            center_crop_prob=0.0,
            flip_prob=0.0
        )),
        ("Center Crop, No Flip", TextualInversionDataset(
            data_root=config["concept_folder"],
            tokenizer=tokenizer,
            size=512,
            placeholder_token=config["placeholder_token"],
            repeats=1,
            learnable_property=config["what_to_teach"],
            center_crop_prob=1.0,
            flip_prob=0.0
        )),
        ("No Crop, Flip", TextualInversionDataset(
            data_root=config["concept_folder"],
            tokenizer=tokenizer,
            size=512,
            placeholder_token=config["placeholder_token"],
            repeats=1,
            learnable_property=config["what_to_teach"],
            center_crop_prob=0.0,
            flip_prob=1.0
        )),
        ("Center Crop, Flip", TextualInversionDataset(
            data_root=config["concept_folder"],
            tokenizer=tokenizer,
            size=512,
            placeholder_token=config["placeholder_token"],
            repeats=1,
            learnable_property=config["what_to_teach"],
            center_crop_prob=1.0,
            flip_prob=1.0
        ))
    ]
    
    # Create debug output directory
    debug_dir = os.path.join(config["output_dir"], "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Number of images in dataset: {len(debug_datasets[0][1].image_paths)}")
    
    all_debug_images = []
    
    # Process each image with all combinations
    for idx in range(len(debug_datasets[0][1].image_paths)):
        image_path = debug_datasets[0][1].image_paths[idx]
        print(f"\nProcessing image {idx + 1}: {os.path.basename(image_path)}")
        
        # Load and analyze original image
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size
        orig_aspect = orig_w / orig_h
        print(f"Original dimensions: {orig_w}x{orig_h}, Aspect ratio: {orig_aspect:.3f}")
        
        # Get processed images from all datasets
        processed_images = []
        for name, dataset in debug_datasets:
            processed = dataset[idx]
            processed_img = tensor_to_image(processed["pixel_values"])
            processed_images.append((name, processed_img))
            
            # Save individual images
            save_path = os.path.join(debug_dir, f"debug_{idx}_{name.replace(', ', '_').lower()}.png")
            processed_img.save(save_path)
            print(f"{name} version saved to: {save_path}")
        
        # Store images for summary visualization
        all_debug_images.append((orig_img, processed_images))
    
    if all_debug_images:
        create_debug_summary(all_debug_images, debug_dir)
    
    print("\n=== Dataloader Debug Complete ===\n")

def create_debug_summary(all_debug_images: List[Tuple[Image.Image, List[Tuple[str, Image.Image]]]], debug_dir: str):
    """Create a summary visualization of all processed images."""
    # Calculate dimensions
    max_orig_width = max(orig.size[0] for orig, _ in all_debug_images)
    summary_width = max_orig_width + (512 * 4)  # 4 variations
    summary_height = sum(max(orig.size[1], 512) for orig, _ in all_debug_images)
    
    # Create summary image
    summary = Image.new('RGB', (summary_width, summary_height), color='white')
    current_y = 0
    
    # Add each set of images
    for orig_img, processed_imgs in all_debug_images:
        # Paste original image
        summary.paste(orig_img, (0, current_y))
        
        # Paste processed versions
        for i, (name, img) in enumerate(processed_imgs):
            summary.paste(img, (max_orig_width + (i * 512), current_y))
        
        current_y += max(orig_img.size[1], 512)
    
    # Save summary
    summary_path = os.path.join(debug_dir, "debug_summary.png")
    summary.save(summary_path)
    print(f"\nSummary visualization saved to: {summary_path}")
    print("In summary image: Original | No Crop, No Flip | Center Crop, No Flip | No Crop, Flip | Center Crop, Flip") 