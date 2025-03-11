import os
import random
from typing import Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class TextualInversionDataset(Dataset):
    """Dataset for textual inversion training."""
    
    def __init__(
        self,
        data_root: str,
        tokenizer,
        size: int = 512,
        repeats: int = 100,
        learnable_property: str = "object",
        placeholder_token: str = "*",
        center_crop_prob: float = 0.5,  # Probability of center cropping
        flip_prob: float = 0.5,  # Probability of horizontal flipping
        set: str = "train"
    ):
        """
        Initialize the dataset for textual inversion.
        
        Args:
            data_root: Directory containing the concept images
            tokenizer: CLIP tokenizer for processing text
            size: Target size for images (square)
            repeats: Number of times to repeat the dataset for training
            learnable_property: "object" or "style" to determine text templates
            placeholder_token: Token to be inserted in text templates
            center_crop_prob: Probability of applying center crop (0.0 to 1.0)
            flip_prob: Probability of applying horizontal flip (0.0 to 1.0)
            set: "train" or other (affects dataset length)
        """
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop_prob = center_crop_prob
        self.flip_prob = flip_prob
        self.learnable_property = learnable_property
        
        self.image_paths = [os.path.join(self.data_root, f) for f in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats if set == "train" else self.num_images
        
        from templates import imagenet_templates_small, imagenet_style_templates_small
        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
    
    def _flip_image(self, image):
        """Horizontally flip the image."""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def __len__(self) -> int:
        return self._length
    
    def _center_crop(self, image):
        """Center crop the image to create a square image without distorting aspect ratio."""
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        return image.crop((left, top, right, bottom))
    
    def _resize_with_padding(self, image, target_size):
        """Resize image to target size while maintaining aspect ratio and adding black padding."""
        # Get original aspect ratio
        width, height = image.size
        aspect = width / height

        if aspect > 1:  # Wider than tall
            new_width = target_size
            new_height = int(target_size / aspect)
            pad_top = (target_size - new_height) // 2
            pad_bottom = target_size - new_height - pad_top
            pad_left = 0
            pad_right = 0
        else:  # Taller than wide
            new_height = target_size
            new_width = int(target_size * aspect)
            pad_left = (target_size - new_width) // 2
            pad_right = target_size - new_width - pad_left
            pad_top = 0
            pad_bottom = 0

        # Resize image while maintaining aspect ratio
        image = image.resize((new_width, new_height), Image.BICUBIC)
        
        # Create new black image with target size
        result = Image.new('RGB', (target_size, target_size), color='black')
        
        # Paste resized image in center
        result.paste(image, (pad_left, pad_top))
        
        return result
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        example = {}
        image = Image.open(self.image_paths[i % self.num_images]).convert("RGB")
        
        # Get text prompt
        text = random.choice(self.templates).format(self.placeholder_token)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        # Process image with random center crop based on probability
        if random.random() < self.center_crop_prob:
            image = self._center_crop(image)
            image = image.resize((self.size, self.size), Image.BICUBIC)
        else:
            image = self._resize_with_padding(image, self.size)
        
        # Apply random horizontal flip based on probability
        if random.random() < self.flip_prob:
            image = self._flip_image(image)
            
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)  # Normalize to [-1, 1] range
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        
        return example 