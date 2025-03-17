from scipy.linalg import sqrtm
import numpy as np

# torch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

vgg_transform = torchvision.transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor (C, H, W)
    torchvision.transforms.Normalize((0.5,), (0.5,)),
    # convert to float
    transforms.Lambda(lambda x: x.float()),
])

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


    # 1) Compute the difference in means and its square norm
    diff = mu1 - mu2
    diff_squared = diff.dot(diff)

    # 2) Compute the sqrt of the product of covariance matrices
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical error can give slight imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 3) Compute trace component
    trace_component = np.trace(sigma1 + sigma2 - 2 * covmean)

    fid = diff_squared + trace_component

    return fid



### CLIP


# Function to calculate CLIP score
def calculate_clip_score(images, prompt, model, processor, device):
    # Preprocess the text (tokenize)
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    
    # Preprocess images
    image_embeddings = []
    for image in images:
        # Convert to tensor if not already
        #image = torch.tensor(image, dtype=torch.float32)

        # Min-max normalization to [0,1]
        image = (image - image.min()) / (image.max() - image.min())
        image_input = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            # Get image features
            image_features = model.get_image_features(**image_input)
            image_embeddings.append(image_features)
    
    # Stack image embeddings into a tensor
    image_embeddings = torch.stack(image_embeddings)
    
    # Get text features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    
    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Compute cosine similarity between image and text embeddings
    similarity = (image_embeddings @ text_features.T).squeeze(1)
    
    # Return the average cosine similarity as the CLIP score
    clip_score = similarity.mean().item()
    return clip_score

