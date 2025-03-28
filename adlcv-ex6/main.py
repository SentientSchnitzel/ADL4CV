import argparse
import itertools
import math
import os
from os.path import join
import random
from typing import Dict, List
import logging
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from templates import imagenet_templates_small, imagenet_style_templates_small
from textual_inversion_dataset import TextualInversionDataset
from debug_utils import debug_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = get_logger(__name__, log_level="INFO")

# add argument parser
parser = argparse.ArgumentParser(description='ADLCV Exercise 6')
parser.add_argument('--config', type=str, help='Path to config file (config_files/...)')
args = parser.parse_args()
with open(args.config, 'r') as f:
    yaml_config = yaml.safe_load(f)

CONFIG = {
    "pretrained_model": "stabilityai/stable-diffusion-2",
    "what_to_teach": "object",  # Choose between "object" or "style"
    "placeholder_token": yaml_config['concept'],  # The token you'll use to trigger your concept #DONE-changes with each TODO: should this be changed for each concept? or can be standalone.
    "initializer_token": yaml_config['init_token'],  # A word that describes your concept # DONE-needed to help it learn about new object TODO dafuq we need this for?
    "learning_rate": yaml_config['learning_rate'],  # 1e-3, 5e-4
    "scale_lr": True,  
    "max_train_steps": yaml_config['max_train_steps'],  # 500, should be 2000
    "save_steps": 100, # 1.2 says save every 100 iterations (originally 250)
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "concept_folder": yaml_config['concept'], # DONE-TODO: Change this to your concept folder,  sec 1.1 Concept Preparation
    "mode": yaml_config['mode'],
}
# Automatically set output_dir based on concept_folder
lr_str = str(CONFIG["learning_rate"]).replace(".", "_")
CONFIG["output_dir"] = join("outputs", "output_" + CONFIG["concept_folder"], f"lr_{lr_str}")

# os.makedirs(CONFIG["concept_folder"], exist_ok=True) # why make it if it needs to be filled???
os.makedirs(CONFIG["output_dir"], exist_ok=True)

if not os.listdir(CONFIG["concept_folder"]):
    raise ValueError(
        f"The concept folder '{CONFIG['concept_folder']}' is empty! "
        "Please add 3-5 images of your concept before running the training."
    )


def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """Create a grid of images."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def setup_model_and_tokenizer(config: Dict) -> tuple:
    """Setup the model components and tokenizer."""
    tokenizer = CLIPTokenizer.from_pretrained(config["pretrained_model"], subfolder="tokenizer")
    
    # Add placeholder token
    num_added_tokens = tokenizer.add_tokens(config["placeholder_token"])
    if num_added_tokens == 0:
        raise ValueError(f"Token {config['placeholder_token']} already exists!")
        
    # Get token ids
    token_ids = tokenizer.encode(config["initializer_token"], add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("Initializer token must be a single token!")
        
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(config["placeholder_token"])
    
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(config["pretrained_model"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config["pretrained_model"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["pretrained_model"], subfolder="unet")
    
    # Initialize placeholder token
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
    
    return tokenizer, text_encoder, vae, unet, placeholder_token_id

def freeze_models(text_encoder, vae, unet):
    """Freeze all parameters except the token embeddings."""
    def freeze_params(params):
        for param in params:
            param.requires_grad = False
            
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

def create_dataloader(batch_size, tokenizer, config=CONFIG, repeats=100):
    """Create the training dataloader."""
    train_dataset = TextualInversionDataset(
        data_root=config["concept_folder"],
        tokenizer=tokenizer,
        size=512,
        placeholder_token=config["placeholder_token"],
        repeats=repeats,
        learnable_property=config["what_to_teach"],
        center_crop_prob=0.5,  # 50% chance of center cropping
        flip_prob=0.5,  # 50% chance of flipping
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

def get_gpu_memory_info():
    """Get current and peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    current = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    peak = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    return current, peak

# to compare embeddings
# TODO: this isnt what was asked - remove?
def compare_prompt_embeddings(pipe, prompt1, prompt2):
    with torch.no_grad():
        # Tokenize
        ids1 = pipe.tokenizer(prompt1, return_tensors="pt").input_ids.to(pipe.device)
        ids2 = pipe.tokenizer(prompt2, return_tensors="pt").input_ids.to(pipe.device)

        # Encode text
        emb1 = pipe.text_encoder(ids1)[0].mean(dim=1)  # shape: (batch=1, hidden_dim)
        emb2 = pipe.text_encoder(ids2)[0].mean(dim=1)

        # Compute L2 distance
        l2_dist = torch.norm(emb1 - emb2, p=2)
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(emb1, emb2)

    return l2_dist.item(), cos_sim.item()


def training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id):
    train_batch_size = CONFIG["train_batch_size"]
    gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"]
    learning_rate = CONFIG["learning_rate"]
    max_train_steps = CONFIG["max_train_steps"]
    output_dir = CONFIG["output_dir"]
    gradient_checkpointing = CONFIG["gradient_checkpointing"]

    # Initialize peak memory tracking
    peak_memory = 0
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=CONFIG["mixed_precision"]
    )

    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    train_dataloader = create_dataloader(train_batch_size, tokenizer)
    train_dataset = train_dataloader.dataset

    if CONFIG["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )

    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()

    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model"], subfolder="scheduler")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

   ### TODO: Implement the training loop here for Section 1.2 Embedding Training
   ### 
   ### You need to:
   ### 1. Loop through epochs and batches
   ### 2. Process images through VAE to get latents
   ### 3. Add noise to latents using the noise scheduler
   ### 4. Get text embeddings from the text encoder
   ### 5. Predict noise with UNet and calculate loss
   ### 6. Update only the embeddings for the placeholder token
   ### 7. Save checkpoints at specified intervals
   ###
   ### Refer to the main.py file for implementation details
   # ...
   #########################################################
    """
    what should we experiment with? (hyperparams)
    lr
    bs
    optim
    dropout
    num_diff_steps
    noise schedule
    
    EXTRA TASKS MISSING
    Train 3 embedding vectors (check corrsponding loss curves showing convergence) 
    and include 10 generated images per concept demonstrating compositional generalization. 
    Compare two training configurations (e.g., η = 1e-3 vs 5e-4) 
    with quantitative metrics (FID, CLIP-score).
    """

    losses = []

    for epoch in range(num_train_epochs):
        text_encoder.train()  # we only train the embedding

        for i, batch in enumerate(train_dataloader):
            # Each batch is a dict with "pixel_values" and "input_ids"
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215  # recommended scaling for stable diffusion

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device
                ).long()

                # Add noise
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                # Only the embedding for <my-concept> is trainable
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states = encoder_hidden_states.to(unet.dtype)

                # Predict the noise residual with UNet
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Depending on the scheduler config, we may predict either noise or v
                if noise_scheduler.config.prediction_type == "epsilon":     # default
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError("Unknown prediction type")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                losses.append(loss.item())
                # backprop on the accelerator object
                accelerator.backward(loss)

                # Gradient clipping (||∇||_2 <= 1.0)
                accelerator.clip_grad_norm_(
                    text_encoder.get_input_embeddings().parameters(), 1.0
                )

                optimizer.step()
                optimizer.zero_grad()

            # -------------------------
            #   Checkpoint Saving
            # -------------------------
            # DONE TODO: save at *specified* intervals - isnt this from config?
            if (global_step + 1) % CONFIG["save_steps"] == 0 and accelerator.is_main_process:
                # Save the embedding / text_encoder
                ckpt_path = os.path.join(output_dir, f"embedding_ckpt_step_{global_step+1}.pt")
                # Save *just* the embeddings or the whole text_encoder
                torch.save(
                    {
                        "placeholder_token_id": placeholder_token_id,
                        "embedding_weight": text_encoder.get_input_embeddings().weight[placeholder_token_id].detach().cpu(),
                    },
                    ckpt_path,
                )
                logger.info(f"Saved checkpoint to {ckpt_path}")
                #####


            global_step += 1
            progress_bar.update(1)

            if global_step >= max_train_steps:
                # # if we break the training, we save the last model checkpoint
                # ckpt_path = os.path.join(output_dir, f"embedding_ckpt_step_{global_step+1}.pt")
                # torch.save(
                #     {
                #         "placeholder_token_id": placeholder_token_id,
                #         "embedding_weight": text_encoder.get_input_embeddings().weight[placeholder_token_id].detach().cpu(),
                #     },
                #     ckpt_path,
                # )
                break

        # seems redundant, since this loop never increases global_step
        if global_step >= max_train_steps:
            break

    # save the losses to output_dir
    # TODO: update to either be standalone or with FID / CLIP-score
    torch.save(losses, os.path.join(output_dir, "losses.pt"))

    logger.info(f"Training completed. Peak GPU memory usage: {peak_memory:.2f}GB")

def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """Helper function to save the trained embeddings."""
    logger = get_logger(__name__)
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {CONFIG["placeholder_token"]: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)



def main():
    print(f"Starting textual inversion training...")
    print(f"Using concept images from: {CONFIG['concept_folder']}")
    print(f"Number of concept images: {len(os.listdir(CONFIG['concept_folder']))}")
    
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    # Setup
    tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
    
    # Debug dataloader before training
    debug_dataloader(tokenizer, CONFIG)
    
    # Continue with training
    freeze_models(text_encoder, vae, unet)
    
    # Train
    # TODO: should we really train every time? add flag for train/use pre-tuned local model.
    if CONFIG['mode'] == 'train':
        training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
        
        # Save the final model
        pipeline = StableDiffusionPipeline.from_pretrained(
            CONFIG["pretrained_model"],
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(CONFIG["output_dir"])
        print(f"Training completed. Model saved to {CONFIG['output_dir']}")

    # check for no pretrained model in output_dir
    else:
        if not os.path.exists(CONFIG["output_dir"]):
            raise ValueError(f"No trained model found in {CONFIG['output_dir']}. Please train first.")
        else:
            print(f"Using pre-trained model from {CONFIG['output_dir']}")

    # Copy concept folder images as a grid in the output folder
    print("Creating a grid of concept images...")
    concept_images = []
    for image_file in os.listdir(CONFIG["concept_folder"]):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_path = os.path.join(CONFIG["concept_folder"], image_file)
            try:
                img = Image.open(image_path).convert('RGB')
                concept_images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    
    if concept_images:
        # Calculate grid dimensions
        num_images = len(concept_images)
        cols = min(4, num_images)  # Max 4 columns
        rows = math.ceil(num_images / cols)
        
        # Pad with blank images if needed
        while len(concept_images) < rows * cols:
            blank = Image.new('RGB', concept_images[0].size, color=(255, 255, 255))
            concept_images.append(blank)
        
        # Create and save the grid
        concept_grid = image_grid(concept_images, rows, cols)
        concept_grid_path = os.path.join(CONFIG["output_dir"], "concept_images_grid.png")
        concept_grid.save(concept_grid_path)
        print(f"Concept images grid saved to {concept_grid_path}")
    else:
        print("No valid images found in the concept folder to create a grid.")


    ### 10 generated images per concept demonstrating compositional generalization
    # get the device (cude, mps or cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    prompts = [
        f"A {CONFIG['placeholder_token']} placed on a wooden table.",
        f"A {CONFIG['placeholder_token']} inside a cozy living room.",
        f"A {CONFIG['placeholder_token']} on top of a mountain.",
        f"A {CONFIG['placeholder_token']} inside a glass jar filled with colorful marbles.",
        f"A child holding a {CONFIG['placeholder_token']} while walking through a park.",
        f"A {CONFIG['placeholder_token']} in outer space.",
        f"A detailed macro shot of a {CONFIG['placeholder_token']} under a microscope.",
        f"A surreal dreamscape with floating {CONFIG['placeholder_token']} surrounded by mist.",
        f"A {CONFIG['placeholder_token']} in a painting inspired by Van Gogh's Starry Night.",
        f"Looking up at a tall {CONFIG['placeholder_token']} in a dense forest.",
    ]
    folder_name = "compositional_generalization"
    save_dir = os.path.join(CONFIG["output_dir"], folder_name)

    os.makedirs(save_dir, exist_ok=True)
    # Generate images for each prompt
    generated_images = []
    for i, prompt in enumerate(prompts):
        # avoid if already generated
        filename = prompt.replace(" ", "_").replace("'", "").replace(".", "") + ".png"
        if os.path.exists(os.path.join(save_dir, filename)):
            print(f"Skipping prompt {i+1}: '{prompt}' as it's already generated.")
            continue


        print(f"Generating images for prompt {i+1}: '{prompt}'")
        result = pipeline(
            prompt,
            num_inference_steps=40, # TODO??
            guidance_scale=7.5, # TODO??
            num_images_per_prompt=1,
        )
        generated_image = result.images[0] if result.images else None
        generated_images.append(generated_image)
        if generated_image:
            generated_image.save(os.path.join(save_dir, filename))


    # Save the generated images as a grid
    if len(generated_images) == 10:
        generated_grid = image_grid(generated_images, 5, 2)
        generated_grid_path = os.path.join(CONFIG["output_dir"],
                                           "generated_images_grid.png")
        generated_grid.save(generated_grid_path)
        print(f"Generated images grid saved to {generated_grid_path}")

    # load the generated images anew
    generated_images = []
    for image_file in os.listdir(save_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_path = os.path.join(save_dir, image_file)
            try:
                img = Image.open(image_path).convert('RGB')
                generated_images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    # Compare generated with real images using FID and CLIP score

    # The datasets
    # Create datasets for real and generated images
    real_dataloader = create_dataloader(len(concept_images),
                                         tokenizer,
                                         repeats=1,
    )
    #real_dataset = real_dataloader.dataset
    generated_dataloader = create_dataloader(len(generated_images),
                                             tokenizer,
                                             config={
                                                    "concept_folder": save_dir,
                                                    "placeholder_token": CONFIG["placeholder_token"],
                                                    "what_to_teach": CONFIG["what_to_teach"]
                                             },
                                             repeats=1,
    )
    #generated_dataset = generated_dataloader.dataset
    
    # extract the images
    real_images = []
    for batch in real_dataloader:
        real_images.extend(batch["pixel_values"].view(-1, 3, 512, 512))
    real_images = torch.stack(real_images)
    generated_images = []
    for batch in generated_dataloader:
        generated_images.extend(batch["pixel_values"].view(-1, 3, 512, 512))
    generated_images = torch.stack(generated_images)

    # Calculate FID
    from eval_func import VGG, vgg_transform, get_features, feature_statistics, frechet_distance
    vgg_model = VGG()
    vgg_model.to(device)
    vgg_model.eval()
    vgg_model.load_state_dict(torch.load('weights/vgg-sprites/model.pth', map_location=device))
    dims = 256 # vgg feature dim

    # for real
    real_images = torch.stack([vgg_transform(np.array(img, dtype=np.float32)) for img in real_images])
    real_images = real_images.to(device)
    real_images = real_images.view(-1, 3, 512, 512)
    real_features = get_features(vgg_model, real_images)
    mu_real, sigma_real = feature_statistics(real_features)

    # for generated
    generated_images = torch.stack([vgg_transform(np.array(img, dtype=np.float32)) for img in generated_images])
    # convert to correct device
    generated_images = generated_images.to(device)
    generated_images = generated_images.view(-1, 3, 512, 512)
    generated_features = get_features(vgg_model, generated_images)
    mu_gen, sigma_gen = feature_statistics(generated_features)

    fid = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"FID: {fid:.4f}")
    

    # CLIP score
    from eval_func import calculate_clip_score
    # Initialize device and model
    from transformers import CLIPProcessor, CLIPModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Calculate CLIP score
    clip_scores = []
    print(generated_images.shape)
    for i, prompt in enumerate(prompts):
        # get the image
        img = generated_images[i]
        # add batch dimension
        img = img.unsqueeze(0)
        score = calculate_clip_score(img, prompt, clip_model, processor, device)
        clip_scores.append(score)

    print(f"CLIP scores: {np.mean(clip_scores)}")

    # save the FID and CLIP scores as txt
    with open(os.path.join(CONFIG["output_dir"], "scores.txt"), "w") as f:
        f.write(f"FID: {fid:.4f}\n")
        f.write(f"CLIP scores: {np.mean(clip_scores)}\n")


    raise NotImplementedError("TODO: Implement the rest of the main function")

    # 1.3 Concept Generation
    #  INSTRUCTIONS:
    # TODO: Implement the concept generation section here
    # 
    # In this section, you will generate example images using your trained model.
    # This helps evaluate how well your model learned the concept.
    #
    # 1. First, load the trained pipeline from your output directory
    # 2. Configure the DPMSolverMultistepScheduler for efficient sampling
    # 3. Move the model to GPU using the .to("cuda") method
    # 4. Create a test prompt that includes your placeholder token
    # 5. Generate a small batch of images (2 samples is sufficient)
    # 6. Arrange the generated images in a grid for easy viewing
    # 7. Save the grid to your output directory
    #
    # Parameters to experiment with:
    # - num_inference_steps: Higher values (30-50) give better quality but take longer
    # - guidance_scale: Values between 7-9 typically work well
    # - Try different prompts to see how your concept generalizes
    #
    # IMPORTANT: Make sure your GPU has enough memory before running this section!

    # 1.3 Concept Generation
    print("Loading the trained pipeline from your output directory for concept generation...")
    # We'll load it at float16 if your GPU supports that (matching your 'mixed_precision')
    weight_dtype = torch.float16 if CONFIG["mixed_precision"] == "fp16" else torch.float32

    # 1) Load the trained pipeline
    inference_pipeline = StableDiffusionPipeline.from_pretrained(
        CONFIG["output_dir"],
        torch_dtype=weight_dtype
    ).to("cuda")

    # 2) Configure DPMSolverMultistepScheduler
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(inference_pipeline.scheduler.config)
    inference_pipeline.scheduler = dpm_scheduler

    # 3) Move the model to GPU: (already done by .to("cuda") above)

    # 4) Create a test prompt with your placeholder token
    prompt = f"A {CONFIG['placeholder_token']} on a sand dune."

    # We'll generate images at two guidance scales to see the difference
    guidance_scales = [7.5, 15.0]

    all_images = []
    for scale in guidance_scales:
        print(f"Generating images at guidance_scale={scale} for prompt: '{prompt}'")
        
        # 5) Generate images
        result = inference_pipeline(
            prompt,
            num_inference_steps=50,       # tweak as desired
            guidance_scale=scale,
            num_images_per_prompt=10       # need 10 images per concept
        )
        generated_images = result.images

        # Collect them for a grid
        all_images.extend(generated_images)

    # 6) Arrange the generated images in a grid.
    #    We have 2 images for each of the 2 scales => 4 images total.
    grid = image_grid(all_images, rows=2, cols=10)
    grid_path = os.path.join(CONFIG["output_dir"], "concept_generation_grid.png")
    grid.save(grid_path)
    print(f"Saved concept generation grid to: {grid_path}")

    # 7) Compare embeddings with a 'baseline' token (e.g., "toy") using L2 and cosine similarity
    baseline_token = "cat"  # or "cat", "dog", or any baseline you'd like to compare
    baseline_prompt = f"A {baseline_token} on a sand dune."

    l2_distance, cosine_sim = compare_prompt_embeddings(inference_pipeline, prompt, baseline_prompt)
    print(f"Comparison of learned concept vs. baseline token:\n"
        f"  Prompt1: '{prompt}'\n"
        f"  Prompt2: '{baseline_prompt}'\n"
        f"  L2 distance = {l2_distance:.4f}\n"
        f"  Cosine similarity = {cosine_sim:.4f}\n")



if __name__ == "__main__":
    main()

    """
    1.2 
    train 3 embedding vectors (the 3 concepts). 10 images each
        Ring
        Bracelet with very small text on it
        Something we already know should be in the training set, Mario, Sonic, etc.
    try out 1e-3 vs 5e-4

    1.3
    Run experiments with
        num_inference_steps: Higher values (30-50) give better quality but take longer
        guidance_scale: Values between 7-9 typically work well
        Try different prompts to see how your concept generalizes
    """