# DDPM lab

## Installation
* `conda create -n ddpm python=3.10.10`
* `conda activate ddpm`
* `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
* `pip install -r requirements.txt`

## Scripts
* `playground.py`: It contains the first three tasks to familiarize yourself with the project.
* `ddpm.py`: In this file, we implement all the functionality for the Denoising Diffusion Probabilistic Models.
* `ddpm_train.py`: In this file, we train a UNet diffusion model.

## Notation
In the lecture, we follow the notation of the [ddpm paper](https://arxiv.org/pdf/2006.11239.pdf), while in the code, we follow the notation from the OpenAI code repository. Here we provide a mapping between the two.
* $T$ is the total number of diffusion steps
* $x_t$ = image at timestep t
* $x_T \sim \mathcal{N}(0, \mathbf{I})$
* $\beta_t$ = betas[t]
* $\alpha_t$ = alphas[t]
* $\bar{\alpha}_t$ = alphas_bar[t]
* $q(x_t|x_0)$ = q_sample
* $p_\theta(x_{t-1}|x_t)$ = p_sample

## Logging
* The code uses tensorboard to log the train loss. Use the command `tensorboard --logdir=runs` to observe the training loss.
* When you perform your own experiments (e.g. cosine schedule) make sure that you change the experiment name in the `ddpm_train.py` file.