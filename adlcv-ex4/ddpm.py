import torch
from tqdm import tqdm
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, T=500, beta_start=1e-4, beta_end=0.02, img_size=16, device="cuda"):
        """
        T : total diffusion steps (X_T is pure noise N(0,1))
        beta_start: value of beta for t=0
        b_end: value of beta for t=T
        """

        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        # TASK 1: Implement beta, alpha, and alpha_bar
        self.betas = self.get_betas('linear').to(device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0) # cumulative products of alpha 


    def get_betas(self, schedule='linear'):
        if schedule == 'linear':
            return ... # HINT: use torch.linspace to create a linear schedule from beta_start to beta_end
        # add your own (e.g. cosine)
        else :
            raise NotImplementedError('Not implemented!')
    

    def q_sample(self, x, t):
        """
        x: input image (x0)
        t: timestep: should be torch.tensor

        Forward diffusion process
        q(x_t | x_0) = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * N(0,1)

        Should return q(x_t | x_0), noise
        """
        # TASK 2: Implement the forward process
        sqrt_alpha_bar =  ... # HINT: use torch.sqrt to calculate the sqrt of alphas_bar at timestep t
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None] # match image dimensions

        sqrt_one_minus_alpha_bar = ... # HINT: calculate the sqrt of 1 - alphas_bar at time step t
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None, None]# match image dimensions
        
        noise = ... # HINT: sample noise from a normal distribution. It should match the shape of x 
        assert noise.shape == x.shape, 'Invalid shape of noise'
        
        x_noised = ... # HINT: Create the noisy version of x. See Eq. 4 in the ddpm paper at page 2
        return ..., noise
    

    def p_mean_std(self, model, x_t, t):
        """
        Calculate mean and std of p(x_{t-1} | x_t) using the reverse process and model
        """
        alpha = self.alphas[t][:, None, None, None] # match image dimensions
        alpha_bar = self.alphas_bar[t][:, None, None, None] # match image dimensions 
        beta = self.betas[t][:, None, None, None] # match image dimensions

        # TASK 3 : Implement the revese process
        predicted_noise = ... # HINT: use model to predict noise
        mean = ... # HINT: calculate the mean of the distribution p(x_{t-1} | x_t). See Eq. 11 in the ddpm paper at page 4
        std = torch.sqrt(beta)

        return mean, std

    def p_sample(self, model, x_t, t):
        """
        Sample from p(x{t-1} | x_t) using the reverse process and model
        """
        # TASK 3: implement the reverse process
        mean, std = self.p_mean_std(model, x_t, t)
        
        # HINT: Having calculate the mean and std of p(x{x_t} | x_t), we sample noise from a normal distribution.
        # see line 3 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.
        noise = ...

        x_t_prev = ... # Calculate x_{t-1}, see line 4 of the Algorithm 2 (Sampling) at page 4 of the ddpm paper.
        return x_t_prev


    def p_sample_loop(self, model, batch_size, timesteps_to_save=None):
        """
        Implements algrorithm 2 (Sampling) from the ddpm paper at page 4
        """
        logging.info(f"Sampling {batch_size} new images....")
        model.eval()
        if timesteps_to_save is not None:
            intermediates = []
        with torch.no_grad():
            x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.T)), position=0, total=self.T-1):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                x = self.p_sample(model, x, t)
                if timesteps_to_save is not None and i in timesteps_to_save:
                    x_itermediate = (x.clamp(-1, 1) + 1) / 2
                    x_itermediate = (x_itermediate * 255).type(torch.uint8)
                    intermediates.append(x_itermediate)

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        if timesteps_to_save is not None:
            intermediates.append(x)
            return x, intermediates
        else :
            return x


    def sample_timesteps(self, batch_size):
        """
        Sample timesteps uniformly for training
        """
        return torch.randint(low=1, high=self.T, size=(batch_size,), device=self.device)