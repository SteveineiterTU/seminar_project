import torch
import torch.nn.functional as F

from helpers import linear_beta_schedule

# ======================================== Consts ======================================================================
TIME_STEPS = 250
SAVE_AND_SAMPLE_MILESTONE = 1000

# ======================================== Variables ===================================================================
betas = linear_beta_schedule(time_steps=TIME_STEPS)

alphas = 1.0 - betas
alphas_cumulative_product = torch.cumprod(alphas, axis=0)
alphas_cumulative_product_previous = F.pad(alphas_cumulative_product[:-1], (1, 0), value=1.0)
square_root_of_recip_alphas = torch.sqrt(1.0 / alphas)

# Calculations for Diffusion q(x_t | x_{t-1}) and Others
square_root_of_alphas_cumulative_product = torch.sqrt(alphas_cumulative_product)
square_root_of_one_minus_alphas_cumulative_product = torch.sqrt(1 - alphas_cumulative_product)

# Calculations for Posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1 - alphas_cumulative_product_previous) / (1 - alphas_cumulative_product)




