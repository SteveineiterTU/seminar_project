import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from constants_and_variables import (
    betas,
    square_root_of_one_minus_alphas_cumulative_product,
    square_root_of_recip_alphas,
    posterior_variance,
    TIME_STEPS,
    square_root_of_alphas_cumulative_product,
)

# ======================================== Diffusion ===================================================================
# forward diffusion (using the nice property)
from helpers import extract


def q_sample(_x_start, _time_step, noise=None):
    if noise is None:
        noise = torch.randn_like(_x_start)

    square_root_of_alphas_cumulative_product_t = extract(
        square_root_of_alphas_cumulative_product, _time_step, _x_start.shape
    )
    square_root_of_one_minus_alphas_cumulative_product_t = extract(
        square_root_of_one_minus_alphas_cumulative_product, _time_step, _x_start.shape
    )

    return (
        square_root_of_alphas_cumulative_product_t * _x_start
        + square_root_of_one_minus_alphas_cumulative_product_t * noise
    )


def p_losses(denoise_model, _x_start, _time_step, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(_x_start)
    x_noisy = q_sample(_x_start, _time_step, noise)
    predicted_noise = denoise_model(x_noisy, _time_step)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, time_step, t_index):
    betas_t = extract(betas, time_step, x.shape)
    square_root_of_one_minus_alphas_cumulative_product_t = extract(
        square_root_of_one_minus_alphas_cumulative_product, time_step, x.shape
    )
    square_root_of_recip_alphas_t = extract(
        square_root_of_recip_alphas, time_step, x.shape
    )

    # Equation 11 in the paper. Use our model (noise predictor) to predict the mean. We use betas_t since 1 - alpha_t =
    #    1 - 1 - betas == betas (Stefan thought tho - aka not sure about that).
    model_mean = square_root_of_recip_alphas_t * (
        x
        - betas_t
        / square_root_of_one_minus_alphas_cumulative_product_t
        * model(x, time_step)
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, time_step, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    batch_size = shape[0]
    # start from pure noise (for each example in the batch)
    image = torch.randn(shape, device=device)
    images = []

    for index in tqdm(
        reversed(range(0, TIME_STEPS)), desc="sampling loop time step", total=TIME_STEPS
    ):
        image = p_sample(
            model,
            image,
            torch.full((batch_size,), index, device=device, dtype=torch.long),
            index,
        )
        images.append(image.cpu().numpy())
    return images


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
