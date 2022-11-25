# My implementation of https://huggingface.co/blog/annotated-diffusion.
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image

from constants_and_variables import TIME_STEPS, SAVE_AND_SAMPLE_MILESTONE
import matplotlib.animation as animation
from diffusion import p_losses, sample
from helpers import num_to_groups
from u_net import UNet

"""  
========================================================================================================================
                                         Diffusion using MNIST  
========================================================================================================================
"""
IMAGES_WILL_BE_SHOWN = False


# ======================================== Creating Data ===============================================================
def dataloader():
    dataset = load_dataset("mnist")
    batch_size = 128

    transformed_dataset = dataset.with_transform(_transforms).remove_columns("label")
    _dataloader = DataLoader(
        transformed_dataset["train"], batch_size=batch_size, shuffle=True
    )
    # batch = next(iter(dataloader))
    # print(batch.keys())
    return _dataloader


def _transforms(examples):
    transform = Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples["image"]
    ]
    del examples["image"]

    return examples


# ======================================== Training ====================================================================
def train():
    image_size = 28
    channels = 1
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _model = UNet(dimension=image_size, channels=channels, dimension_multiplies=(1, 2, 4))
    _model.to(device)
    optimizer = Adam(_model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            time_step = torch.randint(0, TIME_STEPS, (batch_size,), device=device).long()
            loss = p_losses(_model, batch, time_step, loss_type="huber")
            if step % 100 == 0:
                print("Loss:", loss.item())
            loss.backward()

            optimizer.step()

            if step != 0 and step % SAVE_AND_SAMPLE_MILESTONE == 0:
                milestone = step // SAVE_AND_SAMPLE_MILESTONE
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sample(_model, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)
    return _model


# ======================================== Sampling ====================================================================
def sampling(_model):
    image_size = 28
    channels = 1

    _samples = sample(_model, image_size=image_size, batch_size=64, channels=channels)
    random_index = 5
    if IMAGES_WILL_BE_SHOWN:
        plt.imshow(_samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")
    return _samples


def animated_sampling(_samples):
    random_index = 53
    image_size = 28
    channels = 1

    fig = plt.figure()
    ims = []
    for i in range(TIME_STEPS):
        im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()


if __name__ == '__main__':
    dataloader = dataloader()
    model = train()
    samples = sampling(model)
    animated_sampling(samples)
