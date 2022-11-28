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

from constants_and_variables import TIME_STEPS
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

SAVE_AND_SAMPLE_MILESTONE = 1000
IMAGE_SIZE = 28
CHANNELS = 1

MODEL_SAVE_PATH = "./model/unet_5_epochs"
MODEL_LOAD_PATH = "./model/unet_5_epochs"


# ======================================== Creating Data ===============================================================
def dataloader():
    dataset = load_dataset("mnist")
    batch_size = 32

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
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _model = UNet(
        dimension=IMAGE_SIZE, channels=CHANNELS, dimension_multiplies=(1, 2, 4)
    )
    _model.to(device)
    optimizer = Adam(_model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            time_step = torch.randint(
                0, TIME_STEPS, (batch_size,), device=device
            ).long()
            loss = p_losses(_model, batch, time_step, loss_type="huber")
            if step % 100 == 0:
                print("Loss:", loss.item())
            loss.backward()

            optimizer.step()

            # Is bugging and not really necessary.
            # if step != 0 and step % SAVE_AND_SAMPLE_MILESTONE == 0:
            #     milestone = step // SAVE_AND_SAMPLE_MILESTONE
            #     batches = num_to_groups(4, batch_size)
            #     all_images_list = list(
            #         map(
            #             lambda n: sample(
            #                 _model,
            #                 image_size=IMAGE_SIZE,
            #                 batch_size=n,
            #                 channels=CHANNELS,
            #             ),
            #             batches,
            #         )
            #     )
            #     all_images = torch.tensor(all_images_list)
            #     all_images = (all_images + 1) * 0.5
            #     save_image(
            #         all_images, str(results_folder / f"sample-{milestone}.png"), nrow=6
            #     )
    if MODEL_SAVE_PATH is not None:
        torch.save(_model.state_dict(), MODEL_SAVE_PATH)
    return _model


# ======================================== Sampling ====================================================================
def sampling(_model):
    _samples = sample(_model, image_size=IMAGE_SIZE, batch_size=64, channels=CHANNELS)
    random_index = 5
    if IMAGES_WILL_BE_SHOWN:
        plt.imshow(
            _samples[-1][random_index].reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            cmap="gray",
        )
    return _samples


def animated_sampling(_samples):
    random_index = 42  # Max numer should be 63 since we got 64 samples.

    fig = plt.figure()
    ims = []
    for i in range(TIME_STEPS):
        im = plt.imshow(
            samples[i][random_index].reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            cmap="gray",
            animated=True,
        )
        ims.append([im])

    animate = animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )
    animate.save(f"./results/diffusion_{random_index}.gif")
    plt.show()


def load_or_train_model():
    if MODEL_LOAD_PATH is None:
        _model = train()
    else:
        _model = UNet(
            dimension=IMAGE_SIZE, channels=CHANNELS, dimension_multiplies=(1, 2, 4)
        )
        _model.load_state_dict(torch.load(MODEL_LOAD_PATH))
    return _model


if __name__ == "__main__":
    dataloader = dataloader()
    model = load_or_train_model()
    samples = sampling(model)
    animated_sampling(samples)
