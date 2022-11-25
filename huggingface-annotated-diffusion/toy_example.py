import numpy as np
import requests
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import (
    Compose,
    ToTensor,
    Lambda,
    ToPILImage,
    CenterCrop,
    Resize,
)


# ======================================== Consts ======================================================================
from diffusion import q_sample

IMAGES_WILL_BE_SHOWN = False
IMAGE_SIZE = 128
# ======================================== Variables ===================================================================
transform = Compose(
    [
        Resize(IMAGE_SIZE),
        CenterCrop(IMAGE_SIZE),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),
    ]
)

reverse_transform = Compose(
    [
        Lambda(lambda tensor: (tensor + 1) / 2),
        Lambda(lambda tensor: tensor.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda tensor: tensor * 255.0),
        Lambda(lambda tensor: tensor.numpy().astype(np.uint8)),
        ToPILImage(),
    ]
)


# ======================================== Diffusion ===================================================================
def get_noisy_image(_x_start, _time_step):
    x_noisy = q_sample(_x_start, _time_step=_time_step)
    noisy_image = reverse_transform(x_noisy.squeeze())
    return noisy_image


# ======================================== Illustration ================================================================
def illustration_of_multiple_time_steps(_x_start, _time_step):
    torch.manual_seed(69)
    plot([get_noisy_image(_x_start, torch.tensor([t])) for t in _time_step])


def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    # source:
    #   https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(
        figsize=(200, 200), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


# ======================================== Illustration on cats image ==================================================
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
if IMAGES_WILL_BE_SHOWN:
    image.show()

x_start = transform(image).unsqueeze(0)
print(x_start.shape)

if IMAGES_WILL_BE_SHOWN:
    reverse_transform(x_start.squeeze()).show()

# Can be changed to show different time steps of the diffusion process, eg 100 is just noise.
time_step = torch.tensor([30])

if IMAGES_WILL_BE_SHOWN:
    get_noisy_image(x_start, time_step).show()

time_steps = [0, 25, 50, 75, 100, 150, 200]
# illustration_of_multiple_time_steps(time_steps) # plt bugged, and I don't care actually xD
