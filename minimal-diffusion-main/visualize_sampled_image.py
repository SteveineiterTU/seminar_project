import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

INDEX_OF_IMAGE_TO_VISUALIZE = 69
HAS_LABELS = False

with np.load(
        "sampled_images/UNet_celeba-500-sampling_steps-100_images-class_condn_False.npz"
) as data:
    images = data["arr_0"]
    if HAS_LABELS:
        labels = data["arr_1"]

image = images[INDEX_OF_IMAGE_TO_VISUALIZE].reshape(64, 64, 3)
if HAS_LABELS:
    label = labels[INDEX_OF_IMAGE_TO_VISUALIZE]
    plt.title(f"Label is {label}".format(label=label))

plt.imshow(image, cmap="gray")
plt.show()

