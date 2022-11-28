import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

INDEX_OF_IMAGE_TO_VISUALIZE = 69

with np.load(
    "./sampled_images/UNet_mnist-100-sampling_steps-50000_images-class_condn_True.npz"
) as data:
    images = data["arr_0"]
    labels = data["arr_1"]

image = images[INDEX_OF_IMAGE_TO_VISUALIZE].reshape(28, 28)
label = labels[INDEX_OF_IMAGE_TO_VISUALIZE]

plt.title(f"Label is {label}".format(label=label))
plt.imshow(image, cmap="gray")
plt.show()

