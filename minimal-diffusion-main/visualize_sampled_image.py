import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

INDEX_OF_IMAGE_TO_VISUALIZE = 18
USE_GRAYSCALE = True
HAS_LABELS = False

color_image = "sampled_images/UNet_celeba-500-sampling_steps-100_images-class_condn_False.npz"
gray_scale_image = "/home/stefan/Uni/Master/Semester_3/seminar_project/minimal-diffusion-main/trained_models/UNet_celeba_reduced-50-sampling_steps-50_images-class_condn_False.npz"

with np.load(gray_scale_image) as data:
    images = data["arr_0"]
    if HAS_LABELS:
        labels = data["arr_1"]

if USE_GRAYSCALE:
    image = images[INDEX_OF_IMAGE_TO_VISUALIZE].reshape(32, 32, 1)
else:
    image = images[INDEX_OF_IMAGE_TO_VISUALIZE].reshape(64, 64, 3)

if HAS_LABELS:
    label = labels[INDEX_OF_IMAGE_TO_VISUALIZE]
    plt.title(f"Label is {label}".format(label=label))

plt.imshow(image, cmap="gray")
plt.show()

