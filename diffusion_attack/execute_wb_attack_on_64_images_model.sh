#!/bin/bash

cd attack_procedure
python wb_ddim_diffusion.py \
-name "output" \
-posdir "../data/64_images/img_align_celeba_64_samples/class0" \
-negdir "../data/64_images/img_align_celeba_minus_64_images" \
-gdir "../models/training_set_size_64/UNet_celeba_reduced-total_epochs_101-epoch_100-timesteps_1000-class_condn_False.pt" \
-init random \
--distance l2 \
--data_num 16 \
--exp_name "wb_64_images_model"

