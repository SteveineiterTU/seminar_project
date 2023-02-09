#!/bin/bash

cd attack_procedure
python pbb_ddim_diffusion.py \
-name "output" \
-posdir "../data/256_images/img_align_celeba_256_samples/class0" \
-negdir "../data/256_images/img_align_celeba_minus_256_images" \
-gdir "../models/training_set_size_256/UNet_celeba_reduced-total_epochs_101-epoch_100-timesteps_1000-class_condn_False.pt" \
-init random \
--distance l2 \
--data_num 16 \
--exp_name "256_images_model"

