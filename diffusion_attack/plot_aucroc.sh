#!/bin/bash

cd attack_procedure
python eval_roc.py \
--attack_type "wb" \
-ldir "results/wb/wb_64_images_model_l2_test"