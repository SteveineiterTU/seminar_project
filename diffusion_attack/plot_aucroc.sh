#!/bin/bash

cd attack_procedure
python eval_roc.py \
--attack_type "pbb" \
-ldir "results/pbb/64_images_model"