#!/bin/bash

python train.py --model_type bert \
    --my_device cuda:0 \
    --pretrained_model_name_or_path ./PLMs \
    --logging_steps 200 \
    --num_train_epochs 100 \
    --learning_rate 2e-5 \
    --num_warmup_steps_or_radios 0.1 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --seed 3407 \
    --contrastive_method event_description \
    --contrastive_level token_level \
    --event_type_num 8 \
    --save_steps 4400 \
    --description_loss_weight 0.001 \
    --use_role_contrast \
    --role_contrast_weight 0.1 \
    --role_contrast_temperature 0.1 \
    --role_contrast_margin 0.5 \
    --output_dir ./outputs/with_role_contrast