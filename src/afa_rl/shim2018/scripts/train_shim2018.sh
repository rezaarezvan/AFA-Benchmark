#!/usr/bin/bash

uv run src/afa_rl/shim2018/scripts/train_shim2018.py \
    --pretrain_config configs/shim2018/pretrain_shim2018.yml \
    --train_config configs/shim2018/train_shim2018.yml \
    --dataset_type cube \
    --train_dataset_path data/cube/train_split_1.pt \
    --val_dataset_path data/cube/val_split_1.pt \
    --pretrained_model_path models/pretrained/shim2018/temp \
    --hard_budget 10 \
    --seed 42 \
    --afa_method_path models/methods/shim2018/temp
