for split in {1..5}; do
    echo "Training agent for split ${split}..."
    uv run src/afa_rl/scripts/train_shim2018.py \
        --pretrain_config configs/afa_rl/pretrain_shim2018.yml \
        --train_config configs/afa_rl/train_shim2018.yml \
        --dataset_type "cube" \
        --dataset_train_path "data/cube/train_split_${split}.pt" \
        --dataset_val_path "data/cube/val_split_${split}.pt" \
        --pretrained_model_save_path "models/shim2018/pretrained/shim2018-cube_train_split_${split}" \
        --afa_method_save_path "models/shim2018/shim2018-cube_train_split_${split}"
    echo "Finished training agent for split ${split}."
done
