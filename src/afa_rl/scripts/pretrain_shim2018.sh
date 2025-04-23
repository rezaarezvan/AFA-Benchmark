for split in {1..5}; do
    echo "Training agent for split ${split}..."
    uv run src/afa_rl/scripts/pretrain_shim2018.py \
        --config configs/afa_rl/pretrain_shim2018.yml \
        --dataset_type "cube" \
        --dataset_train_path "data/cube/train_split_${split}.pt" \
        --dataset_val_path "data/cube/val_split_${split}.pt" \
        --save_path "models/shim2018/shim2018-cube_train_split_${split}"
    echo "Finished training agent for split ${split}."
done
