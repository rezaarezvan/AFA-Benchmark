
# Shim2018AFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='shim2018' \
    --afa_method_path='models/shim2018/shim2018-cube_train_split_1.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/shim2018/shim2018-cube_train_split_1-cube_val_split_1.pt'
echo "-------------------------------"

# Zannone2019AFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='zannone2019' \
    --afa_method_path='models/zannone2019/zannone2019-cube_train_split_1.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/zannone2019/zannone2019-cube_train_split_1-cube_val_split_1.pt'
echo "-------------------------------"
