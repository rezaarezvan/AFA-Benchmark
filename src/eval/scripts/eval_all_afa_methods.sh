# SequentialDummyAFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/afa_rl/sequential_dummy-cube_train_10000.pt' \
    --dataset_name='cube' \
    --dataset_path='data/cube/cube_val_100.pt' \
    --eval_save_path='results/evaluation/sequential_dummy-cube_train_10000-cube_val_100.pt'
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/afa_rl/sequential_dummy-cube_train_10000.pt' \
    --dataset_name='cube' \
    --dataset_path='data/cube/cube_val_1000.pt' \
    --eval_save_path='results/evaluation/sequential_dummy-cube_train_10000-cube_val_1000.pt'
echo "-------------------------------"

# RandomDummyAFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/afa_rl/random_dummy-cube_train_10000.pt' \
    --dataset_name='cube' \
    --dataset_path='data/cube/cube_val_100.pt' \
    --eval_save_path='results/evaluation/random_dummy-cube_train_10000-cube_val_100.pt'
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/afa_rl/random_dummy-cube_train_10000.pt' \
    --dataset_name='cube' \
    --dataset_path='data/cube/cube_val_1000.pt' \
    --eval_save_path='results/evaluation/random_dummy-cube_train_10000-cube_val_1000.pt'
echo "-------------------------------"

# Don't forget to list all evaluation results (eval_save_path) in the registry.
