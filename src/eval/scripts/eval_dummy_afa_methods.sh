# SequentialDummyAFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/sequential_dummy/sequential_dummy-cube_train_split_1.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_1-cube_val_split_1.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/sequential_dummy/sequential_dummy-cube_train_split_2.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_2-cube_val_split_2.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/sequential_dummy/sequential_dummy-cube_train_split_3.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_3-cube_val_split_3.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/sequential_dummy/sequential_dummy-cube_train_split_4.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_4-cube_val_split_4.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='sequential_dummy' \
    --afa_method_path='models/sequential_dummy/sequential_dummy-cube_train_split_5.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/sequential_dummy/sequential_dummy-cube_train_split_5-cube_val_split_5.pt' \
    --hard_budget=10
echo "-------------------------------"

# RandomDummyAFAMethod
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/random_dummy/random_dummy-cube_train_split_1.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/random_dummy/random_dummy-cube_train_split_1-cube_val_split_1.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/random_dummy/random_dummy-cube_train_split_2.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/random_dummy/random_dummy-cube_train_split_2-cube_val_split_2.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/random_dummy/random_dummy-cube_train_split_3.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/random_dummy/random_dummy-cube_train_split_3-cube_val_split_3.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/random_dummy/random_dummy-cube_train_split_4.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/random_dummy/random_dummy-cube_train_split_4-cube_val_split_4.pt' \
    --hard_budget=10
echo "-------------------------------"
uv run src/eval/scripts/eval_afa_method.py \
    --afa_method_name='random_dummy' \
    --afa_method_path='models/random_dummy/random_dummy-cube_train_split_5.pt' \
    --dataset_type='cube' \
    --dataset_val_path='data/cube/val_split_1.pt' \
    --eval_save_path='results/evaluation/random_dummy/random_dummy-cube_train_split_5-cube_val_split_5.pt' \
    --hard_budget=10
echo "-------------------------------"
