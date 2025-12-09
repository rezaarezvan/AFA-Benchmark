# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    # Format
    uv run ruff format .

    # Linting
    uv run ruff check . --fix

    # LSP checks
    pre-commit run basedpyright --all-files

    # Testing
    uv run pytest . --tb=no

# Run coverage, and build to HTML
coverage:
    uv run coverage run -m pytest .
    uv run coverage report -m
    uv run coverage html

# Build the project, useful for checking that packaging is correct
# build:
#     rm -rf build
#     rm -rf dist
#     uv build

pretrain_shim2018_cube *extra_args='':
    uv run scripts/pretrain/shim2018.py {{extra_args}} \
                    train_dataset_bundle_path=extra/data/datasets/cube/0/train.bundle/ \
                    val_dataset_bundle_path=extra/data/datasets/cube/0/val.bundle/ \
                    save_path=tmp/shim2018_pretrained_cube.bundle \
                    device=cpu \
                    seed=42 \
                    use_wandb=true \
                    +experiment@_global_=cube

train_shim2018_cube_hard *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_cube.bundle \
        save_path=tmp/shim2018_trained_cube_hard.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        hard_budget=5 \
        soft_budget_param=null \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=cube

train_shim2018_cube_soft *extra_args='':
    uv run scripts/train/shim2018.py {{extra_args}} \
        train_dataset_bundle_path=extra/data/datasets/cube/0/train.bundle \
        val_dataset_bundle_path=extra/data/datasets/cube/0/val.bundle \
        pretrained_model_bundle_path=tmp/shim2018_pretrained_cube.bundle \
        save_path=tmp/shim2018_trained_cube_soft.bundle \
        components/initializers@initializer=cold \
        components/unmaskers@unmasker=direct \
        hard_budget=null \
        soft_budget_param=0.5 \
        device=cpu \
        seed=42 \
        use_wandb=true \
        +experiment@_global_=cube
