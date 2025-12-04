DATASET_INSTANCE_INDICES = [0, 1]
INITIALIZER = "cold"  # assume same initializer for now
UNMASKER_PER_DATASET = {
    "cube": "direct",
    "afa_context": "direct",
    "synthetic_mnist": "28x28_to_4x4",
}
EVAL_DATASET_SPLIT = "val"  # whether to use "val" or "test" during evaluation
DEVICE = "cpu"

# Main parameters that we investigate
# - hard budget
# - soft budget parameter during training
PARAMS_PER_METHOD_AND_DATASET = {
    "shim2018": {
        "cube": [
            (5, "null"),
            (10, "null"),
            ("null", 0.4),
            ("null", 0.6),
        ],
        "afa_context": [
            (10, "null"),
            (15, "null"),
            ("null", 0.5),
            ("null", 0.7),
        ],
        "synthetic_mnist": [
            (4, "null"),
            (10, "null"),
            ("null", 0.2),
            ("null", 0.3),
        ],
    },
}
DATASETS = UNMASKER_PER_DATASET.keys()
METHODS = PARAMS_PER_METHOD_AND_DATASET.keys()


rule all:
    input:
        "extra/plot_results/RL",

rule pretrain_model:
    input:
        "extra/data/datasets/{dataset}/{dataset_instance_idx}",
    output:
        directory(
            "extra/pretrained_models/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}+"
        ),


rule train_method:
    input:
        "extra/data/datasets/{dataset}/{dataset_instance_idx}",

        directory(
            "extra/pretrained_models/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}+"
        ),
    output:
        directory(
            "extra/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}"
        ),
    params:
        unmasker=lambda wildcards: UNMASKER_PER_DATASET[wildcards.dataset],
    shell:
        """
        python scripts/train/{wildcards.method}.py \
            dataset_artifact_path={input} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            train_hard_budget={wildcards.train_hard_budget} \
            train_soft_budget_param={wildcards.train_soft_budget_param} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            use_wandb=false
        """


# rule train_classifier:
#     input:
#         "extra/data/datasets/{dataset}/{dataset_instance_idx}",
#     output:
#         directory(
#             "extra/trained_classifiers/masked_mlp_classifier/dataset-{dataset}+instance_idx-{dataset_instance_idx}"
#         ),
#     shell:
#         """
#         python scripts/train/masked_mlp_classifier.py \
#             dataset_artifact_path={input} \
#         """


rule eval_method:
    input:
        "extra/data/datasets/{dataset}/{dataset_instance_idx}",

        "extra/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "train_hard_budget-{hard_budget}+"
                "train_soft_budget_param-{soft_budget_param}",
        # "extra/trained_classifiers/masked_mlp_classifier/dataset-{dataset}+instance-{dataset_instance}",
    output:
        "extra/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "train_hard_budget-{hard_budget}+"
                "train_soft_budget_param-{soft_budget_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{hard_budget}/"
                        "eval_data.csv",
    params:
        save_path=lambda wildcards: (
            f"extra/eval_results/{wildcards.method}/"
                f"dataset-{wildcards.dataset}+"
                f"instance_idx-{wildcards.dataset_instance_idx}/"
                    f"train_seed-{wildcards.train_seed}+"
                    f"train_hard_budget-{wildcards.hard_budget}+"
                    f"train_soft_budget_param-{wildcards.soft_budget_param}/"
                        f"eval_seed-{wildcards.eval_seed}+"
                        f"eval_hard_budget-{wildcards.hard_budget}"
        ),
        unmasker=lambda wildcards: UNMASKER_PER_DATASET[wildcards.dataset],
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_artifact_path={input[1]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_artifact_path={input[0]} \
            save_path={params.save_path} \
            dataset_split={EVAL_DATASET_SPLIT} \
            classifier_artifact_path=null \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            eval_only_n_samples=null \
            batch_size=1 \
            hard_budget={wildcards.hard_budget} \
            use_wandb=false
        """


rule transform_eval:
    input:
        "extra/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "train_hard_budget-{hard_budget}+"
                "train_soft_budget_param-{soft_budget_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{hard_budget}/"
                        "eval_data.csv",
    output:
        "extra/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "train_hard_budget-{hard_budget}+"
                "train_soft_budget_param-{soft_budget_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{hard_budget}/"
                        "transformed_eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py {input} {output} {wildcards.method} \
            --dataset {wildcards.dataset} \
            --train_seed {wildcards.train_seed} \
            --eval_seed {wildcards.eval_seed} \
            --hard_budget {wildcards.hard_budget} \
            --soft_budget_param {wildcards.soft_budget_param}
        """


rule merge_eval:
    input:
        [
            (
                f"extra/eval_results/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"train_seed-{dataset_instance_idx}+"
                        f"train_hard_budget-{hard_budget}+"
                        f"train_soft_budget_param-{soft_budget_param}/"
                            f"eval_seed-{dataset_instance_idx}+"
                            f"eval_hard_budget-{hard_budget}/"
                                f"transformed_eval_data.csv"
            )
            for method in METHODS
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                hard_budget,
                soft_budget_param,
            ) in PARAMS_PER_METHOD_AND_DATASET[
                method
            ][dataset]
        ],
    output:
        "extra/merged_eval_results/dummy.csv",
    shell:
        """
            csvstack {input} > {output}
        """


rule plot:
    input:
        "extra/merged_eval_results/dummy.csv",
    output:
        directory("extra/plot_results/dummy"),
    shell:
        """
            Rscript scripts/plotting/plot_eval.R {input} {output}
        """
