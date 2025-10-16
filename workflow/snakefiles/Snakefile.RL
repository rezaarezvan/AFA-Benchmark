configfile: "workflow/conf/RL.yaml"

# Define global variables from config values
EXPERIMENT_ID = config["experiment_id"]
DATASET_ARTIFACT_ALIASES = config["dataset_artifact_aliases"]
EVAL_BATCH_SIZE_PER_DATASET = config["eval_batch_size_per_dataset"]
DATASET_SPLITS = config["dataset_splits"]
DATASET_SPLIT_MODE = config["dataset_split_mode"]
PRETRAINED_MODEL_ALIASES = config["pretrained_model_aliases"]
CLASSIFIER = config["classifier"]
CLASSIFIER_ARTIFACT_ALIASES = config["classifier_artifact_aliases"]
VALIDATE_ARTIFACTS = config["validate_artifacts"]
SEEDS = config["seeds"]
TRAIN_DEVICE = config["train_device"]
EVAL_DEVICE = config["eval_device"]
COST_PARAMS_PER_METHOD_AND_DATASET = config["cost_params_per_method_and_dataset"]
SMOKE_TEST = config["smoke_test"]

if PRETRAINED_MODEL_ALIASES is None:
    PRETRAINED_MODEL_ALIASES = {}
else:
    METHODS_WITH_PRETRAIN_STAGE = set(PRETRAINED_MODEL_ALIASES.keys())
METHODS = list(COST_PARAMS_PER_METHOD_AND_DATASET.keys())
DATASETS = list(DATASET_ARTIFACT_ALIASES.keys())


def get_cost_params(method: str, dataset: str):
    if dataset in COST_PARAMS_PER_METHOD_AND_DATASET[method]:
        return COST_PARAMS_PER_METHOD_AND_DATASET[method][dataset]
    else:
        return COST_PARAMS_PER_METHOD_AND_DATASET[method]["default"]


# def get_cost_params_indices(method: str, dataset: str):
#     return list(range(len(get_cost_params(method, dataset))))


# Fast way to produce soft budget plot
rule soft_budget:
    input:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot1.pdf",
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot2.pdf",
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot3.pdf",


rule soft_budget_train_all:
    input:
        [
            f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/trained_method/{method}/{dataset}/split{dataset_split}/costparam{cost_param}/seed{seed}.done"
            for method in METHODS
            for dataset in DATASETS
            for dataset_split in DATASET_SPLITS
            for cost_param in get_cost_params(method, dataset)
            for seed in SEEDS
        ],


rule soft_budget_eval_all:
    input:
        [
            f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/eval/{method}/{dataset}/split{dataset_split}/costparam{cost_param}/seed{seed}.done"
            for method in METHODS
            for dataset in DATASETS
            for dataset_split in DATASET_SPLITS
            for cost_param in get_cost_params(method, dataset)
            for seed in SEEDS
        ],


def get_pretrain_alias(method: str, dataset: str):
    if dataset in PRETRAINED_MODEL_ALIASES.get(method, {}):
        return PRETRAINED_MODEL_ALIASES[method][dataset]
    elif method in METHODS_WITH_PRETRAIN_STAGE:
        return PRETRAINED_MODEL_ALIASES[method]["default"]


def get_train_dependency_arguments(wildcards):
    """Return an argument that specifies a dependency for training a method.

    Some AFA methods require a pretraining stage, others don't. Also, the naming convention can differ between methods.
    """
    if wildcards.method in {"shim2018", "kachuee2019", "zannone2019"}:
        return f"pretrained_model_artifact_name=pretrain_{wildcards.method}-{wildcards.dataset}_split_{wildcards.dataset_split}:{get_pretrain_alias(wildcards.method, wildcards.dataset)}"
    elif wildcards.method in {"randomdummy", "sequentialdummy"}:
        return f"dataset_artifact_name={wildcards.dataset}_split_{wildcards.dataset_split}:{DATASET_ARTIFACT_ALIASES[wildcards.dataset]}"
    else:
        msg = f"Unknown method {wildcards.method} when trying to determine training dependency argument."
        raise ValueError(msg)


def get_method_specific_extra_args(wildcards):
    """Return any method-specific extra arguments for training."""
    if wildcards.method == "zannone2019":
        return "n_generated_samples=0"
    else:
        return ""


rule train_soft_budget_method:
    output:
        touch(
            f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/trained_method/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}.done"
        ),
    params:
        train_dependency_arguments=get_train_dependency_arguments,
        dataset_specific_args=lambda wildcards: (
            f"dataset@_global_={wildcards.dataset}_smoke"
            if SMOKE_TEST
            else f"dataset@_global_={wildcards.dataset}"
        ),
        method_specific_extra_args=get_method_specific_extra_args,
    shell:
        """
            WANDB_HTTP_TIMEOUT=60 uv run scripts/train_methods/train_{wildcards.method}.py \
                {params.dataset_specific_args} \
                {params.train_dependency_arguments} \
                {params.method_specific_extra_args} \
                seed={wildcards.seed} \
                hard_budget=null \
                output_artifact_aliases=["{EXPERIMENT_ID}"] \
                cost_param={wildcards.cost_param} \
                device={TRAIN_DEVICE}
        """


def get_trained_method_artifact_name(wildcards):
    """Return the name of the trained method artifact for a given method.

    Used during evaluation.
    """
    if wildcards.method in {
        "shim2018",
        "kachuee2019",
        "zannone2019",
        "randomdummy",
        "sequentialdummy",
    }:
        return f"train_{wildcards.method}-{wildcards.dataset}_split_{wildcards.dataset_split}-costparam_{wildcards.cost_param}-seed_{wildcards.seed}:{EXPERIMENT_ID}"
    else:
        msg = f"Unknown method {wildcards.method} when trying to determine training dependency argument."
        raise ValueError(msg)


rule eval_soft_budget_method:
    input:
        f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/trained_method/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}.done",
    output:
        touch(
            f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/eval/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}.done"
        ),
    params:
        classifier_arg=lambda wildcards: f"trained_classifier_artifact_name={CLASSIFIER}-{wildcards.dataset}_split_{wildcards.dataset_split}:{CLASSIFIER_ARTIFACT_ALIASES[wildcards.dataset]}",
        batch_size=lambda wildcards: EVAL_BATCH_SIZE_PER_DATASET[wildcards.dataset],
        trained_method_artifact_name=get_trained_method_artifact_name,
    shell:
        """
        WANDB_HTTP_TIMEOUT=60 uv run scripts/evaluation/eval_soft_afa_method.py \
            trained_method_artifact_name={params.trained_method_artifact_name} \
            cost_param={wildcards.cost_param} \
            {params.classifier_arg} \
            output_artifact_aliases=["{EXPERIMENT_ID}"] \
            seed=null \
            device={EVAL_DEVICE} \
            eval_only_n_samples=null \
            dataset_split={DATASET_SPLIT_MODE} \
            validate_artifacts={VALIDATE_ARTIFACTS} \
            batch_size={params.batch_size}
        """


rule download_soft_budget_eval_result:
    input:
        f"experiments/{EXPERIMENT_ID}/indicators/soft_budget/eval/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}.done",
    output:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}/soft_eval_data.csv",
    params:
        experiment_id=EXPERIMENT_ID,
    shell:
        """
            WANDB_HTTP_TIMEOUT=60 uv run wandb artifact get train_{wildcards.method}-{wildcards.dataset}_split_{wildcards.dataset_split}-costparam_{wildcards.cost_param}-seed_{wildcards.seed}-{CLASSIFIER}-{wildcards.dataset}_split_{wildcards.dataset_split}:{EXPERIMENT_ID} --root experiments/{params.experiment_id}/results/soft_budget/eval/{wildcards.method}/{wildcards.dataset}/split{wildcards.dataset_split}/costparam{wildcards.cost_param}/seed{wildcards.seed}
        """


rule add_dataset_split_col_to_soft_budget_eval_result:
    input:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}/soft_eval_data.csv",
    output:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/{{method}}/{{dataset}}/split{{dataset_split}}/costparam{{cost_param}}/seed{{seed}}/soft_eval_data_with_split.csv",
    conda:
        "../envs/R.yaml"
    shell:
        """
        Rscript scripts/misc/add_dataset_split_col.R {input} {output} {wildcards.dataset_split}
        """


rule combine_soft_budget_eval_results:
    input:
        [
            f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/{method}/{dataset}/split{dataset_split}/costparam{cost_param}/seed{seed}/soft_eval_data_with_split.csv"
            for method in METHODS
            for dataset in DATASETS
            for dataset_split in DATASET_SPLITS
            for cost_param in get_cost_params(method, dataset)
            for seed in SEEDS
        ],
    output:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/combined.csv",
    conda:
        "../envs/R.yaml"
    shell:
        """
        Rscript scripts/misc/combine_soft_budget_results.R {input} {output}
        """

rule soft_budget_plot:
    input:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/eval/combined.csv",
    output:
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot1.pdf",
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot2.pdf",
        f"experiments/{EXPERIMENT_ID}/results/soft_budget/plot3.pdf",
    conda:
        "../envs/R.yaml"
    shell:
        """
        Rscript scripts/plotting/produce_soft_budget_plots.R {input} {output}
        """
