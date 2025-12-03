import argparse
import ast
from pathlib import Path

import pandas as pd


def nullable_int(value: str) -> int | None:
    """Convert string to int, treating 'null' as None."""
    if value.lower() == "null":
        return None
    return int(value)


def nullable_float(value: str) -> float | None:
    """Convert string to float, treating 'null' as None."""
    if value.lower() == "null":
        return None
    return float(value)


def nullable_str(value: str) -> str | None:
    """Convert string to str, treating 'null' as None."""
    if value.lower() == "null":
        return None
    return value


def transform_eval_data(
    input_path: Path,
    output_path: Path,
    afa_method: str,
    dataset: str | None = None,
    train_seed: int | None = None,
    eval_seed: int | None = None,
    hard_budget: int | None = None,
    soft_budget_param: float | None = None,
) -> None:
    """
    Transform raw evaluation data from eval_afa_method into plotting-ready format.

    Takes the raw output from the eval_afa_method function and transforms it into
    the format expected by plotting scripts. This includes computing derived metrics,
    handling classifier columns, and adding metadata.

    Input DataFrame (from eval_afa_method function):
        - "feature_indices" (list[int]): Features available when deciding next selection
        - "prev_selections_performed" (list[int]): Previous selections made
        - "selection_performed" (int): Selection made at this step
        - "builtin_predicted_class" (int|None): Builtin classifier prediction
        - "external_predicted_class" (int|None): External classifier prediction
        - "true_class" (int): True class label

    Output DataFrame (for plotting):
        - "afa_method" (str): Name of the AFA method
        - "classifier" (str): Classifier type ("builtin", "external", or "none")
        - "dataset" (str|None): Dataset name
        - "selections_performed" (int): Number of selections made (len(prev_selections_performed) + 1)
        - "predicted_class" (int|None): Predicted class for current classifier
        - "true_class" (int): True class label
        - "train_seed" (int|None): Training seed
        - "eval_seed" (int|None): Evaluation seed
        - "hard_budget" (int|None): Hard budget used
        - "soft_budget_param" (float|None): Soft budget parameter used

    Args:
        input_path: Path to raw evaluation CSV file
        output_path: Path to save transformed CSV file
        afa_method: Name of the AFA method being evaluated
        dataset: Dataset name (optional)
        train_seed: Seed used during training (optional)
        eval_seed: Seed used during evaluation (optional)
        hard_budget: Hard budget used (optional)
        soft_budget_param: Soft budget parameter used (optional)
    """
    df = pd.read_csv(input_path)

    # Parse list columns and get lengths
    df["selections_performed"] = df["prev_selections_performed"].apply(
        lambda x: len(ast.literal_eval(x)) + 1 if pd.notna(x) else 1
    )

    # Add basic metadata columns
    df["afa_method"] = afa_method
    df["dataset"] = dataset
    df["train_seed"] = train_seed
    df["eval_seed"] = eval_seed
    df["hard_budget"] = hard_budget
    df["soft_budget_param"] = soft_budget_param

    # Identify which classifier columns are present and have non-null values
    classifier_columns = [
        col for col in df.columns if col.endswith("_predicted_class")
    ]

    # Filter to only include classifier columns that have at least some non-null values
    available_classifier_columns = [
        col for col in classifier_columns if bool(df[col].notna().any())
    ]

    if available_classifier_columns:
        # Use pandas melt to pivot longer, but only for available classifiers
        id_vars = [
            col for col in df.columns if not col.endswith("_predicted_class")
        ]
        df = df.melt(
            id_vars=id_vars,
            value_vars=available_classifier_columns,
            var_name="classifier_type",
            value_name="predicted_class",
        )
        # Clean up the classifier column to remove '_predicted_class' suffix
        # This creates values like "builtin" or "external"
        df["classifier"] = df["classifier_type"].str.replace(
            "_predicted_class", ""
        )
        df = df.drop("classifier_type", axis=1)
    else:
        # Fallback if no classifier columns found
        df["predicted_class"] = None
        df["classifier"] = "none"

    # Select final columns in the expected order
    expected_columns = [
        "afa_method",
        "classifier",
        "dataset",
        "selections_performed",
        "predicted_class",
        "true_class",
        "train_seed",
        "eval_seed",
        "hard_budget",
        "soft_budget_param",
    ]

    # Only include columns that exist in the dataframe
    final_columns = [col for col in expected_columns if col in df.columns]
    df = df[final_columns]

    # Save result
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform evaluation data for plotting"
    )
    parser.add_argument("input_path", type=Path, help="Path to input CSV file")
    parser.add_argument(
        "output_path", type=Path, help="Path to output CSV file"
    )
    parser.add_argument("afa_method", type=str, help="Name of the AFA method")
    parser.add_argument(
        "--dataset", type=nullable_str, help="Dataset name (optional)"
    )
    parser.add_argument(
        "--train_seed", type=nullable_int, help="Training seed (optional)"
    )
    parser.add_argument(
        "--eval_seed", type=nullable_int, help="Evaluation seed (optional)"
    )
    parser.add_argument(
        "--hard_budget", type=nullable_int, help="Hard budget (optional)"
    )
    parser.add_argument(
        "--soft_budget_param",
        type=nullable_float,
        help="Soft budget parameter (optional)",
    )

    args = parser.parse_args()

    transform_eval_data(
        input_path=args.input_path,
        output_path=args.output_path,
        afa_method=args.afa_method,
        dataset=args.dataset,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        hard_budget=args.hard_budget,
        soft_budget_param=args.soft_budget_param,
    )


if __name__ == "__main__":
    main()
