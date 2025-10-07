import numpy as np
import pandas as pd


def generate_mock_data(
    n_methods=5,
    n_seeds=5,
    n_cost_params=5,
    n_datasets=5,
    n_samples=200,
    random_state=42,
):
    np.random.seed(random_state)
    methods = [f"Method-{i + 1}" for i in range(n_methods)]
    seeds = [100 + i for i in range(n_seeds)]
    # Define different cost_param ranges for each method
    method_cost_param_ranges = {
        "Method-1": np.linspace(0.01, 0.1, n_cost_params),
        "Method-2": np.linspace(1, 5, n_cost_params),
        "Method-3": np.linspace(0.5, 2, n_cost_params),
        "Method-4": np.linspace(10, 100, n_cost_params),
        "Method-5": np.linspace(0.001, 0.01, n_cost_params),
    }
    datasets = [f"dataset_{chr(65 + i)}" for i in range(n_datasets)]
    # Assign a different max_features (x) for each dataset
    dataset_max_features = {
        dataset: np.random.randint(5, 25) for dataset in datasets
    }

    rows = []
    for method in methods:
        # Use method-specific cost_param range
        cost_params = method_cost_param_ranges.get(
            method, np.linspace(0.1, 1, n_cost_params)
        )
        for seed in seeds:
            for cost_param in cost_params:
                for dataset in datasets:
                    for sample in range(n_samples):
                        # More features chosen should generally mean higher accuracy/F1
                        # For each dataset, average features_chosen should range from 0 to x as cost_param increases
                        max_features = dataset_max_features[dataset]
                        # Normalize cost_param to [0, 1] for this method
                        cost_param_range = np.ptp(cost_params)
                        if cost_param_range == 0:
                            norm_cost = 0
                        else:
                            norm_cost = (
                                cost_param - np.min(cost_params)
                            ) / cost_param_range
                        # Mean features chosen increases with norm_cost, up to max_features
                        mean_features = norm_cost * max_features
                        # Sample features_chosen around this mean, clipped to [0, max_features]
                        features_chosen = int(
                            np.clip(
                                np.random.normal(
                                    loc=mean_features,
                                    scale=max(1, 0.15 * max_features),
                                ),
                                0,
                                max_features,
                            )
                        )
                        # Randomly decide if this method has a builtin classifier
                        has_builtin = np.random.rand() > 0.3

                        # Method-specific sigmoid and noise parameters for F1/accuracy
                        method_sigmoid_params = {
                            "Method-1": {
                                "k_f1": 0.30,
                                "x0_f1": 7,
                                "max_f1": 0.95,
                                "min_f1": 0.55,
                                "k_acc": 0.25,
                                "x0_acc": 6,
                                "max_acc": 0.90,
                                "min_acc": 0.55,
                                "noise_f1": 0.03,
                                "noise_acc": 0.025,
                            },
                            "Method-2": {
                                "k_f1": 0.45,
                                "x0_f1": 12,
                                "max_f1": 0.98,
                                "min_f1": 0.50,
                                "k_acc": 0.40,
                                "x0_acc": 10,
                                "max_acc": 0.97,
                                "min_acc": 0.50,
                                "noise_f1": 0.04,
                                "noise_acc": 0.03,
                            },
                            "Method-3": {
                                "k_f1": 0.20,
                                "x0_f1": 10,
                                "max_f1": 0.90,
                                "min_f1": 0.60,
                                "k_acc": 0.18,
                                "x0_acc": 9,
                                "max_acc": 0.88,
                                "min_acc": 0.60,
                                "noise_f1": 0.05,
                                "noise_acc": 0.04,
                            },
                            "Method-4": {
                                "k_f1": 0.60,
                                "x0_f1": 15,
                                "max_f1": 0.99,
                                "min_f1": 0.45,
                                "k_acc": 0.55,
                                "x0_acc": 13,
                                "max_acc": 0.99,
                                "min_acc": 0.45,
                                "noise_f1": 0.02,
                                "noise_acc": 0.02,
                            },
                            "Method-5": {
                                "k_f1": 0.33,
                                "x0_f1": 8,
                                "max_f1": 0.92,
                                "min_f1": 0.52,
                                "k_acc": 0.28,
                                "x0_acc": 7,
                                "max_acc": 0.91,
                                "min_acc": 0.52,
                                "noise_f1": 0.06,
                                "noise_acc": 0.05,
                            },
                        }
                        params = method_sigmoid_params.get(
                            method, method_sigmoid_params["Method-1"]
                        )

                        def sigmoid(x, k, x0):
                            return 1 / (1 + np.exp(-k * (x - x0)))

                        # F1
                        base_f1 = params["min_f1"] + (
                            params["max_f1"] - params["min_f1"]
                        ) * sigmoid(
                            features_chosen, params["k_f1"], params["x0_f1"]
                        )
                        # Accuracy
                        base_acc = params["min_acc"] + (
                            params["max_acc"] - params["min_acc"]
                        ) * sigmoid(
                            features_chosen, params["k_acc"], params["x0_acc"]
                        )
                        # Add method-specific noise
                        f1_external = np.clip(
                            base_f1 + np.random.normal(0, params["noise_f1"]),
                            0.5,
                            1.0,
                        )
                        acc_external = np.clip(
                            base_acc
                            + np.random.normal(0, params["noise_acc"]),
                            0.5,
                            1.0,
                        )

                        if has_builtin:
                            f1_builtin = np.clip(
                                base_f1
                                + np.random.normal(
                                    0, params["noise_f1"] * 1.2
                                ),
                                0.5,
                                1.0,
                            )
                            acc_builtin = np.clip(
                                base_acc
                                + np.random.normal(
                                    0, params["noise_acc"] * 1.2
                                ),
                                0.5,
                                1.0,
                            )
                        else:
                            f1_builtin = None
                            acc_builtin = None

                        rows.append(
                            {
                                "Method": method,
                                "Training seed": seed,
                                "Cost parameter": float(cost_param),
                                "Dataset": dataset,
                                "Sample": sample,
                                "Features chosen": features_chosen,
                                "F1 (builtin)": f1_builtin,
                                "F1 (external)": f1_external,
                                "Accuracy (builtin)": acc_builtin,
                                "Accuracy (external)": acc_external,
                            }
                        )
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_mock_data()
    output_path = "mock_evaluation_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Mock evaluation data saved to {output_path}")
