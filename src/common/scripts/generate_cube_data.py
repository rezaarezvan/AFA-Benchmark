"""
Generate cube data and store it in a .pt file
"""

from argparse import ArgumentParser

from common.datasets import CubeDataset


def main(args):
    dataset = CubeDataset(
        n_features=args.n_features,
        data_points=args.data_points,
        sigma=args.sigma,
        seed=args.seed
    )
    dataset.generate_data()
    dataset.save(args.save_path)
    print(f"Generated {args.data_points} data points with {args.n_features} features and saved to {args.save_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate cube data and store it in a .pt file")
    parser.add_argument(
        "--n_features",
        type=int,
        required=True,
        help="How many features there should be",
    )
    parser.add_argument(
        "--data_points",
        type=int,
        required=True,
        help="How many data points there should be",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Noise level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
