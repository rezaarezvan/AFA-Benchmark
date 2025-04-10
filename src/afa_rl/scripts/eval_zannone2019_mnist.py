"""
Check if pretrain_zannone2019 with mnist dataset produces reasonable results.
"""

import argparse
from functools import partial

import matplotlib.pyplot as plt
import torch
import yaml
from torchrl.modules import MLP

from afa_rl.models import PartialVAE, PointNetPlus, Zannone2019PretrainingModel
from afa_rl.utils import dict_to_namespace, get_2D_identity


def main():
    torch.set_float32_matmul_precision("medium")

    # Use argparse to choose config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load config from yaml file
    with open(args.config, "r") as file:
        config_dict: dict = yaml.safe_load(file)
    config = dict_to_namespace(config_dict)

    args = parser.parse_args()

    # Config must be for MNIST dataset
    assert config.dataset.name == "mnist", "Config must be for MNIST dataset"

    naive_identity_fn = partial(get_2D_identity, image_shape=(28, 28))
    naive_identity_size = 2
    pointnet = PointNetPlus(
        naive_identity_fn=naive_identity_fn,
        identity_network=MLP(
            in_features=naive_identity_size,
            out_features=config.pointnet.identity_size,
            num_cells=config.pointnet.identity_network_num_cells,
        ),
        element_encoder=MLP(
            in_features=config.pointnet.identity_size,
            out_features=config.pointnet.output_size,
            num_cells=config.pointnet.element_encoder_num_cells,
        ),
    )
    encoder = MLP(
        in_features=config.pointnet.output_size,
        out_features=config.encoder.output_size,
        num_cells=config.encoder.num_cells,
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=encoder,
        mu_net=MLP(
            in_features=config.encoder.output_size,
            out_features=config.partial_vae.latent_size,
            num_cells=config.partial_vae.mu_net_num_cells,
        ),
        logvar_net=MLP(
            in_features=config.encoder.output_size,
            out_features=config.partial_vae.latent_size,
            num_cells=config.partial_vae.logvar_net_num_cells,
        ),
        decoder=MLP(
            in_features=config.partial_vae.latent_size,
            out_features=config.dataset.n_features,
            num_cells=config.partial_vae.decoder_num_cells,
        ),
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        classifier=MLP(
            in_features=config.encoder.output_size,
            out_features=config.classifier.output_size,
            num_cells=config.classifier.num_cells,
        ),
        lr=config.lr,
    )

    # Load checkpoint
    checkpoint = torch.load(f"models/afa_rl/{config.checkpoint}", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])

    # Sample 10 images from latent space
    latent_samples = torch.randn(10, config.partial_vae.latent_size)
    # Decode samples
    with torch.no_grad():
        decoded_samples = model.partial_vae.decoder(latent_samples)
        # Pass the decoded samples through the partial VAE, as fully observed
        encoding, mu, logvar, z, estimated_features = model.partial_vae(
            decoded_samples, torch.full_like(decoded_samples, 1, dtype=torch.bool)
        )
        # Pass encoding through classifier
        logits = model.classifier(encoding)
    # Get predicted labels
    predicted_labels = logits.argmax(dim=1)

    # Reshape to images
    decoded_images = decoded_samples.view(-1, 1, 28, 28)
    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        ax = axs[i // 5, i % 5]
        ax.imshow(decoded_images[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")
        # Add predicted label
        ax.set_title(f"Pred: {predicted_labels[i].item()}")
    plt.show()


if __name__ == "__main__":
    main()
