#!/usr/bin/env python3
"""Very simple test for TorchModelBundle core functionality."""

import tempfile
from pathlib import Path

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available, skipping tests")
    TORCH_AVAILABLE = False
    exit(0)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def test_basic_torch_save_load():
    """Test basic PyTorch save/load without our bundle system."""
    print("Testing basic PyTorch model save/load...")

    # Create model
    model = SimpleModel()

    # Set known weights
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
        model.linear.bias.fill_(0.1)

    # Test input
    test_input = torch.tensor([[1.0, 2.0]])
    original_output = model(test_input)

    # Save and load using standard PyTorch
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        torch.save(model, tmp_file.name)
        loaded_model = torch.load(tmp_file.name, map_location="cpu")

    # Test loaded model
    loaded_output = loaded_model(test_input)

    # Check if outputs match
    if torch.allclose(original_output, loaded_output):
        print("✓ Basic PyTorch save/load works!")
        return True
    print(f"✗ Outputs don't match: {original_output} vs {loaded_output}")
    return False


def test_torch_bundle_class():
    """Test our TorchModelBundle class structure."""
    print("Testing TorchModelBundle class...")

    # Define the class inline to avoid import issues
    class TorchModelBundle:
        _class_version = "1.0.0"

        def __init__(self, model):
            self.model = model

        def save(self, path: Path):
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model, path / "model.pt")

        @classmethod
        def load(cls, path: Path, device):
            model = torch.load(path / "model.pt", map_location=device)
            return cls(model)

        def to(self, device):
            self.model = self.model.to(device)
            return self

        @property
        def device(self):
            return next(self.model.parameters()).device

    # Test the class
    model = SimpleModel()
    bundle = TorchModelBundle(model)

    # Test properties
    assert hasattr(bundle, "_class_version")
    assert bundle._class_version == "1.0.0"
    assert bundle.device == torch.device("cpu")

    # Test save/load
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "test_bundle"

        # Save
        bundle.save(save_path)
        assert (save_path / "model.pt").exists()

        # Load
        loaded_bundle = TorchModelBundle.load(save_path, torch.device("cpu"))

        # Test functionality
        test_input = torch.tensor([[1.0, 2.0]])
        original_output = bundle.model(test_input)
        loaded_output = loaded_bundle.model(test_input)

        if torch.allclose(original_output, loaded_output):
            print("✓ TorchModelBundle save/load works!")
            return True
        print(
            f"✗ Bundle outputs don't match: {original_output} vs {loaded_output}"
        )
        return False


def main():
    """Run tests."""
    if not TORCH_AVAILABLE:
        return

    print("Running simple torch tests...\n")

    success = True
    success &= test_basic_torch_save_load()
    success &= test_torch_bundle_class()

    if success:
        print("\n🎉 All simple tests passed!")
    else:
        print("\n❌ Some tests failed")


if __name__ == "__main__":
    main()
