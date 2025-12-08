#!/usr/bin/env python3
"""Simple standalone test for TorchModelBundle without requiring pytest."""

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

# Add the project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.torch_bundle import TorchModelBundle


class SimpleTestModel(nn.Module):
    """A simple model for testing."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def test_basic_save_load():
    """Test basic save and load functionality."""
    print("Testing basic save/load functionality...")

    # Create original model
    original_model = SimpleTestModel(5, 3)

    # Create some test data
    test_input = torch.randn(2, 5)
    original_output = original_model(test_input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "test_model.bundle"

        # Save model using TorchModelBundle
        model_bundle = TorchModelBundle(original_model)
        metadata = {
            "test_type": "basic_save_load",
            "model_info": "SimpleTestModel",
        }
        save_bundle(model_bundle, bundle_path, metadata)

        # Verify bundle was created
        assert bundle_path.exists(), "Bundle file was not created"
        assert (bundle_path / "manifest.json").exists(), (
            "Manifest was not created"
        )
        assert (bundle_path / "data").exists(), (
            "Data directory was not created"
        )
        assert (bundle_path / "data" / "model.pt").exists(), (
            "Model file was not created"
        )

        # Load model back
        loaded_bundle, loaded_metadata = load_bundle(
            bundle_path, device=torch.device("cpu")
        )
        loaded_model = loaded_bundle.model

        # Test that loaded model produces same output
        loaded_output = loaded_model(test_input)

        # Check outputs match
        output_match = torch.allclose(
            original_output, loaded_output, atol=1e-6
        )
        assert output_match, (
            f"Outputs don't match: original={original_output}, loaded={loaded_output}"
        )

        # Check metadata
        assert "metadata" in loaded_metadata, "Metadata section missing"
        custom_metadata = loaded_metadata["metadata"]
        assert custom_metadata["test_type"] == "basic_save_load", (
            "Custom metadata not preserved"
        )
        assert custom_metadata["model_info"] == "SimpleTestModel", (
            "Model info not preserved"
        )

        # Check bundle metadata
        assert loaded_metadata["class_name"] == "TorchModelBundle", (
            "Class name incorrect"
        )
        assert "bundle_version" in loaded_metadata, "Bundle version missing"

    print("✓ Basic save/load test passed!")


def test_parameter_preservation():
    """Test that model parameters are preserved correctly."""
    print("Testing parameter preservation...")

    # Create model with specific weights
    model = SimpleTestModel(3, 2)

    # Set specific weights for testing
    with torch.no_grad():
        model.linear.weight.fill_(0.5)
        model.linear.bias.fill_(0.1)

    # Get original parameters
    original_params = {
        name: param.clone() for name, param in model.named_parameters()
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "param_test.bundle"

        # Save and load model
        model_bundle = TorchModelBundle(model)
        save_bundle(model_bundle, bundle_path, {"test": "parameters"})

        loaded_bundle, _ = load_bundle(bundle_path, device=torch.device("cpu"))
        loaded_model = loaded_bundle.model

        # Check that parameters match
        for name, param in loaded_model.named_parameters():
            original_param = original_params[name]
            params_match = torch.equal(original_param, param)
            assert params_match, (
                f"Parameter {name} doesn't match: original={original_param}, loaded={param}"
            )

    print("✓ Parameter preservation test passed!")


def test_device_handling():
    """Test device handling functionality."""
    print("Testing device handling...")

    model = SimpleTestModel(4, 2)
    model_bundle = TorchModelBundle(model)

    # Test initial device (should be CPU)
    initial_device = model_bundle.device
    assert initial_device == torch.device("cpu"), (
        f"Expected CPU device, got {initial_device}"
    )

    # Test moving to different device (only test CPU since CUDA may not be available)
    cpu_device = torch.device("cpu")
    model_bundle.to(cpu_device)
    final_device = model_bundle.device
    assert final_device == cpu_device, (
        f"Device not updated correctly: {final_device}"
    )

    print("✓ Device handling test passed!")


def test_trained_model():
    """Test saving/loading a model after some training."""
    print("Testing trained model save/load...")

    # Create model and train it briefly
    model = SimpleTestModel(4, 2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Generate some training data
    train_input = torch.randn(8, 4)
    train_target = torch.randn(8, 2)

    # Train for a few steps
    model.train()
    initial_loss = None
    for i in range(5):
        optimizer.zero_grad()
        output = model(train_input)
        loss = criterion(output, train_target)
        if i == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    # Get trained model predictions
    model.eval()
    test_input = torch.randn(2, 4)
    trained_output = model(test_input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir) / "trained_model.bundle"

        # Save trained model
        model_bundle = TorchModelBundle(model)
        metadata = {"trained": True, "steps": 5}
        save_bundle(model_bundle, bundle_path, metadata)

        # Load model back
        loaded_bundle, loaded_metadata = load_bundle(
            bundle_path, device=torch.device("cpu")
        )
        loaded_model = loaded_bundle.model
        loaded_model.eval()

        # Test that loaded model produces same output
        loaded_output = loaded_model(test_input)

        output_match = torch.allclose(trained_output, loaded_output, atol=1e-6)
        assert output_match, (
            "Trained model outputs don't match after save/load"
        )
        assert loaded_metadata["metadata"]["trained"] is True, (
            "Training metadata not preserved"
        )

    print("✓ Trained model test passed!")


def test_class_version():
    """Test that class version is properly set."""
    print("Testing class version...")

    model = SimpleTestModel(2, 1)
    model_bundle = TorchModelBundle(model)

    assert hasattr(model_bundle, "_class_version"), (
        "Class version attribute missing"
    )
    assert model_bundle._class_version == "1.0.0", (
        f"Unexpected class version: {model_bundle._class_version}"
    )

    print("✓ Class version test passed!")


def main():
    """Run all tests."""
    print("Running TorchModelBundle tests...\n")

    try:
        test_basic_save_load()
        test_parameter_preservation()
        test_device_handling()
        test_trained_model()
        test_class_version()

        print("\n🎉 All tests passed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
