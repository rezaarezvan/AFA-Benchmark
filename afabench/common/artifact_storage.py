import json
from pathlib import Path
from typing import Any, Protocol, Self

from afabench.common.registry import get_class


class Saveable(Protocol):
    """Protocol for objects that can be saved to disk."""

    def save(self, path: Path) -> None: ...


class Loadeable(Protocol):
    """Protocol for objects that can be loaded from disk."""

    def load(self, path: Path, kwargs: dict[str, Any]) -> Self: ...


def save_artifact(obj: Saveable, path: Path, metadata: dict[str, Any]) -> None:
    """Save an artifact to disk. `path` is required to end with the `.art` extension to make it clear that this is an artifact."""
    assert path.suffix == ".art", "Artifact path must end with .art extension"

    # Save object to path/data/. Object decides what this data folder should contain
    obj.save(path / "data")

    # Save class name to path/class_name.txt. Will make it possible to load the object again with a lookup in the registry.
    with (path / "class_name.txt").open("w") as f:
        f.write(obj.__class__.__name__)

    # Save metadata to path/metadata.json
    with (path / "metadata.json").open("w") as f:
        json.dump(metadata, f)


def load_artifact(
    path: Path, kwargs: dict[str, Any]
) -> tuple[Loadeable, dict[str, Any]]:
    """
    Load an artifact from disk. `path` is required to end with the `.art` extension to make it clear that this is an artifact.

    Returns:
        A tuple of the loaded object and the metadata dictionary.
    """
    assert path.suffix == ".art", "Artifact path must end with .art extension"

    # Load class name from path/class_name.txt
    with (path / "class_name.txt").open("r") as f:
        class_name = f.read().strip()

    # Lookup class in registry
    cls = get_class(class_name)

    # Load object
    obj = cls.load(path / "data", **kwargs)

    # Get metadata
    with (path / "metadata.json").open("r") as f:
        metadata = json.load(f)

    return obj, metadata
