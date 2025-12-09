"""
Sync exclude patterns from ruff.toml to pyrightconfig.json.

This script ensures that both ruff and basedpyright use the same exclude patterns,
maintaining consistency without manual duplication.

Usage:
    # Manual sync (run from project root)
    python scripts/dev/sync_excludes.py

    # Automatic sync via pre-commit hook
    # The sync runs automatically when ruff.toml or pyrightconfig.json are modified

    # Test that excludes are synced
    uv run pre-commit run sync-excludes
"""

import json
import sys
import tomllib
from pathlib import Path


def load_ruff_excludes(ruff_config_path: Path) -> list[str]:
    """Load exclude patterns from ruff.toml."""
    try:
        with ruff_config_path.open("rb") as f:
            ruff_config = tomllib.load(f)

        excludes = ruff_config.get("exclude", [])
        if not isinstance(excludes, list):
            print(
                f"Error: exclude patterns in {ruff_config_path} is not a list"
            )
            sys.exit(1)
        else:
            return excludes

    except FileNotFoundError:
        print(f"Error: {ruff_config_path} not found")
        sys.exit(1)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading {ruff_config_path}: {e}")
        sys.exit(1)


def update_pyright_config(
    pyright_config_path: Path, excludes: list[str]
) -> None:
    """Update pyrightconfig.json with the exclude patterns."""
    try:
        # Load existing config
        with pyright_config_path.open() as f:
            pyright_config = json.load(f)

        # Preserve existing excludes that aren't from ruff (like node_modules, __pycache__)
        existing_excludes = pyright_config.get("exclude", [])

        # Keep common excludes that should always be there
        common_excludes = ["**/node_modules", "**/__pycache__"]
        preserved_excludes = [
            exc for exc in existing_excludes if exc in common_excludes
        ]

        # Combine preserved excludes with ruff excludes
        new_excludes = preserved_excludes + excludes

        # Remove duplicates while preserving order
        seen = set()
        deduped_excludes = []
        for item in new_excludes:
            if item not in seen:
                seen.add(item)
                deduped_excludes.append(item)

        # Update config
        pyright_config["exclude"] = deduped_excludes

        # Write back with nice formatting
        with pyright_config_path.open("w") as f:
            json.dump(pyright_config, f, indent=2)

        print(
            f"✅ Updated {pyright_config_path} with {len(excludes)} exclude patterns from ruff.toml"
        )

    except FileNotFoundError:
        print(f"Error: {pyright_config_path} not found")
        sys.exit(1)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error updating {pyright_config_path}: {e}")
        sys.exit(1)


def main() -> None:
    """Sync exclude patterns from ruff.toml to pyrightconfig.json."""
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    ruff_config_path = project_root / "ruff.toml"
    pyright_config_path = project_root / "pyrightconfig.json"

    print(
        f"🔄 Syncing exclude patterns from {ruff_config_path.name} to {pyright_config_path.name}"
    )

    # Load excludes from ruff config
    excludes = load_ruff_excludes(ruff_config_path)
    print(f"📋 Found {len(excludes)} exclude patterns in ruff.toml")

    # Update pyright config
    update_pyright_config(pyright_config_path, excludes)

    print("🎉 Sync completed successfully!")


if __name__ == "__main__":
    main()
