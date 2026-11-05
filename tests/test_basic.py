"""Basic sanity tests for FlowCast structure and module imports."""
import importlib


def test_package_structure():
    """Verify that all core FlowCast subpackages can be imported."""
    modules = [
        "src",
        "src.data",
        "src.features",
        "src.models",
        "src.api",
    ]
    for mod in modules:
        imported = importlib.import_module(mod)
        assert imported is not None, f"Failed to import {mod}"


def test_project_directories_exist():
    """Verify standard project directories are present."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/api",
        "dashboard",
        "models",
        "tests",
        "docs",
    ]
    for d in expected_dirs:
        dir_path = root / d
        assert dir_path.exists(), f"Expected directory does not exist: {d}"

