"""Console entry point for running the FastAPI service from a source checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_repo_module(relative_path: str, module_name: str) -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / relative_path
    if not module_path.exists():
        raise RuntimeError(
            f"Unable to locate {relative_path}. Run this command from a full source checkout."
        )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_repo_module("deployment/api/main.py", "rlaif_deployment_api")
    if not hasattr(module, "main"):
        raise RuntimeError("deployment/api/main.py does not expose a main() entry point")
    module.main()
