"""Reusable hashing and immutable run metadata helpers."""

from datetime import datetime, timezone
import hashlib
import glob
import json
import math
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


CLUSTERING_EVALUATION_SCHEMA_VERSION = 3


def file_identity(path: Any) -> Dict[str, Any]:
    """Return a path-independent content identity for a file or directory."""
    target = Path(path).resolve()
    if target.is_file():
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        stat = target.stat()
        return {
            "path": str(target),
            "sha256": digest.hexdigest(),
            "size_bytes": stat.st_size,
        }
    if target.is_dir():
        entries = [file_identity(child) for child in sorted(target.rglob("*")) if child.is_file()]
        digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        return {"path": str(target), "sha256": digest, "files": len(entries)}
    raise FileNotFoundError(f"Cannot identify missing input: {target}")


def write_json(path: Any, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(value), stream, indent=2, sort_keys=True,
            default=_json_default, allow_nan=False,
        )
        stream.write("\n")


def load_protocol(path: Any = "docs/RESULTS_PROTOCOL_REPORT.md") -> str:
    """Load the project protocol document; fail explicitly if it is unavailable."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Protocol document not found: {target}")
    return target.read_text(encoding="utf-8")


class EvaluationRun:
    """Create a non-overwriting run directory and persist its lifecycle metadata."""

    def __init__(
        self,
        stage: str,
        split: str,
        config: Dict[str, Any],
        output_root: str = "runs/clustering/evaluations",
        run_name: Optional[str] = None,
        input_paths: Optional[Iterable[Any]] = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        suffix = run_name or f"{stage}_{split}"
        self.run_dir = create_artifact_directory(output_root, suffix)
        self.run_json = self.run_dir / "run.json"
        self.metrics_path = self.run_dir / "metrics.json"
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        source_paths = _clustering_source_paths()
        source_identities = {
            str(path): file_identity(path)
            for path in source_paths
        }
        source_scope = [
            "src/clustering", "scripts/train_clustering.py", "scripts/predict_clustering.py",
            "src/utils/evaluation.py", "configs", "tests",
        ]
        git_status = _git_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *source_scope]
        )
        git_diff = _git_output(["git", "diff", "--binary", "HEAD", "--", *source_scope])
        code_provenance_hash = hashlib.sha256(
            json.dumps(source_identities, sort_keys=True).encode()
            + git_status.encode()
            + git_diff.encode()
        ).hexdigest()
        packages = {}
        for package in (
            "numpy", "pandas", "scikit-learn", "scipy", "torch", "torchvision",
            "opencv-python", "shapely", "matplotlib", "joblib", "umap-learn",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        self.metadata: Dict[str, Any] = {
            "created_at": stamp,
            "stage": stage,
            "split": split,
            "status": "in_progress",
            "git_commit": commit,
            "git_worktree_status": git_status,
            "git_relevant_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
            "source_files": source_identities,
            "code_provenance_hash": code_provenance_hash,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hardware": {
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "cuda_available": _cuda_available(),
                "gpu": _gpu_name(),
            },
            "packages": packages,
            "config": config,
            "inputs": {
                str(Path(path).resolve()): file_identity(path)
                for path in (input_paths or [])
            },
            "outputs": {},
        }
        write_json(self.run_json, self.metadata)

    def complete(self, metrics: Dict[str, Any], output_paths: Optional[Iterable[Any]] = None) -> None:
        self.metadata["status"] = "completed"
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.metrics_path, metrics)
        paths = list(output_paths or []) + [self.metrics_path]
        self.metadata["outputs"] = {
            str(Path(path).resolve()): file_identity(path)
            for path in paths
            if Path(path).exists()
        }
        write_json(self.run_json, self.metadata)

    def fail(self, error: BaseException) -> None:
        self.metadata["status"] = "failed"
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(self.run_json, self.metadata)


def create_artifact_directory(root: Any, name: str) -> Path:
    """Create a short named directory; use numeric suffixes without overwriting."""
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("Artifact name must be a single directory name")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / (name if index == 1 else f"{name}_{index}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def _clustering_source_paths() -> list[Path]:
    candidates = [
        *map(Path, glob.glob("src/clustering/**/*.py", recursive=True)),
        *map(Path, glob.glob("src/clustering/workflows/*.sh")),
        Path("scripts/train_clustering.py"),
        Path("scripts/predict_clustering.py"),
        Path("src/utils/evaluation.py"),
        *map(Path, glob.glob("tests/**/*.py", recursive=True)),
        *map(Path, glob.glob("configs/clustering_config*.yaml")),
    ]
    return sorted(path.resolve() for path in candidates if path.is_file())


def _cuda_available() -> Optional[bool]:
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.cuda.is_available())


def _gpu_name() -> Optional[str]:
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
