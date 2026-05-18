from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import hashlib
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4

from .artifacts import save_json


@dataclass
class RunContext:
    """Runtime handles shared by experiment phases.

    Attributes:
        run_id: Stable run identifier used in paths and logs.
        run_dir: Directory where run artifacts are read and written.
        log_path: Path to the run log file.
        logger: Logger configured for console and run.log output.
    """

    run_id: str
    run_dir: Path
    log_path: Path
    logger: logging.Logger


def _build_logger(run_id: str, log_path: Path) -> logging.Logger:
    """Create the per-run logger used by CLI and workflow phases.

    Args:
        run_id: Run identifier included in the logger name.
        log_path: File path receiving persistent logs.

    Returns:
        Logger with stream and file handlers attached.
    """
    logger = logging.getLogger(f"research.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def start_run(
    output_root: Path,
    experiment_name: str,
    run_id: str | None,
    config_raw: dict[str, Any],
    *,
    command: str | None = None,
) -> RunContext:
    """Create a run directory and write immutable run metadata.

    Args:
        output_root: Root directory that contains the runs subdirectory.
        experiment_name: Human-readable experiment name.
        run_id: Optional explicit run ID; generated when None.
        config_raw: Raw config dictionary saved into metadata and hashed.
        command: Optional CLI command name that started the run.

    Returns:
        RunContext for the new run directory.

    Side Effects:
        Creates output_root/runs/run_id, opens run.log, and writes
        run_metadata.json with Python/platform info, git state, config hash,
        optional manifest hash, model metadata, and the raw config.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if run_id is None:
        run_id = f"{experiment_name}_{ts}_{str(uuid4())[:8]}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logger = _build_logger(run_id, log_path)

    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "timestamp_utc": ts,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "command": command,
        "git": _git_state(),
        "config_hash": _stable_hash(config_raw),
        "manifest_hash": _file_hash(_manifest_path_from_config(config_raw)),
        "models": _model_metadata(config_raw),
        "config": config_raw,
    }
    save_json(run_dir / "run_metadata.json", metadata)
    logger.info("Run started run_id=%s", run_id)
    return RunContext(run_id=run_id, run_dir=run_dir, log_path=log_path, logger=logger)


def resume_run(run_dir: Path) -> tuple[RunContext, dict[str, Any]]:
    """Recreate a RunContext from an existing run directory.

    Args:
        run_dir: Directory containing run_metadata.json.

    Returns:
        Tuple of RunContext and parsed metadata.

    Raises:
        FileNotFoundError: If run_metadata.json is missing.

    Side Effects:
        Reopens the run logger and appends resume messages to run.log.
    """
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    from .artifacts import load_json

    metadata = load_json(metadata_path, default={})
    run_id = str(metadata.get("run_id") or run_dir.name)
    log_path = run_dir / "run.log"
    logger = _build_logger(run_id, log_path)
    logger.info("Resumed run run_id=%s run_dir=%s", run_id, run_dir)
    return (
        RunContext(run_id=run_id, run_dir=run_dir, log_path=log_path, logger=logger),
        metadata,
    )


def _stable_hash(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash for a JSON-serializable config payload."""
    import json

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_hash(path_value: object) -> str | None:
    """Return a SHA-256 hash for an existing file path value.

    Args:
        path_value: Path-like value from config, or a falsey value.

    Returns:
        Hex digest when the path exists and is a file, otherwise None.
    """
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path_from_config(config_raw: dict[str, Any]) -> object:
    """Extract the manifest_file value from a raw config dictionary."""
    paths = config_raw.get("paths")
    if not isinstance(paths, dict):
        return None
    return paths.get("manifest_file")


def _git_state() -> dict[str, Any]:
    """Capture best-effort git commit and dirty-state metadata.

    Returns:
        Dictionary containing commit, dirty flag, and short status output.
        Values may be None when git is unavailable or the command times out.
    """

    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        return completed.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status,
    }


def _model_metadata(config_raw: dict[str, Any]) -> dict[str, Any]:
    """Extract model configuration blocks for run provenance metadata."""
    return {
        key: config_raw.get(key)
        for key in ("description_model", "vocabulary_model", "draft_sgg_model")
        if key in config_raw
    }
