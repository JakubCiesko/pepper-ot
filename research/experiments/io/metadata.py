from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any
from uuid import uuid4

from .artifacts import save_json


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    log_path: Path
    logger: logging.Logger


def _build_logger(run_id: str, log_path: Path) -> logging.Logger:
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
    output_root: Path, experiment_name: str, config_raw: dict[str, Any]
) -> RunContext:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
        "config": config_raw,
    }
    save_json(run_dir / "run_metadata.json", metadata)
    logger.info("Run started run_id=%s", run_id)
    return RunContext(run_id=run_id, run_dir=run_dir, log_path=log_path, logger=logger)
