from pathlib import Path
import sys


def _repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "server" / "app").exists():
            return parent
    return here.parents[4]


def ensure_server_app_importable() -> None:
    repo_root = _repo_root_from_here()
    server_root = repo_root / "server"
    for p in (str(server_root), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
