from pathlib import Path
import sys


def _repo_root_from_here() -> Path:
    """Locate the repository root from the adapter package path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "server" / "app").exists():
            return parent
    return here.parents[4]


def ensure_server_app_importable() -> None:
    """Add repository and server roots to sys.path for in-process adapters.

    Side Effects:
        Prepends the server directory and repository root to sys.path when they
        are not already present, allowing research code to import app.* modules
        without installing the server package.
    """
    repo_root = _repo_root_from_here()
    server_root = repo_root / "server"
    for p in (str(server_root), str(repo_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
