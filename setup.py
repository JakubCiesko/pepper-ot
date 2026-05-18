from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

ROOT = Path(__file__).parent


def read_requirements():
    requirements_path = ROOT / "requirements.txt"
    requirements = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement"):
            continue
        requirements.append(line)
    return requirements


server_packages = ["app"] + [
    f"app.{package}"
    for package in find_namespace_packages(
        where="server/app",
        include=[
            "api*",
            "core*",
            "inference*",
            "orchestration*",
            "providers*",
            "schemas*",
            "worker*",
        ],
    )
]
research_packages = find_namespace_packages(
    where=".",
    include=["research", "research.experiments", "research.experiments.*"],
)


setup(
    name="pepper-ot",
    version="0.1.0",
    description=(
        "Scene-aware dialogue system for the Pepper robot using object tracking "
        "and large language models."
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    package_dir={"app": "server/app"},
    packages=server_packages + research_packages,
    include_package_data=True,
    package_data={
        "app": [
            "static/*",
            "static/css/*",
            "static/js/*",
            "static/img/*",
        ],
        "research": [
            "configs/experiments/*.yaml",
            "configs/vocab/*.yaml",
            "configs/vocab/*.json",
        ],
    },
    install_requires=read_requirements(),
    author="Jakub Ciesko",
    author_email="jakub.ciesko@gmail.com"
)
