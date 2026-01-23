from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pepper_server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.12",
)
