from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as handle:
    requirements = [line.strip() for line in handle.readlines() if line.strip()]

setup(
    name="pepper-grounded-client",
    version="0.1.0",
    package_dir={"": "app/scripts"},
    packages=find_packages("app/scripts"),
    install_requires=requirements,
    python_requires=">=2.7,<3",
)
