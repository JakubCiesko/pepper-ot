from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pepper_client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,  # install pynaoqi manually according to this manual: https://nlp.fi.muni.cz/trac/pepper/wiki/InstallationInstructions
    python_requires=">=2.7,<3",  # pepper supports python 2 only!
)
