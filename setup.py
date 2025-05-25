# setup.py
from setuptools import setup, find_packages

setup(
    name="act_test",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # e.g. "numpy>=1.20", "torch", …
    ],
    author="Vignesh Anand",
    description="ACT-test package",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
