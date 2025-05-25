# setup.py
from setuptools import setup, find_packages

setup(
    name="act_test",
    version="0.1.0",
    packages=find_packages(),   # will pick up the act_test directory
    install_requires=[
        # e.g. "numpy", …
    ],
)
