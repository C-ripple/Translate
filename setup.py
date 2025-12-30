"""Setup script for cuda2ripple package."""

from setuptools import setup, find_packages

setup(
    name="cuda2ripple",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["flask>=2.0.0"],
    entry_points={
        "console_scripts": [
            "cuda2ripple=cuda2ripple.interfaces.cli.cuda2ripple:main",
            "cuda2ripple-web=cuda2ripple.interfaces.web.server:run_server",
        ],
    },
    python_requires=">=3.9",
)
