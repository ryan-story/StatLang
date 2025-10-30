"""
Setup script for StatLang package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="statlang",
    version="0.1.2",
    author="Ryan Story",
    author_email="ryan@stryve.com",
    description="Python-based statistical scripting language with Jupyter notebook support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryan-story/StatLang",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Interpreters",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pyparsing>=3.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.3.0",
        "requests>=2.25.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "duckdb>=1.0.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "statlang=stat_lang.cli:main",
            "statlang-kernel=stat_lang.kernel:main",
        ],
    },
)
