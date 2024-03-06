#!/usr/bin/env python
import ast, os, re
from setuptools import setup
from pathlib import Path

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # noqa

# read the contents of the README file
long_description = (Path(__file__).parent / "README.md").read_text()

init_file = Path(__file__).parent / "rnalib/__init__.py"

# get metadata from __init__.py
meta = {}
with open(init_file, "r") as f:
    rx = re.compile("(__version__) = (.*)")
    for line in f:
        m = rx.match(line)
        if m:
            meta[m.group(1)] = ast.literal_eval(m.group(2))
print(f"Installing rnalib {meta['__version__']}")

# github URL
github_url = "https://github.com/popitsch/rnalib"
setup(
    name="rnalib",
    version=meta["__version__"],  # parsed from __init__.py
    python_requires=">=3.10",
    license="Apache-2.0",
    author="Niko Popitsch",
    author_email="niko.popitsch@univie.ac.at",
    description="A python-based transcriptomics library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=github_url,
    project_urls={
        "Changelog": f"{github_url}/blob/main/CHANGELOG.md",
        "Issues": f"{github_url}/issues",
        "Source Code": github_url,
    },
    packages=["rnalib", "rnalib.static_test_files"],
    package_data={"rnalib.static_test_files": ["*"]},
    include_package_data=True,
    scripts=["rnalib/rnalib"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "setuptools",
        "bioframe",
        "biotite",
        "dill",
        "h5py",
        "intervaltree",
        "ipython",
        "matplotlib",
        "more_itertools",
        "mygene",
        "numpy",
        "pandas",
        "pybedtools",
        "pyranges",
        "pysam",
        "pytest",
        "sortedcontainers",
        "tqdm",
        "arrow",
        "pyarrow",
        "s3fs",
        "HTSeq",
        "pyBigWig",
        "termcolor",
    ],
)
