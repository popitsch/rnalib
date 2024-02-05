from setuptools import setup

# read the contents of your README file
from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name='rnalib',
    version='0.0.2',
    download_url = 'https://github.com/popitsch/rnalib/archive/refs/tags/1.0.0.tar.gz',
    packages=['rnalib', 'rnalib.static_test_files'],
    package_data={'rnalib.static_test_files': ["*"]},
    include_package_data=True,
    url='https://github.com/popitsch/rnalib',
    license='Apache-2.0',
    author='niko.popitsch@univie.ac.at',
    author_email='niko.popitsch@univie.ac.at',
    description='a python-based transcriptomics library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    scripts=['rnalib/rnalib_create_testdata', 'rnalib/rnalib'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3'
      ],
    install_requires=[
        "bioframe==0.6.1",
        "biotite==0.39.0",
        "dill==0.3.7",
        "h5py==3.8.0",
        "intervaltree==3.1.0",
        "ipython==8.13.2",
        "matplotlib==3.7.4",
        "more_itertools==10.2.0",
        "mygene==3.2.2",
        "numpy==1.26.3",
        "pandas==2.2.0",
        "pybedtools==0.9.1",
        "pyranges==0.0.129",
        "pysam==0.22.0",
        "pytest==7.4.4",
        "sortedcontainers==2.4.0",
        "tqdm==4.65.2",
        "HTSeq==2.0.5"
    ]
)
