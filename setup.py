from setuptools import setup

setup(
    name='rnalib',
    version='1.0.0',
    packages=['tests', 'rnalib'],
    url='https://github.com/popitsch/rnalib',
    license='Apache-2.0',
    author='niko.popitsch@univie.ac.at',
    author_email='niko.popitsch@univie.ac.at',
    description='a python-based transcriptomics library',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache-2.0',
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
        "tqdm==4.65.2"
    ]
)
