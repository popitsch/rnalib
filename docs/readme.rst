rnalib: a python-based genomics library
=========================================

*Rnalib* is a python utilities library for handling genomics data with a focus on transcriptomics.
It implements a transcriptome model and provides efficient iterators for the annotation of its features
(genes, transcripts, exons, etc.). It also provides a number of utility functions for working with
genomics data.

Design Principles
-----------------

rnalib is designed with the following principles in mind:

* Genomic data is represented by an (immutable) location object and arbitrary associated (mutable) annotation data.
* Immutable representations of genomic intervals (`GI`) and features (e.g., genes, transcripts, exons, etc.) can be
  used in indexing and hashing.
* Underlying reference genomes are represented by a `RefDict` object that store chromosome names, their order and
  (possibly) length. RefDicts are used to validate and merge genomic datasets from different sources.
* Annotation data can be incrementally added by direct assignment or by using genomic iterators that yield genomic
  data (location/data tuples).
* Genomic iterators are based on the `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ library and leverage
  respective indexing data structures (e.g., tabix or bai files) for efficient random access. This enables users
  to quickly switch between genomic sub regions (e.g., for focussing on difficult/complex regions) and whole
  transcriptome analyses during development.
* *rnalib* implements a transcriptome model that models parent/child relationships between genomic features
  (e.g., genes, transcripts, exons, etc.) as python objects and references that are dynamically created when loading
  a GFF/GTF file. rnalib understands respective GFF/GTF 'flavours' (e.g., ID attribute names) from different major
  providers such as gencode, ensembl, refseq, etc.

Most importantly, *rnalib* was not designed to replace the great work of others but to integrate with it and fill
gaps. For example, *rnalib* provides interfaces for integrating with `pybedtools <https://daler.github
.io/pybedtools/index.html>`__, `bioframe <https://bioframe.readthedocs.io/>`__ and `HTSeq <https://htseq.readthedocs
.io/>`__.

Installation
------------

Rnalib is hosted on PyPI and can be installed via pip:

.. code:: bash

   $ pip install rnalib

The source code is `available on GitHub <https://github.com/popitsch/rnalib>`_.

You can then import the library in your python code:
.. code:: python

   $ python
   >>> import rnalib as rna

To use *rnalib* in jupyter lab (recommended), you should
* Install jupyter lab
* Install and create a new virtual environment (venv)
* Activate the venv and install the required packages from the requirements.txt file
* Add the venv to jupyter lab
* Start jupyter lab and create/load a notebook

Here is an example of how to use *rnalib &in jupyter lab (adapt paths to your system):

.. code:: bash

    $ cd /Users/niko/.virtualenvs
    $ python3 -m venv rnalib # create venv with name 'rnalib'
    $ source rnalib/bin/activate # activate venv
    (rnalib) $ python3 -m pip install ipykernel ipywidgets # install required ipython packages
    (rnalib) $ python3 -m pip install -r https://raw.githubusercontent.com/popitsch/rnalib/main/requirements.txt # install required packages
    (rnalib) $ python3 -m ipykernel install --user --name=rnalib # add currently activated venv to jupyter
    (rnalib) $ deactivate # deactivate venv
    $ jupyter lab # start jupyter lab

Test data
---------

The *rnalib* test suite and the various tutorial notebooks use various genomic test data files that are not included in
the GitHub repository. These test resources are 'configured' in the `rnalib.testdata <https://github.com/popitsch/rnalib/blob/main/rnalib/testdata.py>`__
module (i.e., their source file/URL, the contained genomic region(s) and a short description of the data).

You can get final test data files in one of the following ways:
* A zipped version (~260M) of the files can be downloaded from the GitHub release page of the rnalib repository (or
  from the respective most recent release with an attached ZIP file).
* The files can also be created by running the `rnalib_create_testdata` python script that is included in the rnalib
  package. This script downloads the source files from public URLs and creates the test files by slicing,
  sorting, compressing and indexing the files. For this, however, you need some external tools (bedtools, bgzip,
  tabix) to be installed.
* The tutorial notebooks provide code snippets for creating the test files via `rna.testdata.create_testdata()`.
  Again, this is only possible if you have the required external tools installed.

Once you have successfully created the testdata folder, you need to tell *rnalib* about its location.
To do so, you can either

* set the `RNALIB_TESTDATA` environment variable
* monkey-patch the global __RNALIB_TESTDATA__ variable to point to your testdata directory as done in the ipython
  notebooks

You can then access test resources via the `rnalib.get_resource(<resource_id>) <https://github.com/search?q=repo%3Apopitsch/rnalib%20get_resource&type=code>`__ method.
The list of valid resource_ids is accessible via the `rnalib.list_resources() <https://github.com/search?q=repo%3Apopitsch/rnalib%20list_resources&type=code>`__ method.

Usage
-----

A detailed description of the API, its design and several usage examples is provided in the
`README.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/README.ipynb>`_ jupyter
notebook. If you don't have jupyter installed, you can also view the notebook on GitHub or run it on Google Colab.

To run rnalib in jupyter lab, it is recommended to create a new  conda or penvironment and install the required packages

you need to install rnalib and its dependencies first
(see fist, commented code cell). You also need to upload the test data files to your Google Drive and mount the drive
or upload the files to the Colab runtime.

We compare rnalib with other genomics libraries in the following notebook:

* `RelatedWork notebook <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/RelatedWork_performance.ipynb>`_

We provide a set of tutorials for demonstrating rnalib in realistic usage scenarios:

* `Tutorial: Read mismatch analysis <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_mismatch_analysis.ipynb>`_
* `Tutorial: Comparison of gene annotation sets <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_compare_annotation_sets.ipynb>`_
* `Tutorial: shRNA analysis <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_shRNA_analysis.ipynb>`_
* `Tutorial: Transcriptome analysis <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_transcriptome_annotation.ipynb>`_

Finally, we showcase how the combination of (the strengths of) multiple genomics libraries leads to an overall benefit in multiple tutorials:

* `Tutorial: CTCF analysis with rnalib and bioframe <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_CTCF_analysis.ipynb>`_
* `Tutorial: Expression analysis with rnalib and genemunge <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_expression_analysis.ipynb>`_

Related work
------------
There exists a broad range of python libraries for working with genomics data that have more or less overlap with
*rnalib*. Here is a selection:

* `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ Python wrapper for the samtools suite. Most rnalib
  iterators are based on pysam.
* `bioframe <https://bioframe.readthedocs.io/>`__ A python library
  enabling flexible and scalable operations on genomic intervals built
  on top of pandas dataframes. rnalib provides interfaces for integrating with bioframe.
* `pybedtools <https://daler.github.io/pybedtools/index.html>`__ Python wrapper for the bedtools suite.
  rnalib provides interfaces for integrating with pybedtools.
* `pyranges <https://pyranges.readthedocs.io/>`__ Python library for efficient and intuitive manipulation of
  genomic intervals. rnalib provides interfaces for integrating with pyranges.
* `biotite <https://www.biotite-python.org/>`__ Python genomics library
* `biopython <https://biopython.org/>`__ Python genomics library
* `HTSeq <https://htseq.readthedocs.io/en/release_0.11.1/>`__ A python library for working with high-throughput sequencing data
* `scikit-bio <https://github.com/biocore/scikit-bio>`__ A general python library for working with biological data
* `cyvcf2 <https://brentp.github.io/cyvcf2/>`__ A fast python VCF parser
* `Pygenomics <https://gitlab.com/gtamazian/pygenomics>`__ A general python genomics library
* `BioNumPy <https://bionumpy.github.io/bionumpy/>`__ A python library for efficient representation and analysis of biological data built on top of NumPy
* `RNAlysis <https://guyteichman.github.io/RNAlysis/build/index.html>`__ Python based RNA-seq analysis software
* `biocantor <https://biocantor.readthedocs.io/en/latest/>`__ is another API targeted at transcriptomics analyses but it
  is unclear whether it is still supported.

We are happy to include other libraries in this list. Please open an issue or a pull request.
