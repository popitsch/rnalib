rnalib: a python-based genomics library
=========================================

rnalib is a python utilities library for handling genomics data with a focus on transcriptomics.
It implements a transcriptome model and provides efficient iterators for the annotation of its features
(genes, transcripts, exons, etc.). It also provides a number of utility functions for working with
genomics data.

Design Principles
-----------------

rnalib is designed with the following principles in mind:

* Genomic data is represented by an (immutable) location object and arbitrary associated (mutable) annotation data.
* Immutable representations of genomic intervals (gi) and features (e.g., genes, transcripts, exons, etc.) can be
  used in indexing and hashing.
* Underlying reference genomes are represented by a `ReferenceDict` object that store chromosome names, their order and
  (possibly) length. ReferenceDicts are used to validate and merge genomic datasets from different sources.
* Annotation data can be incrementally added by direct assignment or by using genomic iterators that yield genomic
  data (location/data tuples).
* Genomic iterators are based on the `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ library and leverage
  respective indexing data structures (e.g., tabix or bai files) for efficient random access. This enables users
  to quickly switch between genomic sub regions (e.g., for focussing on difficult/complex regions) and whole
  transcriptome analyses during development.
* rnalib implements a transcriptome model that models parent/child relationships between genomic features
  (e.g., genes, transcripts, exons, etc.) as python objects and references that are dynamically created when loading
  a GFF/GTF file. rnalib understands respective GFF/GTF 'flavours' (e.g., ID attribute names) from different major
  providers such as gencode, ensembl, refseq, etc.

Most importantly, rnalib was not designed to replace the great work of others but to integrate with it and fill
gaps. For example, rnalib provides interfaces for integrating with `pybedtools <https://daler.github
.io/pybedtools/index.html>`__ and `bioframe <https://bioframe.readthedocs.io/>`__.

Installation
------------

.. code:: bash

   $ pip install rnalib


Test data
---------

The rnalib test suite and the tutorial notebooks shown below use various genomic test data files.
These files are not included in the rnalib package but can be produced in one of the following ways:

* A zipped version (~260M) of the files can be downloaded from the GitHub release page of the rnalib repository.
* The files can be created by running the `rnalib_create_testdata` python script that is included in the rnalib
  package. This script downloads the source files from public URLs and creates the test files by slicing,
  sorting, compressing and indexing the files.

Once you have created the testdata folder, you need to tell rnalib about its location.
To do so, you can either

* set the `RNALIB_TESTDATA` environment variable
* monkey-patch the global __RNALIB_TESTDATA__ variable to point to your testdata directory as done in the ipython
  notebooks

You can then acccess test resources via the `get_resource(<resource_id>)` method. If `<resource_id>` starts with
'pybedtools::<id>' then this method will return the filename of the respective test file from the `pybedtools` package.
The list of valid ids is accessible via the `rnalib.testdata.list_resources()` method.

Usage
-----

A detailed description of the API, its design and several usage examples can be found in the
`README.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/README.ipynb>`_ jupyter
notebook. To successfully run the notebook on Google Colab, you need to install rnalib and its dependencies first
(see fist, commented code cell). You also need to upload the test data files to your Google Drive and mount the drive
or upload the files to the Colab runtime.


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
rnalib. Here is a selection:

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
* `biocantor` <https://biocantor.readthedocs.io/en/latest/> is another API targeted at transcriptomics analyses but it
  is unclear whether it is still supported.

We are happy to include other libraries in this list. Please open an issue or a pull request.
