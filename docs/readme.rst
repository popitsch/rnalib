pygenlib: a python-based genomics library
=========================================

Pygenlib is a python utilities library for handling genomics data with a focus on transcriptomics.
It implements a transcriptome model and provides efficient iterators for the annotation of its features
(genes, transcripts, exons, etc.). It also provides a number of utility functions for working with
genomics data.

Design Principles
-----------------

Pygenlib is designed with the following principles in mind:

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
* Pygenlib implements a transcriptome model that models parent/child relationships between genomic features
  (e.g., genes, transcripts, exons, etc.) as python objects and references that are dynamically created when loading
  a GFF/GTF file. Pygenlib understands respective GFF/GTF 'flavours' (e.g., ID attribute names) from different major
  providers such as gencode, ensembl, refseq, etc.

Most importantly, pygenlib was not designed to replace the great work of others but to integrate with it and fill
gaps. For example, pygenlib provides interfaces for integrating with `pybedtools <https://daler.github
.io/pybedtools/index.html>`__ and `bioframe <https://bioframe.readthedocs.io/>`__.

Installation
------------

.. code:: bash

   $ pip install pygenlib

Usage
-----

A detailed description of the API, its design and several usage examples can be found in the
`README.ipynb <https://github.com/popitsch/pygenlib/blob/main/notebooks/README.ipynb>`_ jupyter
notebook.

.. raw:: html

    <a target="_blank" href="https://colab.research.google.com/github/popitsch/pygenlib/blob/main/notebooks/README.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>



Tests
-----

Pygenlib tests use various test data files that can be created by running the testdata python script.
This class contains a `test_resources` dict that describes the various test resources and their origin.
Briefly, this script does the following for each configured resource:

        * Download source file from a public URL or copy from the static_files directory
        * Ensure that the files are sorted by genomic coordinates and are compressed and indexed with bgzip and tabix.
        * Slice genomic subregions from the files if configured
        * Copy the result files and corresponding indices to the testdata directory

Once you have created the testdata folder, you can get the filenames of test resources via the
`get_resource(<resource_id>)` method. If `<resource_id>` starts with 'pybedtools::<id>' then this method
will return the filename of the respective  test file from the `pybedtools` package.
To list the ids of all available test resources, use the `list_resources()` method.

Additionally, some of the more complex usage examples in the `README.ipynb` notebook require some
larger genomics files that are not included in the testdata folder. Follow the respective documentation
in the notebook to download these files.

Related work
------------
There exists a broad range of python libraries for working with genomics data that have more or less overlap with
pygenlib. Here is a selection:

* `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ Python wrapper for the samtools suite. Most pygenlib
  iterators are based on pysam.
* `bioframe <https://bioframe.readthedocs.io/>`__ A python library
  enabling flexible and scalable operations on genomic intervals built
  on top of pandas dataframes. Pygenlib provides interfaces for integrating with bioframe.
* `pybedtools <https://daler.github.io/pybedtools/index.html>`__ Python wrapper for the bedtools suite.
  Pygenlib provides interfaces for integrating with pybedtools.
* `pyranges <https://pyranges.readthedocs.io/>`__ Python library for efficient and intuitive manipulation of
  genomic intervals. Pygenlib provides interfaces for integrating with pyranges.
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
