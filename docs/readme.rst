rnalib: a python-based genomics library
=========================================

*Rnalib* is a python utilities library for handling genomics data with a focus on transcriptomics.
It implements a transcriptome model and provides efficient iterators for the annotation of its features
(genes, transcripts, exons, etc.). It also provides a number of utility functions for working with
genomics data.

Design Principles
-----------------

*Rnalib* was designed with the following principles in mind:

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

   >>> import rnalib as rna
   >>> print(f"imported rnalib {rna.__version__}")

To use *rnalib* in jupyter lab (recommended), you should:

* Install jupyter lab
* Install and create a new virtual environment (venv)
* Activate the venv and install the required packages from the requirements.txt file
* Add the venv to jupyter lab
* Start jupyter lab and create/load a notebook

Here is an example of how to use *rnalib* in jupyter lab (adapt paths to your system):

.. code:: bash

    $ cd /Users/niko/.virtualenvs # change to your venv directory
    $ python3 -m venv rnalib      # create venv with name 'rnalib'
    $ source rnalib/bin/activate  # activate venv
    (rnalib) $ python3 -m pip install ipykernel ipywidgets # install required ipython packages
    (rnalib) $ python3 -m pip install -r https://raw.githubusercontent.com/popitsch/rnalib/main/requirements.txt # install required packages
    (rnalib) $ python3 -m ipykernel install --user --name=rnalib # add currently activated venv to jupyter
    (rnalib) $ deactivate # deactivate venv
    $ jupyter lab # start jupyter lab

Now, you can load an *rnalib* notebook and select 'rnalib' as kernel. All basic requirements of rnalib should be
installed, some notebook-specific requirements (e.g., seaborn) might need to be installed separately (see the respective
notebook).

Test data
---------

The *rnalib* test suite and the tutorial ipython notebooks use various genomic test data files that are not included in
the GitHub repository. These test resources are 'configured' in the `rnalib.testdata <https://github.com/popitsch/rnalib/blob/main/rnalib/testdata.py>`__
module (i.e., their source file/URL, the contained genomic region(s) and a short description of the data).

You can get final test data files in one of the following ways:

* A zipped version (~260M) of the files can be downloaded from the `GitHub release page <https://github.com/popitsch/rnalib/releases>`__ of the rnalib repository (or
  from the respective most recent release with an attached ZIP file).
* The files can also be created by running the `rnalib_create_testdata` python script that is included in the rnalib
  package. This script downloads the source files from public URLs and creates the test files by slicing,
  sorting, compressing and indexing the files. For this, however, you need some external tools (bedtools, bgzip,
  tabix) to be installed.
* The tutorial notebooks provide code snippets for creating the test files via `rna.testdata.create_testdata()`.
  Again, this is only possible if you have the required external tools installed.

Once you have created the testdata folder, you need to tell *rnalib* about its location.
To do so, you can either:

* set the `RNALIB_TESTDATA` environment variable (e.g., in your IDE or in the terminal before starting the python
  interpreter). Example:  "``RNALIB_TESTDATA=notebooks/rnalib_testdata python3``"

* or monkey-patch the global __RNALIB_TESTDATA__ variable to point to your testdata directory as done in the ipython
  notebooks as shown below.

You can then access test resources via the `rnalib.get_resource(<resource_id>) <https://github.com/search?q=repo%3Apopitsch/rnalib%20get_resource&type=code>`__ method.
The list of valid resource_ids is accessible via the `rnalib.list_resources() <https://github.com/search?q=repo%3Apopitsch/rnalib%20list_resources&type=code>`__ method.

.. code:: python

   >>> rna.__RNALIB_TESTDATA__ = "rnalib_testdata/" # point __RNALIB_TESTDATA__ to the testdata directory
   >>> print(rna.get_resource('test_bed')) # get file path of test_bed resource

.. note::

   Larger test data files are not included in the rnalib package to keep the package size small and to avoid
   potential licensing issues. The test data files are not required for using the rnalib package itself.
   To test the rnalib package from commandline, you can run
   "``RNALIB_TESTDATA=<path_to_testdata> pytest``" in the rnalib source directory.



Usage
-----

An introduction to the API, its design and several usage examples is provided in the
`README.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/README.ipynb>`_ jupyter
notebook. A second notebook, `AdvancedUsage.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/AdvancedUsage.ipynb>`_
provides more advanced usage examples and demonstrates some utility functions for working with genomics data.

If you don't have jupyter installed, you can also view the notebooks on `GitHub <https://github.com/popitsch/rnalib/tree/main/notebooks>`_ or run them on Google Colab.
On Google Colab, you need to install rnalib and its dependencies first (see fist, commented code cell).
You also need to upload the required test data files to your Google Drive and mount the drive or upload the files to the Colab runtime.


Tutorials
---------

We also provide a set of tutorials for further demonstrating *rnalib*'s API:

We compare *rnalib* to other genomics libraries with a focus on performance and memory usage in the following notebook:

* `RelatedWork notebook <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/RelatedWork_performance.ipynb>`_

We provide a set of tutorials for demonstrating *rnalib* in realistic usage scenarios:

* `Tutorial: Transcriptome annotation with genemunge, archs4 and mygene.info: annotation with data from public databases <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_transcriptome_annotation.ipynb>`_
* `Tutorial: SLAM-seq analysis: Simplified analysis of a SLAM-seq timecourse dataset <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_SLAM-seq.ipynb>`_
* `Tutorial: Comparison of gene annotation sets: Comparison of different gene annotation sets (human and fly) <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_compare_annotation_sets.ipynb>`_
* `Tutorial: CTCF analysis with rnalib and bioframe: Annotation of genes with CTCF sites <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_CTCF_analysis.ipynb>`_
* `Tutorial: shRNA analysis: a small analysis of shRNA targets <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_shRNA_analysis.ipynb>`_

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



Contributing
------------

Contributions to *rnalib* are highly welcome. Please contact the main author directly or open an issue or a pull request
on the GitHub repository.

Testing
"""""""

We use `pytest <https://docs.pytest.org/en/stable/>`__ and `tox <https://tox.wiki/>`__ for testing *rnalib* against
different python versions as configured in the tox.ini file. We also use `black <https://black.readthedocs.io/>`__
for code formatting.
You can run the tests by running the following command in the rnalib source directory:

.. code:: bash

   $ RNALIB_TESTDATA=<testdata_dir> tox

To run a specific tests with a specific python version, you can use the following command:

.. code:: bash

    $ RNALIB_TESTDATA=<testdata_dir> tox -epy312 -- tests/test_gi.py::test_loc_simple

To skip missing interpreters, you can use the ``--skip-missing-interpreters`` switch.


Screencasts
"""""""""""

We use `terminalizer <https://www.terminalizer.com/>`__ to create animated GIF screencasts that demonstrate *rnalib*'s
API. All required resources can be found in the ``docs/_static/screencasts`` directory. The screencasts are created by
running record_screencasts.sh. The script uses the ``execute_screencast()`` (implemented in `utils.py`) that simulated
a user interaction with the *rnalib* API. Note that the current version requires multi-line commands to start with an
indentation beyond the first line. Note that all python files in the screencasts directory are excluded from
reformatting with black (see tox.ini)


Documentation
"""""""""""""

We use sphinx to generate the documentation. The documentation can be built by running the `build_docs.sh` script in
the `docs/` directory. The documentation of official realases is hosted on
`ReadTheDocs <https://rnalib.readthedocs.io/en/latest/>`_. and currently needs to be built manually.
