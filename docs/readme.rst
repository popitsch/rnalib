.. |PyPI status| image:: https://img.shields.io/pypi/status/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/

.. |PyPI version| image:: https://img.shields.io/pypi/v/rnalib.svg
    :target: https://pypi.python.org/pypi/rnalib/

.. |GitHub license| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://github.com/Naereen/StrapDown.js/blob/master/LICENSE

rnalib: a python-based genomics library
=========================================

*Rnalib*  is a python library for handling transcriptomics data. It implements a transcriptome model and provides
efficient iterators for the annotation of its features (genes, transcripts, exons, etc.).
It also provides a number of utility functions for working with genomics data.

Design
------
Here are our main *rnalib* design considerations:

* In *rnalib*, genomic data is represented by **tuples of genomic locations and associated data**.

* Genomic locations are represented by **immutable genomic intervals** (`GI <_api/rnalib.html#rnalib.GI>`_) that
  can safely be used in indexing and hashing. Genomic intervals are **named tuples** (chromosome, start, end,
  strand)

* Chromosome order is determined by **reference dictionaries** (`RefDict <_api/rnalib.html#rnalib.RefDict>`_ )
  that store chromosome names, their order and (possibly) lengths.
  Reference dictionaries are used to **validate and merge** genomic datasets from different sources.

* Associated annotation data are represented by **arbitrary, mutable objects** (e.g., dicts, *numpy* arrays or
  *pandas* dataframes).

* *Rnalib* implements a `Transcriptome <_api/rnalib.html#rnalib.Transcriptome>`_ class that explicitly models **genomic
  features** (e.g., genes, transcripts, exons, etc.) and their **relationships** (e.g., parent/child relationships)
  using dynamically created python *dataclasses* that inherit from the `GI <_api/rnalib.html#rnalib.GI>`_ class.
  Associated annotations are stored in a separate dictionary that maps features to annotation data.

* A transcriptome can be **instantiated from a GFF/GTF file** and *rnalib* understands various popular GFF/GTF
  '`flavours <_api/rnalib.constants.html#rnalib.constants.GFF_FLAVOURS>`_' (e.g., gencode, ensembl, refseq, flybase,
  etc.).
  Users can then **incrementally add annotation data** to transcriptomes, either by direct assignment or by using
  `LocationIterators <_api/rnalib.html#rnalib.LocationIterator>`_ that yield genomic locations and associated data.

* *Rnalib* implements `a number of LocationIterators <_api/rnalib.html#rnalib.it>`_ for iterating genomic data
  (location/data tuples) via a common interface. Most are based on respective
  `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ classes and leverage associated indexing data structures
  (e.g., .tbi or .bai files) for **efficient random access**.
  This enables users to quickly switch between genomic sub regions (e.g., for focussing on difficult/complex regions)
  and whole transcriptome analyses during development.

* Annotated transcriptomes can be **exported** in various formats (e.g., GFF, BED, pandas dataframes etc.) for further
  processing using other tools/libraries.

* **Most importantly**, *rnalib* was not designed to replace the great work of others but to integrate with it and fill
  gaps. For example, *rnalib* provides interfaces for integrating with `pybedtools <https://daler.github.io/pybedtools/index.html>`__,
  `bioframe <https://bioframe.readthedocs.io/>`__ and `HTSeq <https://htseq.readthedocs.io/>`__.

*Rnalib*'s target audience are bioinformatics analysts and developers and its main design goal is to enable
**fast, readable, reproducible and robust development of novel bioinformatics tools and methods**.

Installation
------------

*Rnalib* is hosted on PyPI and can be installed via pip:

.. code:: bash

   $ pip install rnalib

The source code is `available on GitHub <https://github.com/popitsch/rnalib>`_.

You can import the library as follows:

.. code:: python

   >>> import rnalib as rna
   >>> print(f"imported rnalib {rna.__version__}")

To use *rnalib* in jupyter lab (recommended), you should:

* Install jupyter lab
* Install and create a new virtual environment (*venv*)
* Activate the *venv* and install the required packages from the `requirements.txt <https://raw.githubusercontent.com/popitsch/rnalib/main/requirements.txt>`_ file
* Add the *venv* to jupyter lab
* Start jupyter lab, create/load a notebook and select the *venv* as kernel

Here is an example of how to use *rnalib* in jupyter lab (adapt paths to your system):

.. code:: bash

    $ cd /Users/myusername/.virtualenvs # change to your venv directory
    $ python3 -m venv rnalib      # create venv with name 'rnalib'
    $ source rnalib/bin/activate  # activate venv
    (rnalib) $ python3 -m pip install ipykernel ipywidgets # install required ipython packages
    (rnalib) $ python3 -m pip install -r https://raw.githubusercontent.com/popitsch/rnalib/main/requirements.txt # install required packages
    (rnalib) $ python3 -m ipykernel install --user --name=rnalib # add currently activated venv to jupyter
    (rnalib) $ deactivate # deactivate venv
    $ jupyter lab # start jupyter lab

Now, you can load an *rnalib* notebook and select 'rnalib' as kernel. All basic requirements of *rnalib* should be
installed, however, some *notebook-specific* requirements might still need to be installed separately. Respective
instructions are provided at the beginning of each notebook.


Test data
---------

The *rnalib* test suite and the tutorial ipython notebooks use various genomic **test data files** that are not included
in the GitHub repository due to size restrictions and potential licensing issues.
These test resources are 'configured' in the `rnalib.testdata <https://github.com/popitsch/rnalib/blob/main/rnalib/testdata.py>`__
module (i.e., their source file/URL, the contained genomic region(s) and a short description of the data).

You can get final test data files in **one of the following ways**:

* A zipped version (~260M) of the files can be downloaded from the `GitHub release page <https://github.com/popitsch/rnalib/releases>`__
  of the *rnalib* repository (or from the respective *most recent release* with an attached ZIP file).
* The files can also be created by running :code:`rnalib create_testdata` from the commandline.
  This will download the source files from public sources and creates the test files by slicing,
  sorting, compressing and indexing the files. For this to work, however, you need some external tools (bedtools, bgzip,
  tabix and samtools) to be installed.
* The tutorial notebooks provide code snippets for creating the test files via :code:`rna.testdata.create_testdata()` which
  does the same as `rnalib create_testdata`. Again, this is only possible if you have the required external tools
  installed.


.. note::

   The test data files are not required for using the *rnalib package* itself but only for testing it or
   for running the tutorial notebooks. The additional tools (e.g., tabix) required for creating the test data files are
   also not required for using the *rnalib package* itself.


Usage
-----

An introduction to the API, its design and several usage examples is provided in the
`README.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/README.ipynb>`_ and
in the `AdvancedUsage.ipynb <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/AdvancedUsage.ipynb>`_
notebooks.

If you don't have jupyter installed, you can also view the notebooks on `GitHub <https://github.com/popitsch/rnalib/tree/main/notebooks>`_
or run them on `Google Colab <https://colab.research.google.com/>`_. On Google Colab, you need to install *rnalib* and
its dependencies first. You also need to upload the required test data files to your Google Drive and mount the drive or
upload the files directly to the Colab runtime.


Quick Start
-----------
Here are some examples of how to use *rnalib*:

.. image:: https://github.com/popitsch/rnalib/raw/main/docs/_static/screencasts/introduction.gif
   :alt: Introduction to rnalib
   :align: center

And how to use *rnalib* LocationIterators:

.. image:: https://github.com/popitsch/rnalib/raw/main/docs/_static/screencasts/iterator_demo.gif
   :alt: Introduction to rnalib LocationIterators
   :align: center

Commandline tools
-----------------
*Rnalib* provides a growing number of commandline tools for working with genomics data. These tools are implemented
in the *rnalib* `tools <https://rnalib.readthedocs.io/en/latest/_api/rnalib.tools.html>`_ modulde and can be called from
the commandline via `rnalib <tool>` or from within python scripts. Here is a list of the available tools:

* `rnalib create_testdata <https://rnalib.readthedocs.io/en/latest/_api/rnalib.testdata.html#rnalib.testdata.create_testdata>`_ - Create test data files from public sources
* `rnalib tag_tc <https://rnalib.readthedocs.io/en/latest/_api/rnalib.tools.html#rnalib.tools.tag_tc>`_ - Annotate T-to-C reads
* `rnalib filter_tc <https://rnalib.readthedocs.io/en/latest/_api/rnalib.tools.html#rnalib.tools.filter_tc>`_ - Filter T-to-C reads
* `rnalib prune_tags <https://rnalib.readthedocs.io/en/latest/_api/rnalib.tools.html#rnalib.tools.prune_tags>`_ - Remove TAGs from a BAM file
* `rnalib build_amplicon_resources <https://rnalib.readthedocs.io/en/latest/_api/rnalib.tools.html#rnalib.tools.build_amplicon_resources>`_ - Build amplicon resources


.. note::

   Call :code:`rnalib <tool> --help` for more information on the respective tool.


Tutorials
---------

We also provide a set of tutorials for further demonstrating *rnalib*'s API:

We compare *rnalib* to other genomics libraries with a focus on performance and memory usage in the following notebook:

* `RelatedWork notebook <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/RelatedWork_performance.ipynb>`_

We provide a set of tutorials for demonstrating *rnalib* in realistic usage scenarios:

* `Tutorial: SLAM-seq time-course data analysis <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_SLAM-seq.ipynb>`_
* `Tutorial: Comparison of different gene annotation sets (human and fly) <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_compare_annotation_sets.ipynb>`_
* `Tutorial: Transcriptome annotation with genemunge, archs4 and mygene.info <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_transcriptome_annotation.ipynb>`_
* `Tutorial: CTCF analysis with rnalib and bioframe <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_CTCF_analysis.ipynb>`_
* `Tutorial: A small analysis of shRNA targets <https://colab.research.google.com/github/popitsch/rnalib/blob/main/notebooks/Tutorial_shRNA_analysis.ipynb>`_

Related work
------------
There exists a broad range of python libraries for working with genomics data that have more or less overlap with
*rnalib*. Here is a selection:

* `pysam <https://pysam.readthedocs.io/en/latest/api.html>`__ Python wrapper for the samtools suite. Most *rnalib*
  iterators are based on pysam.
* `bioframe <https://bioframe.readthedocs.io/>`__ A python library
  enabling flexible and scalable operations on genomic intervals built
  on top of pandas dataframes. *Rnalib* provides interfaces for integrating with bioframe.
* `pybedtools <https://daler.github.io/pybedtools/index.html>`__ Python wrapper for the bedtools suite.
  *Rnalib* provides interfaces for integrating with pybedtools.
* `pyranges <https://pyranges.readthedocs.io/>`__ Python library for efficient and intuitive manipulation of
  genomic intervals. *Rnalib* provides interfaces for integrating with pyranges.
* `HTSeq <https://htseq.readthedocs.io/en/release_0.11.1/>`__ A python library for working with high-throughput
  sequencing data. *Rnalib* provides interfaces for integrating with pyranges.
* `biotite <https://www.biotite-python.org/>`__ Python genomics library
* `biopython <https://biopython.org/>`__ Python genomics library
* `Pygenomics <https://gitlab.com/gtamazian/pygenomics>`__ Python genomics library
* `scikit-bio <https://github.com/biocore/scikit-bio>`__ A general python library for working with biological data
* `cyvcf2 <https://brentp.github.io/cyvcf2/>`__ A fast python VCF parser
* `BioNumPy <https://bionumpy.github.io/bionumpy/>`__ Python library for efficient representation and analysis of
  biological data built on top of NumPy
* `RNAlysis <https://guyteichman.github.io/RNAlysis/build/index.html>`__ Python based RNA-seq analysis software
* `biocantor <https://biocantor.readthedocs.io/en/latest/>`__ Another API targeted at transcriptomics analyses but it
  is unclear whether it is still supported.
* `OmicVerse <https://github.com/Starlitnightly/omicverse>`__ A python library for multi omics included bulk, single cell and spatial RNA-seq analysis

We are **happy to include other libraries in this list**. Please open an issue or a pull request.


Getting Help
------------

If you have questions of how to use *rnalib* that is not addressed in the documentation,
please post it on `StackOverflow using the rnalib tag <https://stackoverflow.com/questions/tagged/rnalib>`__.
For bugs and feature requests, please open a `Github Issue <https://github.com/popitsch/rnalib/issues>`__.



Contributing
------------

Contributions to *rnalib* are highly welcome. Please contact the main author directly or open an issue or a pull request
on the GitHub repository.

Testing
"""""""

.. |Pytest| image:: https://img.shields.io/badge/logo-pytest-blue?logo=pytest&labelColor=5c5c5c&label=%20
   :target: https://github.com/pytest-dev/pytest

.. |Tox| image:: https://img.shields.io/badge/logo-tox-blue?logo=tox&labelColor=5c5c5c&label=testing
   :target: https://tox.wiki/

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


Documentation
"""""""""""""

We use sphinx to generate the documentation. The documentation can be built by running the `build_docs.sh` script in
the `docs/` directory. The documentation of official releases is hosted on
`ReadTheDocs <https://rnalib.readthedocs.io/en/latest/>`_. and is built automatically via an
`AutomationRule <https://docs.readthedocs.io/en/stable/automation-rules.html>`_.


Screencasts
"""""""""""

We use `terminalizer <https://www.terminalizer.com/>`__ to create animated GIF screencasts that demonstrate *rnalib*'s
API. All required resources can be found in the ``docs/_static/screencasts`` directory. The screencasts are created by
running ``record_screencasts.sh``. This script uses the *execute_screencast()* method (implemented in `utils.py`) that
simulates user interactions with the *rnalib* API. Note that the current version requires multi-line commands to start
with an indentation beyond the first line, see the existing examples. Note, that all python files in the screencasts
directory are excluded from reformatting with black (see tox.ini)

