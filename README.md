# pygenlib: a python-based genomics library

Pygenlib is a python utilities library for handling genomics data.
It is roughly structured into the following modules:

- [iterators](#genomic-iterators): efficient iteration over large-scaled genomics datasets. Iterators keep track of the genomic region of the
  yielded data enabling their efficient integration with other genomics data
- [genemodel](#gene-model): python classes for modeling genomics (annotation) data. This includes a 'transcriptome' implementation 
  that models gene/transcript annotations and many useful querying/annotation methods.
- [utils](#utility-methods): general (low-level) utility functions for working with genomics datasets.

## Installation

```bash
pip install pygenlib
```

## Usage

Please refer to the
[README.ipynb](notebooks%2FREADME.ipynb)
jupyter notebook for detailed usage examples.

## Genomic Iterators 



## Gene model

_pygenlib_ can instantiate a gene annotation model from a GTF/GFF3 file and currently supports the various GFF flavours
from the following annotation sources: gencode, ensembl, ucsc, flybase, mirgenedb. It supports transcript filtering based on location, attribute 
values and explicit lists of included transcript ids.

![transcriptome datamodel](notebooks/pygenlib_transcriptome.png "Transcriptome datamodel")

_pygenlib_ explicitly models genes, transcripts, exons, introns, 5'UTRs and 3'UTRs. 
Nucleotide sequences of these features can be loaded from a configured reference genome and will be 
efficiently stored by _pygenlib_. Sorted features can be iterated/filtered or efficiently queried
using gene-based inteval trees.

## Utility methods



## Related work

- [Pygenomics](https://gitlab.com/gtamazian/pygenomics) A general python genomics library
- [scikit-bio](https://github.com/biocore/scikit-bio) A general python library for working with biological data 