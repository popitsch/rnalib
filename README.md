# pygenlib: a python-based genomics library

Pygenlib is a python utilities library for handling genomics data.
It is roughly structured into the following modules:

- utils: general (low-level) utility functions
- genemodel: python classes for modeling genomics (annotation) data
- iterators: efficient iteration over large-scaled genomics datasets

## Installation

```bash
pip install pygenlib
```


## Gene model

pygenlib can instantiate a gene annotation model from a GTF/GFF3 file and currently supports the following annotation
sources: gencode, ensembl, ucsc, flybase, mirgenedb. It supports transcript filtering based on location, attribute 
values and explicit lists of included transcript ids.

pygenlib explicitly models genes, transcripts, exons, introns, 5'UTRs and 3'UTRs. Nucleotide sequences of these features
can be loaded on demand. Efficient location-based queries are implemented via chromosome/feature class specific interval 
trees.


