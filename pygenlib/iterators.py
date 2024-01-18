"""
    This module implements genomic iterators (mostly based on pysam) for efficient, indexed iteration over genomic
    datasets. Most iterables inherit from LocationIterator and yield named tuples containing data and its
    respective genomic location.

    Some key features:

    * Supports chunked I/O where feasible and not supported by the underlying (pysam) implementation (e.g., FastaIterator)
    * Standard interface for region-type filtering
    * Iterators keep track of their current genomic position and number of iterated/yielded items per chromosome.

    Notes
    -----
    TODOs:

    * improve docs
    * strand specific iteration
    * url streaming
    * remove self.chromosome and get from self.location
    * add is_exhausted flag?

    @LICENSE
"""

from pygenlib import FastqRead

# class PyrangesIterator(PandasIterator):
#     def __init__(self, pyrangesobject, feature, chromosome=None, start=None, end=None, region=None, strand=None,
#                  is_sorted=False, fun_alias=None):
#         super().__init__(df, feature, chromosome, start, end, region, strand, \
#         coord_columns=('Chromosome', 'Start', 'End', 'Strand'), coord_off=(1, 0), per_position=False)


# ---------------------------------------------------------
# SAM/BAM iterators
# ---------------------------------------------------------

# Default BAM flag filter (3844) as used, e.g., in IGV


# ---------------------------------------------------------
# grouped iterators
# ---------------------------------------------------------


# ---------------------------------------------------------
# Additional, non location-bound iterators
# ---------------------------------------------------------


