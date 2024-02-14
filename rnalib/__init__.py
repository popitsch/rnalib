"""
    Rnalib is a python utilities library for handling genomics data with a focus on transcriptomics.

    It implements a transcriptome model that can be instantiated from various popular
    GFF 'flavours' as published in encode, ensembl, ucsc, chess, mirgendb and flybase databases.
    Several filtering options are available to filter transcripts by gene type, biotype, etc.

    The transcriptome model is implemented as a hierarchical tree of frozen dataclasses
    (derived from the 'Feature' class) and supports incremental and flexible annotation of
    transcriptome features. It also supports the extraction of genomic sequences for
    transcriptome features.

    The transcriptome model supports genomic range queries via `query()` which are
    implemented by a combination of interval and linear search queries. A transcriptome
    object maintains one intervaltree per chromosome built from gene annotations.
    Overlap/envelop queries will first be applied to the respective intervaltree and the
    (typically small result sets) will then be filtered, e.g., for requested sub-feature types.


"""
import logging
import math
from abc import abstractmethod, ABC
from collections import Counter, abc
from dataclasses import dataclass, make_dataclass  # type: ignore # import dataclass to avoid PyCharm warnings
from itertools import chain
from os import PathLike
from typing import List, Callable, NamedTuple, Any, Tuple, Iterable

import HTSeq
import bioframe
import dill
import pyranges
from intervaltree import IntervalTree
from more_itertools import pairwise, triplewise, windowed, peekable
from sortedcontainers import SortedList, SortedSet

from ._version import __version__
from .constants import *
from .interfaces import Archs4Dataset
from .testdata import get_resource, list_resources
from .tools import *
from .utils import *

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# location of the test data directory. Use the 'RNALIB_TESTDATA' environment variable or monkey patching to set to your
# favourite location, e.g., rnalib.__RNALIB_TESTDATA__ = "your_path'
__RNALIB_TESTDATA__ = os.environ.get('RNALIB_TESTDATA')


# ------------------------------------------------------------------------
# Genomic Interval (gi) model
# ------------------------------------------------------------------------
class GI(NamedTuple):
    """
        Genomic intervals (GI) in rnalib are inclusive, continuous and 1-based. This model differs from other
        genomics libraries but was chosen to make interpretation of GIs straightforward: start and end coordinates
        represent the first/last included nucleotide as also seen in a genome browser (such as IGV).

        GIs are implemented as (readonly) namedtuples and can safely be used as keys in a dict.
        They are instantiated by the gi() factory method that either parses from a string representation or by passing
        explicitly chrom/start/stop coordinates. Coordinate components can be 'unrestricted' by setting them to
        None, e.g., gi('chr1', 1, None) refers to the whole chromosome 1, gi('chr1', 100000) refers to the section on
        chr1 from (and including) position 100k on, gi(start=100, end=200) refers to positions 100-200 (inclusive) on
        any chromosome.

        Intervals can be stranded. The strand can be '+' or '-' or None if unstranded. Note that '.' will be converted
        to None. Individual genomic positions (points) are represented by intervals with same start and stop
        coordinate. Empty intervals can be represented by start>end coordinates (e.g., gi('chr1', 1,0).is_empty() ->
        True) but note that this feature is still experimental and may not be supported in all methods.

        GIs can be compared for equality, overlap, envelopment, etc. and can be sorted by chromosome and coordinates.

        GIs are grouped by chromosomes/contigs of a given reference genome and their order is defined by a
        `RefDict` (that extends regular python dicts). RefDicts are typically instantiated automatically
        from indexing data structures (e.g., .tbi or .bai files) and keep track of the order of chromosomes/contigs
        and their lengths. They are used throughout rnalib to assert the compatibility of different genomic datasets
        and to sort GIs by chromosome and coordinate, e.g., via
        `sorted(gis, key=lambda x: (refdict.index(x.chromosome), x))`. Note that the index of chromosome 'None'
        is always 0, i.e., it is the first chromosome in the sorted list.

        GIs can be iterated over to yield individual positions (points) as GIs, e.g.,
        tuple(gi('chr2', 1, 3)) -> (gi('chr2', 1, 1), gi('chr2', 2, 2), gi('chr2', 3, 3)).

        GIs should be instantiated by the `gi()` factory method that allows to configure each coordinate component
        individually or can parse from a string representation, e.g., `gi('chr1:1-10 (+)')`. The factory method
        conducts type and value checking and conversions (e.g., '.' to None for strand).

        Attributes
        ----------
        chromosome: str
            Chromosome (default: None)
        start: int
            First included position, 1-based (default: 0).
        end: int
            Last included position, 1-based (default: MAX_INT)
        strand: str
            Strand, either '+' or '-' or None if unstranded. Note that '.' will be converted to None. Default: None

        Examples
        --------
        >>> gi('chr3')
        >>> gi('chr2', 1, 10)
        >>> gi('1', 1, 10, strand='+')
        >>> gi('chr4:1-10 (-)')
        >>> assert gi('chrX:2-1').is_empty()


        Notes
        -----
        Note that there is also a frozen (immutable) dataclass version of this class, `GI_dataclass` that
        shares most functionality with GIs and is used as superclass for genomic features (genes, transcripts,
        etc.) as instantiated by the `Transcriptome` class. The reason for this separation is that dataclass
        instantiation is relatively slow and has performance impacts when instantiating large sets of GIs.

    """
    chromosome: str = None
    start: int = 0  # unbounded, ~-inf
    end: int = rna.MAX_INT  # unbounded, ~+inf
    strand: str = None

    def __len__(self):
        if self.is_empty():  # empty intervals have zero length
            return 0
        if self.start == 0 or self.end == MAX_INT:
            return MAX_INT  # length of (partially) unbounded intervals is always max_int.
        return self.end - self.start + 1

    @classmethod
    def from_str(cls, loc_string):
        """ Parse from <chr>:<start>-<end> (<strand>). Strand is optional"""
        pattern = re.compile(r"(\w+):(\d+)-(\d+)(?:[\s]*\(([+-])\))?$")  # noqa
        match = pattern.findall(loc_string.strip().replace(',', ''))  # convenience
        if len(match) == 0:
            return None
        chromosome, start, end, strand = match[0]
        strand = None if strand == '' else strand
        return gi(chromosome, int(start), int(end), strand)

    @staticmethod
    def sort(intervals, refdict):
        """ Returns a chromosome + coordinate sorted iterable over the passed intervals. Chromosome order is defined by
        the passed reference dict."""
        return sorted(intervals, key=lambda x: (refdict.index(x.chromosome), x))

    def __repr__(self):
        if self.is_empty():
            return f"{self.chromosome}:<empty>"
        return f"{self.chromosome}:{self.start}-{self.end}{'' if self.strand is None else f' ({self.strand})'}"

    def get_stranded(self, strand):
        """Get a new object with same coordinates; the strand will be set according to the passed variable."""
        return gi(self.chromosome, self.start, self.end, strand)

    def to_file_str(self):
        """ returns a sluggified string representation "<chrom>_<start>_<end>_<strand>"        """
        return f"{self.chromosome}_{self.start}_{self.end}_{'u' if self.strand is None else self.strand}"

    def is_unbounded(self):
        return [self.chromosome, self.start, self.end, self.strand] == [None, 0, MAX_INT, None]

    def is_empty(self):
        return self.start > self.end

    def is_stranded(self):
        return self.strand is not None

    def cs_match(self, other, strand_specific=False):
        """ True if this location is on the same chrom/strand as the passed one.
            will not compare chromosomes if they are unrestricted in one of the intervals.
            Empty intervals always return False hee
        """
        if strand_specific and self.strand != other.strand:
            return False
        if self.chromosome and other.chromosome and (self.chromosome != other.chromosome):
            return False
        return True

    def __cmp__(self, other, cmp_str, refdict=None):
        if not self.cs_match(other, strand_specific=False):
            if refdict is not None:
                return getattr(refdict.index(self.chromosome), cmp_str)(refdict.index(other.chromosome))
            return None
        if self.start != other.start:
            return getattr(self.start, cmp_str)(other.start)
        return getattr(self.end, cmp_str)(other.end)

    def __lt__(self, other):
        """
            Test whether this interval is smaller than the other.
            Defined only on same chromosome but allows unrestricted coordinates.
            If chroms do not match, None is returned.
        """
        return self.__cmp__(other, '__lt__')

    def __le__(self, other):
        """
            Test whether this interval is smaller or equal than the other.
            Defined only on same chromosome but allows unrestricted coordinates.
            If chroms do not match, None is returned.
        """
        return self.__cmp__(other, '__le__')

    def __gt__(self, other):
        """
            Test whether this interval is greater than the other.
            Defined only on same chromosome but allows unrestricted coordinates.
            If chroms do not match, None is returned.
        """
        return self.__cmp__(other, '__gt__')

    def __ge__(self, other):
        """
            Test whether this interval is greater or equal than the other.
            Defined only on same chromosome but allows unrestricted coordinates.
            If chroms do not match, None is returned.
        """
        return self.__cmp__(other, '__ge__')

    def left_match(self, other, strand_specific=False):
        if not self.cs_match(other, strand_specific):
            return False
        return self.start == other.start

    def right_match(self, other, strand_specific=False):
        if not self.cs_match(other, strand_specific):
            return False
        return self.end == other.end

    def left_pos(self):
        return gi(self.chromosome, self.start, self.start, strand=self.strand)

    def right_pos(self):
        return gi(self.chromosome, self.end, self.end, strand=self.strand)

    def envelops(self, other, strand_specific=False) -> bool:
        """ Tests whether this interval envelops the passed one.
        """
        if self.is_unbounded():  # envelops all
            return True
        if self.is_empty() or other.is_empty():  # zero overlap with empty intervals
            return False
        if not self.cs_match(other, strand_specific):
            return False
        return self.start <= other.start and self.end >= other.end

    def overlaps(self, other, strand_specific=False) -> bool:
        """ Tests whether this interval overlaps the passed one.
            Supports unrestricted start/end coordinates and optional strand check
        """
        if self.is_unbounded() or other.is_unbounded():  # overlaps all
            return True
        if self.is_empty() or other.is_empty():  # zero overlap with empty intervals
            return False
        if not self.cs_match(other, strand_specific):
            return False
        return self.start <= other.end and other.start <= self.end

    def overlap(self, other, strand_specific=False) -> float:
        """Calculates the overlap with the passed one"""
        if self.is_unbounded() or other.is_unbounded():  # overlaps all
            return MAX_INT
        if self.is_empty() or other.is_empty():  # zero overlap with empty intervals
            return 0
        if not self.cs_match(other, strand_specific):
            return 0
        return min(self.end, other.end) - max(self.start, other.start) + 1.

    def split_coordinates(self) -> (str, int, int):
        return self.chromosome, self.start, self.end

    @staticmethod
    def merge(loc):
        """ Merges a list of intervals.
            If intervals are not on the same chromosome or if strand is not matching, None is returned
            The resulting interval will inherit the chromosome and strand of the first passed one.
        """
        if loc is None:
            return None
        if len(loc) == 1:
            return loc[0]
        merged = None
        for x in loc:
            if x is None:
                continue
            if merged is None:
                merged = [x.chromosome, x.start, x.end, x.strand]
            else:
                if (x.chromosome != merged[0]) or (x.strand != merged[3]):
                    return None
                merged[1] = min(merged[1], x.start)
                merged[2] = max(merged[2], x.end)
        return gi(*merged)

    def is_adjacent(self, other, strand_specific=False):
        """ true if intervals are directly next to each other (not overlapping!) """
        if not self.cs_match(other, strand_specific=strand_specific):
            return False
        a, b = (self.end + 1, other.start) if self.end < other.end else (other.end + 1, self.start)
        return a == b

    def get_downstream(self, width=100):
        """Returns an upstream genomic interval """
        if self.is_stranded():
            s, e = (self.end + 1, self.end + width) if self.strand == '+' else (self.start - width, self.start - 1)
            return gi(self.chromosome, s, e, self.strand)
        else:
            return None

    def get_upstream(self, width=100):
        """Returns an upstream genomic interval """
        if self.is_stranded():
            s, e = (self.end + 1, self.end + width) if self.strand == '-' else (self.start - width, self.start - 1)
            return gi(self.chromosome, s, e, self.strand)
        else:
            return None

    def get_extended(self, width=100):
        """Returns an genomic interval that is extended up- and downstream by width nt"""
        return gi(self.chromosome, self.start - width, self.end + width, self.strand)

    def split_by_maxwidth(self, maxwidth):
        """ Splits this into n intervals of maximum width """
        k, m = divmod(self.end - self.start + 1, maxwidth)
        ret = [
            gi(self.chromosome, self.start + i * maxwidth, self.start + (i + 1) * maxwidth - 1, strand=self.strand)
            for i in range(k)]
        if m > 0:
            ret += [gi(self.chromosome, self.start + k * maxwidth, self.end, strand=self.strand)]
        return ret

    def copy(self):
        """ Returns a copy of this gi """
        return gi(self.chromosome, self.start, self.end, self.strand)

    def distance(self, other, strand_specific=False):
        """
            Distance to other interval.
            - None if chromosomes do not match
            - 0 if intervals overlap
            - negative if other < self
        """
        if self.cs_match(other, strand_specific=strand_specific):
            if self.overlaps(other):
                return 0
            return other.start - self.end if other > self else other.end - self.start
        return None

    def to_pybedtools(self):
        """
            Returns a corresponding pybedtools interval object.
            Note that this will fail on open or empty intervals as those are not supported by pybedtools.

            Examples
            --------
            >>> gi('chr1',1,10).to_pybedtools()
            >>> gi('chr1',1,10, strand='-').to_pybedtools()
            >>> gi('chr1').to_pybedtools() # uses maxint as end coordinate

            Warning
            -------
            Note that len(gi('chr1',12,10).to_pybedtools()) reports wrong length 4294967295 for this empty interval!
        """
        # pybedtools cannot deal with unbounded intervals, so we replace with [0; maxint]
        start = 1 if self.start == 0 else self.start
        # pybedtools: `start` is *always* the 0-based start coordinate
        return pybedtools.Interval(self.chromosome, start - 1, self.end,  # noqa
                                   strand='.' if self.strand is None else self.strand)

    def to_htseq(self):
        """
            Returns a corresponding HTSeq interval object.
            Note that this will fail on open or empty intervals as those are not supported by HTSeq.

            Examples
            --------
            >>> gi('chr1',1,10).to_htseq()
            >>> gi('chr1',1,10, strand='-').to_htseq()
            >>> gi('chr1').to_htseq()
        """
        return HTSeq.GenomicInterval(self.chromosome,
                                     self.start - 1,
                                     self.end,
                                     strand='.' if self.strand is None else self.strand)

    def __iter__(self):
        for pos in range(self.start, self.end + 1):
            yield gi(self.chromosome, pos, pos, self.strand)


def gi(chromosome: str = None, start: int = 0, end: int = MAX_INT, strand: str = None):
    """ Factory function for genomic intervals (GI).

        Examples
        --------
        >>> gi('chr1', 1, 10, strand='-') # explicit coordinates
        >>> gi('chr1:1-10 (-)') # parsed from string
        >>> gi('chr1') # whole chromosome
        >>> gi() # unbounded interval
    """
    if chromosome is not None and ':' in chromosome:
        return GI.from_str(chromosome)
    start = 0 if start is None else start
    end = rna.MAX_INT if end is None else end
    if strand == '.':
        strand = None
    elif strand is not None:
        assert strand in [None, '+', '-']
    if start > end:  # empty interval, set start/end to 0/-1
        start, end = 0, -1
    return GI(chromosome, start, end, strand)


@dataclass(frozen=True, init=True, slots=True)
class GI_dataclass:  # noqa
    """
        Dataclass for genomic intervals (GIs) in rnalib.
        Copies the functionality of the namedtuple GI, but is slower to instantiate due to the post_init assertions.
        Needed for `Feature` hierarchies and other dataclasses that need to be frozen.
    """
    chromosome: str = None
    start: int = 0  # unbounded, ~-inf
    end: int = MAX_INT  # unbounded, ~+inf
    strand: str = None

    def __post_init__(self):
        """ Some sanity checks and default values.
            This is slow, adds ~500ns per object creation, so we disable it for performance critical code.
        """
        object.__setattr__(self, 'start', 0 if self.start is None else self.start)
        object.__setattr__(self, 'end', MAX_INT if self.end is None else self.end)
        object.__setattr__(self, 'strand', self.strand if self.strand != '.' else None)
        assert isinstance(self.start, numbers.Number)
        assert isinstance(self.end, numbers.Number)
        if self.start > self.end:  # empty interval, set start/end to 0/-1
            object.__setattr__(self, 'start', 0)
            object.__setattr__(self, 'end', -1)
        assert self.strand in [None, '+', '-']

    # copy the methods from the GI named tuple implementation
    __len__ = GI.__len__
    __repr__ = GI.__repr__
    get_stranded = GI.get_stranded
    to_file_str = GI.to_file_str
    is_unbounded = GI.is_unbounded
    is_empty = GI.is_empty
    is_stranded = GI.is_stranded
    cs_match = GI.cs_match
    __cmp__ = GI.__cmp__
    __lt__ = GI.__lt__
    __le__ = GI.__le__
    __gt__ = GI.__gt__
    __ge__ = GI.__ge__
    left_match = GI.left_match
    right_match = GI.right_match
    left_pos = GI.left_pos
    right_pos = GI.right_pos
    envelops = GI.envelops
    overlaps = GI.overlaps
    overlap = GI.overlap
    split_coordinates = GI.split_coordinates
    merge = GI.merge
    is_adjacent = GI.is_adjacent
    get_downstream = GI.get_downstream
    get_upstream = GI.get_upstream
    get_extended = GI.get_extended
    split_by_maxwidth = GI.split_by_maxwidth
    copy = GI.copy
    distance = GI.distance
    to_pybedtools = GI.to_pybedtools
    to_htseq = GI.to_htseq
    __iter__ = GI.__iter__


def _transcript_to_bed(idx, tx, item):
    """ Default conversion of transcripts to BED12 format.
        CDS are used as thickStart/thickEnd if available, otherwise the transcript start/end is used.
        Note that the BED12 format is 0-based, half-open, i.e., the end coordinate is not included.

        Parameters
        ----------
        idx : int
            Index of the transcript in the transcriptome
        tx : Transcript
            Transcript object
        item : Any
            Item associated with the transcript (not used here)
        Returns
        -------
         tuple
            name, score, thickStart, thickEnd, rgb, blockCount, blockSizes, blockStarts as required by BED12
    """
    if len(tx.CDS) > 0:
        thickStart, thickEnd = (tx.CDS[-1].start - 1, tx.CDS[0].end) if tx.strand == "-" else (
            tx.CDS[0].start - 1, tx.CDS[-1].end)
    else:
        thickStart, thickEnd = (tx.start - 1, tx.end)
    blockCount = len(tx.exon)
    blockSizes, blockStarts = zip(*[(str(len(ex)), str(ex.start - tx.start))
                                    for ex in (reversed(tx.exon) if tx.strand == "-" else tx.exon)])
    color = "255,153,153" if tx.strand == "-" else "153,255,153"  # this works in standalone IGV but not JS !
    return tx.feature_id, '.', thickStart, thickEnd, color, blockCount, ','.join(blockSizes), ','.join(
        blockStarts)


class FixedKeyTypeDefaultdict(defaultdict):
    """
        A defaultdict that allows only keys of a certain type.

        Examples
        --------
        >>> d = rna.FixedKeyTypeDefaultdict(defaultdict, allowed_key_type=int)
        >>> d[1] = 1 # this works, key is an int
        >>> import pytest
        >>> with pytest.raises(TypeError) as e_info:
        >>>     d['x'] = 2 # this raises a TypeError as key is a string
        >>> d.allowed_key_type = str # you can change the allowed key type
        >>> d['x'] = 2 # now it works
        >>> assert dill.loads(dill.dumps(d)).allowed_key_type == str # pickling/dill works
    """

    def __init__(self, /, *args, **kwargs):
        self.allowed_key_type = kwargs.pop('allowed_key_type', None)  # remove it from kwargs
        super().__init__(*args, **kwargs)
        # check if all keys are of the allowed type
        if self.allowed_key_type is not None and not all(
                [isinstance(key, self.allowed_key_type) for key in self.keys()]):
            raise TypeError(f"Only {self.allowed_key_type} objects can be added to this dict.")

    def __setitem__(self, key, value):
        # check key type
        if (self.allowed_key_type is not None) and (not isinstance(key, self.allowed_key_type)):
            raise TypeError(
                f"Only {self.allowed_key_type} objects can be added to this dict, you passed a {type(key)} object")
        super().__setitem__(key, value)

    def __getstate__(self):
        return self.allowed_key_type  # needed for pickling

    def __setstate__(self, state):
        self.allowed_key_type = state  # needed for pickling

    def __missing__(self, key):
        # check key type
        if (self.allowed_key_type is not None) and (not isinstance(key, self.allowed_key_type)):
            raise TypeError(
                f"Only {self.allowed_key_type} objects can be added to this dict, you passed a {type(key)} object")
        return super().__missing__(key)

    def __reduce__(self):
        """ Pickle support. See https://docs.python.org/3/library/pickle.html#object.__reduce__
            add allowed_key_type to the state
        """
        c, arg, state, it, i2 = super().__reduce__()
        state = self.allowed_key_type
        return c, arg, state, it, i2


class Transcriptome:
    """
    Represents a transcriptome as modelled by a GTF/GFF file.

    *   Model contains genes, transcripts and arbitrary sub-features (e.g., exons, intron, 3'/5'-UTRs, CDS) as
        defined in the GFF file. Frozen dataclasses (derived from the 'Feature' class) are created for all parsed
        feature types automatically and users may configure which GTF/GFF attributes will be added to those (and are
        thus accessible via dot notation, e.g., gene.gene_type).
    *   This `transcriptome` implementation exploits the hierarchical relationship between genes and their
        sub-features to optimize storage and computational requirements, see the `Feature` documentation for
        examples. To enable this, however, parent features *must* envelop (i.e., completely contain) child feature
        locations and this requirement is asserted when building the `transcriptome`.
    *   A `transcriptome` maintains an `anno` dict mapping (frozen) features to dicts of arbitrary annotation
        values. This supports incremental and flexible annotation of `transcriptome` features. Values can directly
        be accessed via dot notation <feature>.<attribute> and can be stored/loaded to/from a (pickled) file.
    *   `Feature` sequences can be added via `load_sequences()` which will extract the sequence of the top-level
        feature ('gene') from the configured reference genome. Sequences can then be accessed via get_sequence().
        For sub-features (e.g., transcripts, exons, etc.) the respective sequence will be sliced from the gene
        sequence. If mode='rna' is passed, the sequence is returned in 5'-3' orientation, i.e., they are
        reverse-complemented for minus-strand transcripts. The returned sequence will, however, still use the DNA
        alphabet (ACTG) to enable direct alignment/comparison with genomic sequences.
        if mode='spliced', the spliced 5'-3' sequence will be returned.
        if mode='translated', the spliced 5'-3' CDS sequence will be returned.
    *   Genomic range queries via `query()` are supported by a combination of interval and linear search queries.
        A transcriptome object maintains one intervaltree per chromosome built from gene annotations.
        Overlap/envelop queries will first be applied to the respective intervaltree and the (typically small
        result sets) will then be filtered, e.g., for requested sub-feature types.
    *   When building a transcriptome model from a GFF/GTF file, contained transcripts can be filtered using a
        :func:`TranscriptFilter <rnalib.TranscriptFilter>`.
    * | The current implementation does not implement the full GFF3 format as specified in
      | https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
      | but currently supports various popular gff 'flavours' as published in
      | encode, ensembl, ucsc, chess, mirgendb and flybase databases (see
      | :func:`GFF_FLAVOURS <rnalib.constants.GFF_FLAVOURS>`). As such this implementation will likely be extended
      | in the future.

    @see the `README.ipynb <https://github.com/popitsch/rnalib/blob/main/notebooks/README.ipynb>`_ jupyter
    notebook for various querying and iteration examples

    Parameters
    ----------
    annotation_gff : str
        Path to a GFF/GTF file containing the transcriptome annotations.
    annotation_flavour : str
        The annotation flavour of the GFF/GTF file. Currently supported: encode, ensembl, ucsc, chess, mirgendb and
        flybase.
    genome_fa : str
        Path to a FASTA file containing the reference genome. If provided, sequences will be loaded and can be
        accessed via get_sequence().
    gene_name_alias_file : str
        Path to a gene name alias file. If provided, gene symbols will be normalized using the alias file.
    annotation_fun_alias : Callable
        Name of a function that will be used to alias gene names.
        The function must be defined in the global namespace and must accept a gene name and return the alias.
        If provided, gene symbols will be normalized using the alias function.
    copied_fields : tuple
        List of GFF/GTF fields that will be copied to the respective feature annotations.
    calc_introns : bool
        If true, intron features will be added to transcripts that do not have them.
    load_sequence_data : bool
        If true, sequences will be loaded from the genome_fa file.
    disable_progressbar : bool
        If true, progressbars will be disabled.
    genome_offsets : dict
        A dict mapping chromosome names to offsets. If provided, the offset will be added to all coordinates.
    name : str
        An (optional) human-readable name of the transcriptome object. default: 'Transcriptome'
    feature_filter : TranscriptFilter
        A TranscriptFilter object that will be used to filter transcripts.


    Attributes
    ----------
    genes : List[Gene]
        List of genes in the transcriptome.
    transcripts : List[Transcript]
        List of transcripts in the transcriptome.
    anno : Dict[Feature, Dict[str, Any]]
        Dictionary mapping features (e.g., genes, transcripts) to their annotations.

    """

    def __init__(self,
                 annotation_gff: str,
                 annotation_flavour: str,
                 genome_fa: str = None,
                 gene_name_alias_file: str = None,
                 annotation_fun_alias: Callable = None,
                 copied_fields: tuple = (),
                 load_sequence_data: bool = False,
                 calc_introns: bool = True,
                 disable_progressbar: bool = False,
                 genome_offsets: dict = None,
                 name="Transcriptome",
                 feature_filter=None):
        self.annotation_gff = annotation_gff
        self.file_format = guess_file_format(self.annotation_gff)
        self.annotation_flavour = annotation_flavour.lower()
        assert (self.annotation_flavour, self.file_format) in GFF_FLAVOURS, \
            ("Unsupported annotations flavour. Supported:\n" + ', '.join([f"{k}/{v}" for k, v in GFF_FLAVOURS]))
        self.genome_fa = genome_fa
        self.gene_name_alias_file = gene_name_alias_file
        self.annotation_fun_alias = annotation_fun_alias
        # get GFF aliasing function
        if self.annotation_fun_alias is not None:
            assert self.annotation_fun_alias in globals(), (f"fun_alias func {self.annotation_fun_alias} undefined in "
                                                            f"globals()")
            self.annotation_fun_alias = globals()[self.annotation_fun_alias]
            logging.info(f"Using aliasing function for annotation_gff: {self.annotation_fun_alias}")
        self.copied_fields = {'source', 'gene_type'} if copied_fields is None else \
            set(copied_fields) | {'source', 'gene_type'}  # ensure source and gene_type are copied
        self.load_sequence_data = load_sequence_data
        self.calc_introns = calc_introns
        self.disable_progressbar = disable_progressbar
        self.genome_offsets = {} if genome_offsets is None else genome_offsets
        self.name = name
        self.feature_filter = TranscriptFilter() if feature_filter is None else TranscriptFilter(
            feature_filter) if isinstance(feature_filter, dict) else feature_filter
        self.log = Counter()
        self.merged_refdict = None
        self.gene = {}  # gid: gene
        self.transcript = {}  # tid: gene
        self.cached = False  # if true then transcriptome was loaded from a pickled file
        self.has_seq = False  # if true, then gene objects are annotated with the respective genomic (dna) sequences
        self.anno = FixedKeyTypeDefaultdict(defaultdict)  # a dict of dicts that holds annotation data for each feature
        self.chr2itree = {}  # a dict mapping chromosome ids to annotation interval trees.
        self.genes = []  # list of genes
        self.transcripts = []  # list of transcripts
        self.duplicate_gene_names = {}  # dict mapping gene names to lists of genes with the same name
        self._ft2anno_class: dict[str, dict] = None  # mapping feature types to annotation fields parsed from GFF
        self._ft2child_ftype = None  # mapping feature types to child feature types
        self._ft2subclass = None  # mapping feature types to the respective subclass
        self.build()  # build the transcriptome object

    def build(self):
        # reset log
        self.log = Counter()
        # read gene aliases (optional)
        aliases, current_symbols = (None, None) if self.gene_name_alias_file is None else read_alias_file(
            self.gene_name_alias_file, disable_progressbar=self.disable_progressbar)
        fmt = GFF_FLAVOURS[self.annotation_flavour, self.file_format]

        # estimate valid chromosomes
        rd = [] if self.genome_fa is None else [RefDict.load(open_file_obj(self.genome_fa))]
        rd += [RefDict.load(open_file_obj(self.annotation_gff), fun_alias=self.annotation_fun_alias)]
        self.merged_refdict = RefDict.merge_and_validate(*rd,
                                                         check_order=False,
                                                         included_chrom=self.feature_filter.get_chromosomes()
                                                         )
        assert len(self.merged_refdict) > 0, "No shared chromosomes!"
        # iterate gff
        genes = {}
        transcripts = {}
        line_number = 0
        for chrom in tqdm(self.merged_refdict,
                          f"Building transcriptome ({len(self.merged_refdict)} chromosomes)\n",
                          disable=self.disable_progressbar):
            # PASS 1: build gene objects
            filtered_gene_ids = set()
            with GFF3Iterator(self.annotation_gff, chrom, fun_alias=self.annotation_fun_alias) as it:
                try:
                    for line_number, (loc, info) in enumerate(it):
                        self.log['parsed_gff_lines'] += 1
                        feature_type = fmt['ftype_to_SO'].get(info['feature_type'], None)
                        if feature_type == 'gene':  # build gene object
                            # filter...
                            filtered, filter_message = self.feature_filter.filter(loc, info)
                            if filtered:
                                self.log[f"filtered_{info['feature_type']}_{filter_message}"] += 1
                                filtered_gene_ids.add(info.get(fmt['gid'], 'None'))
                                continue
                            gid = info.get(fmt['gid'], 'None')
                            if gid is None:
                                warnings.warn(f"Skipping {self.annotation_flavour} {self.file_format} line "
                                              f"{line_number + 1} ({info['feature_type']}), info:\n\t{info} as no gene_id found.")
                                continue
                            genes[gid] = _Feature(self, 'gene', gid, loc,
                                                  parent=None, children={'transcript': []})
                            for cf in self.copied_fields:
                                genes[gid].anno[cf] = info.get(cf, None)
                            genes[gid].anno['gene_name'] = norm_gn(info.get(fmt['gene_name'], gid), current_symbols,
                                                                   aliases)  # normalized gene symbol/name
                            genes[gid].anno['gff_feature_type'] = info['feature_type']
                except Exception as exc:
                    logging.error(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line "
                                  f"{line_number + 1},  info:\n\t{info}")
                    raise exc
            # PASS 2: build transcript objects and add missing gene annotations
            missing_genes = {}
            with GFF3Iterator(self.annotation_gff, chrom, fun_alias=self.annotation_fun_alias) as it:
                try:
                    for line_number, (loc, info) in enumerate(it):
                        feature_type = fmt['ftype_to_SO'].get(info['feature_type'], None)
                        if feature_type == 'transcript':  # build tx object
                            # filter...
                            filtered, filter_message = self.feature_filter.filter(loc, info)
                            if filtered:
                                self.log[f"filtered_{info['feature_type']}_{filter_message}"] += 1
                                continue
                            # get transcript and gene id
                            tid = info.get(fmt['tid'], None)
                            if tid is None:
                                warnings.warn(f"Skipping {self.annotation_flavour} {self.file_format} line "
                                              f" {line_number + 1} ({info['feature_type']}), info:\n\t{info} as  no"
                                              f" {fmt['tid']} field found.")
                                continue
                            gid = f'gene_{tid}' if fmt['tx_gid'] is None else info.get(fmt['tx_gid'], None)
                            if gid is None:
                                warnings.warn(f"Skipping {self.annotation_flavour} {self.file_format} line "
                                              f"{line_number + 1} {info['feature_type']}), info:\n\t{info} as no "
                                              f" {fmt['tx_gid']} field found.")
                                continue
                            if gid in filtered_gene_ids:
                                self.log[f"filtered_{info['feature_type']}_parent_gene_filtered"] += 1
                                continue
                            # create transcript object
                            transcripts[tid] = _Feature(self, 'transcript', tid, loc,
                                                        parent=genes.get(gid, None),
                                                        children={k: [] for k in set(fmt['ftype_to_SO'].values()) - {
                                                            'gene', 'transcript'}})
                            for cf in self.copied_fields:
                                transcripts[tid].anno[cf] = info.get(cf, None)
                            transcripts[tid].anno['gff_feature_type'] = info['feature_type']
                            # add missing gene annotation (e.g., ucsc, flybase, chess)
                            if gid not in genes:
                                if gid in missing_genes:
                                    newloc = GI.merge([missing_genes[gid].loc, loc])
                                    if newloc is None:
                                        # special case, e.g., in Chess annotation/tx CHS.40038.9 is annotated on the
                                        # opposite strand. We skip this tx and keep the gene annotation.
                                        # In chess 3.0.1 there are 3 such entries, all pseudo genes.
                                        warnings.warn(f"Gene {gid} has tx with incompatible coordinates! "
                                                      f"{missing_genes[gid].loc} vs {loc}, skipping tx {tid}")
                                        self.log[(f"filtered_"
                                                  f"{info['feature_type']}_incompatible_coordinates_filtered")] += 1
                                        del transcripts[tid]
                                    else:
                                        missing_genes[gid].loc = newloc
                                        missing_genes[gid].children['transcript'].append(transcripts[tid])  # add child
                                else:
                                    missing_genes[gid] = _Feature(self, 'gene', gid, loc, parent=None,
                                                                  children={'transcript': [transcripts[tid]]})
                                    for cf in self.copied_fields:
                                        missing_genes[gid].anno[cf] = info.get(cf, None)
                                    missing_genes[gid].anno['gene_id'] = gid
                                    missing_genes[gid].anno['gene_name'] = norm_gn(info.get(fmt['gene_name'], gid),
                                                                                   current_symbols, aliases)
                                    # normalized gene symbol/name
                            else:  # add as child
                                genes[gid].children['transcript'].append(transcripts[tid])
                    for gid, mg in missing_genes.items():
                        genes[gid] = missing_genes[gid]
                except Exception as exc:
                    logging.error(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line"
                                  f" {line_number + 1}, info:\n\t{info}")
                    raise exc
            # PASS 3: add features
            allowed_feature_types = set(fmt['ftype_to_SO'].values()) - {'gene',
                                                                        'transcript'}
            # {'CDS', 'exon', 'five_prime_UTR', 'intron', 'three_prime_UTR'}
            with GFF3Iterator(self.annotation_gff, chrom, fun_alias=self.annotation_fun_alias) as it:
                try:
                    for line_number, (loc, info) in enumerate(it):
                        feature_type = fmt['ftype_to_SO'].get(info['feature_type'], None)
                        if feature_type in allowed_feature_types:  # build gene object
                            # filter...
                            filtered, filter_message = self.feature_filter.filter(loc, info)
                            if filtered:
                                self.log[f"filtered_{info['feature_type']}_{filter_message}"] += 1
                            # get transcript and gene id
                            tid = info.get(fmt['feat_tid'], None)
                            if (tid is None) or (tid not in transcripts):  # no parent tx found
                                continue
                            feature_id = f"{tid}_{feature_type}_{len(transcripts[tid].children[feature_type])}"
                            feature = _Feature(self, feature_type, feature_id, loc, parent=transcripts[tid],
                                               children={})
                            for cf in self.copied_fields:
                                feature.anno[cf] = info.get(cf, None)
                            feature.anno['gff_feature_type'] = info['feature_type']
                            transcripts[tid].children[feature_type].append(feature)
                except Exception as exc:
                    logging.error(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line"
                                  f" {line_number + 1}, info:\n\t{info}")
                    raise exc
        # drop genes w/o transcripts (e.g., after filtering)
        for k in [k for k, v in genes.items() if len(v.children['transcript']) == 0]:
            self.log['dropped_empty_genes'] += 1
            genes.pop(k, None)
        # add intron features if not parsed
        if self.calc_introns:
            for tid, tx in transcripts.items():
                if ('exon' not in tx.children) or (len(tx.children['exon']) <= 1):
                    continue
                strand = tx.loc.strand
                for rnk, (ex0, ex1) in enumerate(pairwise(tx.children['exon'])):
                    loc = gi(tx.loc.chromosome, ex0.loc.end + 1, ex1.loc.start - 1, strand)
                    if loc.is_empty():
                        continue  # TODO: what happens to rnk?!
                    feature_type = 'intron'
                    feature_id = f"{tid}_{feature_type}_{len(tx.children[feature_type])}"
                    intron = _Feature(self, feature_type, feature_id, loc, parent=tx, children={})
                    # copy fields from previous exon
                    intron.anno = ex0.anno.copy()
                    # add to transcript only if this is non-empty
                    ex0.parent.children[feature_type].append(intron)

        # step1: create custom dataclasses
        self._ft2anno_class = {}  # contains annotation fields parsed from GFF
        self._ft2child_ftype = {}  # feature 2 child feature types
        fts = set()
        for g in genes.values():
            a, t, s = g.get_anno_rec()
            self._ft2anno_class.update(a)
            self._ft2child_ftype.update(t)
            fts.update(s)
        self._ft2subclass = {
            ft: Feature.create_sub_class(ft, self._ft2anno_class.get(ft, {}), self._ft2child_ftype.get(ft, [])) for ft
            in fts
        }
        # step2: freeze and add to auxiliary data structures
        self.genes = [g.freeze(self._ft2subclass) for g in genes.values()]
        all_features = list()
        for g in self.genes:
            all_features.append(g)
            for f in g.features():
                all_features.append(f)
        all_features.sort(key=lambda x: (self.merged_refdict.index(x.chromosome), x))
        self.anno = FixedKeyTypeDefaultdict(defaultdict, {f: {} for f in all_features})
        # self.anno = FixedKeyTypeDefaultdict({f: {} for f in all_features})
        # assert that parents intervals always envelop their children
        for f in all_features:
            if f.parent is not None:
                assert f.parent.envelops(
                    f), f"parents intervals must envelop their child intervals: {f.parent}.envelops({f})==False"
        # build some auxiliary dicts
        self.transcripts = [f for f in all_features if f.feature_type == 'transcript']
        self.gene = {f.feature_id: f for f in all_features if f.feature_type == 'gene'}
        self.gene.update({f.gene_name: f for f in all_features if f.feature_type == 'gene'})
        self.transcript = {f.feature_id: f for f in self.transcripts}
        # Create a dict with genes that share the same gene_name (buf different ids), such as PAR genes
        self.duplicate_gene_names = Counter(g.gene_name for g in self.genes)
        self.duplicate_gene_names = {x: list() for x, count in self.duplicate_gene_names.items() if count > 1}
        for g in [g for g in self.genes if g.gene_name in self.duplicate_gene_names.keys()]:
            self.duplicate_gene_names[g.gene_name].append(g)
        # load sequences
        if self.load_sequence_data:
            self.load_sequences()
        # build interval trees
        for g in tqdm(self.genes, desc=f"Build interval trees", total=len(self.genes),
                      disable=self.disable_progressbar):
            if g.chromosome not in self.chr2itree:
                self.chr2itree[g.chromosome] = IntervalTree()
            # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
            self.chr2itree[g.chromosome].addi(g.start, g.end + 1, g)

    def load_sequences(self):
        """Loads feature sequences from a genome FASTA file.
            Requires a 'genome_fa' config entry.
        """
        # show or hide progressbar
        with pysam.Fastafile(self.genome_fa) as fasta:
            for g in tqdm(self.genes, desc='Load sequences', total=len(self.genes), disable=self.disable_progressbar):
                start = g.start - self.genome_offsets.get(g.chromosome, 1)
                end = g.end - self.genome_offsets.get(g.chromosome, 1) + 1
                prefix = ""
                if start < 0:  # add 'N' prefix if coordinates start before (offset-corrected) FASTA
                    prefix = 'N' * abs(start)
                    start = 0
                self.anno[g]['dna_seq'] = prefix + fasta.fetch(reference=g.chromosome, start=start, end=end)
                # add 'N' postfix if coordinates exceed available sequence in fasta
                self.anno[g]['dna_seq'] += 'N' * (len(g) - len(self.anno[g]['dna_seq']))
        self.has_seq = True

    def __getitem__(self, key):
        """ Returns the feature with the passed feature_id """
        if isinstance(key, str):
            if key in self.gene:
                return self.gene[key]
            return self.transcript[key]
        elif isinstance(key, GI) or isinstance(key, GI_dataclass):
            return self.anno[key]
        else:
            raise TypeError('Index must be a GI or a feature id string, not {type(key).__name__}')

    def add(self, location: GI, feature_id: str, feature_type: str, parent=None, children: tuple = ()):  # -> Feature:
        """ Adds a feature to the transcriptome """
        assert location.chromosome in self.merged_refdict, (f"Chromosome {location.chromosome} not in RefDict of this "
                                                            f"transcriptome")
        f = Feature(chromosome=location.chromosome, start=location.start, end=location.end, strand=location.strand,
                    transcriptome=self, feature_id=feature_id, feature_type=feature_type, parent=parent,
                    subfeature_types=children)
        self.anno[f] = {}
        return f

    def find_attr_rec(self, f, attr):
        """ recursively finds attribute from parent(s) """
        if f is None:
            return None, None
        if attr in self.anno[f]:
            return f, self.anno[f][attr]
        return self.find_attr_rec(f.parent, attr)

    def get_sequence(self, feature, mode='dna', show_exon_boundaries=False):
        """
            Returns the sequence of the passed feature.

            - If mode is 'rna' then the reverse complement of negative-strand features (using a DNA alphabet) will be
              returned.
            - if mode is 'spliced', the fully spliced sequence of a transcript will be returned. This will always use
              'rna' mode and is valid only for containers of exons.
            - if mode is 'translated' then the CDS sequence is reported. To, e.g., calculate the amino-acid
              sequence of a transcript using biopython's Seq() implementation, you can do:
              `Seq(t.transcript[my_tid].translated_sequence).translate()`
            - else, the 5'-3' DNA sequence (as shown in a genome browser) of this feature is returned

            show_exon_boundaries=True can be used to insert '*' characters at splicing boundaries of spliced/translated
            sequences.

        """
        if mode == 'spliced':
            assert 'exon' in feature.subfeature_types, "Can only splice features that have annotated exons"
            sep = '*' if show_exon_boundaries else ''
            fseq = self.get_sequence(feature, mode='dna')
            if fseq is None:
                return None
            if feature.strand == '-':
                seq = reverse_complement(
                    sep.join([fseq[(ex.start - feature.start):(ex.start - feature.start) + len(ex)] for ex in
                              reversed(feature.exon)]))
            else:
                seq = sep.join([fseq[(ex.start - feature.start):(ex.start - feature.start) + len(ex)] for ex in
                                feature.exon])
        elif mode == 'translated':
            assert 'CDS' in feature.subfeature_types, "Can only translate features that have annotated CDS"
            sep = '*' if show_exon_boundaries else ''
            fseq = self.get_sequence(feature, mode='dna')
            if fseq is None:
                return None
            if feature.strand == '-':
                seq = reverse_complement(
                    sep.join([fseq[(cds.start - feature.start):(cds.start - feature.start) + len(cds)] for cds in
                              reversed(feature.CDS)]))
            else:
                seq = sep.join([fseq[(cds.start - feature.start):(cds.start - feature.start) + len(cds)] for cds in
                                feature.CDS])
        else:
            p, pseq = self.find_attr_rec(feature, 'dna_seq')
            if p is None:
                return None
            if p == feature:
                seq = pseq
            else:
                idx = feature.start - p.start
                seq = pseq[idx:idx + len(feature)]  # slice from parent sequence
            if (seq is not None) and (mode == 'rna') and (feature.strand == '-'):  # revcomp if rna mode and - strand
                seq = reverse_complement(seq)
        return seq

    def slice_from_parent(self, f, attr, default_value=None):
        """
            Gets an attr from the passed feature or its predecessors (by traversing the parent/child relationships).
            If retrieved from an (enveloping) parent interval, the returned value will be sliced.
            Use only to access attributes that contain one item per genomic position (e.g, arrays of per-position
            values)
        """
        p, pseq = self.find_attr_rec(f, attr)
        if p is None:
            return default_value
        if p == f:
            return pseq
        else:
            idx = f.start - p.start
            return pseq[idx:idx + len(f)]  # slice from parent sequence

    def gene_triples(self, max_dist=None):
        """
            Convenience method that yields genes and their neighbouring (up-/downstream) genes.
            If max_dist is set and the neighbours are further away (or on other chromosomes),
            None is returned.

            To iterate over all neighbouring genes within a given genomic window, consider query()
            or implement a custom iterator.
        """
        for (x, y, z) in triplewise(chain([None], self.genes, [None])):
            if max_dist is not None:
                dx = None if x is None else x.distance(y)
                if (dx is None) or (abs(dx) > max_dist):
                    x = None
                dz = None if z is None else z.distance(y)
                if (dz is None) or (abs(dz) > max_dist):
                    z = None
            yield x, y, z

    def query(self, query, feature_types=None, envelop=False, sort=True):
        """
            Query features of the passed class at the passed query location.

            Parameters
            ----------
            query : GenomicInterval or string that is parsed by GI.from_str()
                Query interval
            feature_types : str or List[str]
                Feature types to query. If None, all feature types will be queried.
            envelop : bool
                If true, only features fully contained in the query interval are returned.
            sort : bool
                If true, the returned features will be sorted by chromosome and start coordinate.
        """
        if isinstance(query, str):
            query = gi(query)
        if query.chromosome not in self.chr2itree:
            return []
        if isinstance(feature_types, str):
            feature_types = (feature_types,)
        # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
        overlapping_genes = [x.data for x in self.chr2itree[query.chromosome].overlap(query.start, query.end + 1)]
        overlapping_features = overlapping_genes if (feature_types is None) or ('gene' in feature_types) else []
        if envelop:
            overlapping_features = [g for g in overlapping_features if query.envelops(g)]
        for g in overlapping_genes:
            if envelop:
                overlapping_features += [f for f in g.features(feature_types) if query.envelops(f)]
            else:
                overlapping_features += [f for f in g.features(feature_types) if query.overlaps(f)]
        if sort:
            overlapping_features.sort(key=lambda x: (self.merged_refdict.index(x.chromosome), x))
        return overlapping_features

    def annotate(self, anno_its, fun_anno, labels=None, region=None,
                 feature_types=None, disable_progressbar=True):
        """ Annotates all features of the configured type and in the configured genomic region using the passed fun_anno
            function.
            NOTE: consider removing previous annotations with the clear_annotations() functions before (re-)annotating
            a transcriptome.
        """
        with AnnotationIterator(
                TranscriptomeIterator(self, region=region, feature_types=feature_types),
                anno_its, labels) as it:
            for item in (pbar := tqdm(it, disable=disable_progressbar)):
                pbar.set_description(f"buffer_size={[len(x) for x in it.buffer]}")
                fun_anno(item)

    def save(self, out_file):
        """
            Stores this transcriptome and all annotations as dill (pickle) object.
            Note that this can be slow for large-scaled transcriptomes and will produce large ouput files.
            Consider using save_annotations()/load_annotations() to save/load only the annotation dictionary.
        """
        with open(out_file, 'wb') as out:
            dill.dump(self, out, recurse=True)
            # byref=True cannot vbe used as dynamically created dataclasses are not supported yet

    @classmethod
    def load(cls, in_file):
        """Load transcriptome from pickled file"""
        import gc
        gc.disable()  # disable garbage collector
        with open(in_file, 'rb') as infile:
            obj = dill.load(infile)
        gc.enable()
        obj.cached = True
        return obj

    def clear_annotations(self, retain_keys=('dna_seq',)):
        """
        Clears this transcriptome's annotations (except for retain_keys annotations (by default: 'dna_seq')).
        """
        for a in self.anno:
            if retain_keys is None:
                self.anno[a] = {}
            else:
                for k in {k for k in self.anno[a].keys() if k not in retain_keys}:
                    del self.anno[a][k]

    def save_annotations(self, out_file, keys=None):
        """
            Stores this transcriptome annotations as dill (pickle) object.
            Note that the data is stored not by object reference but by comparison
            key, so it can be assigned to newly created transcriptome objects
        """
        with open(out_file, 'wb') as out:
            if keys:  # subset some keys
                dill.dump({k.key(): {x: v[x] for x in v.keys() & keys} for k, v in self.anno.items()}, out,
                          recurse=True)
            else:
                dill.dump({k.key(): v for k, v in self.anno.items()}, out, recurse=True)

    def load_annotations(self, in_file, update=False):
        """
            Loads annotation data from the passed dill (pickle) object.
            If update is true, the current annotation dictionary will be updated.
        """
        with open(in_file, 'rb') as infile:
            anno = dill.load(infile)
            k2o = {f.key(): f for f in self.anno}
            for k, v in anno.items():
                assert k in k2o, f"Could not find target feature for key {k}"
                if update:
                    self.anno[k2o[k]].update(v)
                else:
                    self.anno[k2o[k]] = v

    def to_gff3(self, out_file, bgzip=True,
                feature_types=('gene', 'transcript', 'exon', 'intron', 'CDS', 'three_prime_UTR', 'five_prime_UTR')):
        """
            Writes a GFF3 file with all features of the configured types.

            Parameters
            ----------
            out_file : str
                The output file name
            bgzip : bool
                If true, the output file will be bgzipped and tabixed.
            feature_types : tuple
                The feature types to be included in the output file.
                For the used feature type names, see
                @see https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md

            Returns
            -------
            the name of the (bgzipped) output file

            Examples
            --------
            >>> transcriptome = ...
            >>> transcriptome.to_gff3('introns.gff3', feature_types=['intron'])
            # creates a file containing all intron annotations
        """

        def write_line(o, feature_type, data_dict, out_stream):
            print("\t".join([str(x) for x in [
                o.chromosome,
                'rnalib',
                feature_type,
                o.start,  # start
                o.end,  # end
                to_str(o.score if hasattr(o, 'score') else None, na='.'),
                '.' if o.strand is None else o.strand,
                to_str(o.phase if hasattr(o, 'phase') else None, na='.'),
                to_str([f'{k}={v}' for k, v in data_dict.items()], sep=';')
            ]]), file=out_stream)

        copied_fields = [x for x in self.copied_fields if x not in ['score', 'phase']]
        with (open(out_file, 'w') as out):
            with self.iterator(feature_types=feature_types) as it:
                for f, dat in it:
                    if f.feature_type == 'gene':
                        info = {'ID': f.feature_id, 'gene_name': f.gene_name}
                    else:
                        info = {'ID': f.feature_id}
                        if f.parent:
                            info['Parent'] = f.parent.feature_id
                    info.update({k: getattr(f, k) for k in copied_fields if hasattr(f, k)})  # add copied fields
                    write_line(f, f.feature_type, info, out)
        if bgzip:
            bgzip_and_tabix(out_file)
            return out_file + '.gz'
        return out_file

    def to_bed(self, out, region=None, feature_types=('transcript',),
               fun_anno=_transcript_to_bed, bed_header=None, disable_progressbar=True, no_header=False,
               ):
        """Outputs transcripts of this transcriptome in BED format.
            Pass your custom annotation function via fun_anno to output custom BED fields (e.g., color based on
            transcript type).

            Parameters
            ----------
            out : file-like object
                The output file-like object.
            region : gi or str
                The genomic region to be included in the output file. If None (default), all features will be included.
            feature_types : tuple
                The feature types to be included in the output file. Default: transcripts only.
            fun_anno : function
                A function that takes a feature index, a location and a feature as input and returns a tuple of
                strings that will be added to the BED output file.
            bed_header : dict
                A dict containing the BED header fields. If None (default), a default header will be used.
            disable_progressbar : bool
                If true, the progressbar will be disabled.
            no_header : bool
                If true, no header will be written to the output file.

            Example
            -------
            >>> transcriptome = ...
            >>> transcriptome.to_bed('transcripts.bed')
        """
        if bed_header is None:
            bed_header = {"name": self.name, "description": self.name, "useScore": 0, "itemRgb": "On"}
        self.iterator(region=region, feature_types=feature_types).to_bed(out, fun_anno, bed_header,
                                                                         disable_progressbar, no_header)

    def __len__(self):
        return len(self.anno)

    def __repr__(self):
        return f"{self.name} with {len(self.genes)} genes and {len(self.transcripts)} tx" + (
            " (+seq)" if self.has_seq else "") + (" (cached)" if self.cached else "")

    def iterator(self, region=None, feature_types=None):
        """ returns a :class:`.TranscriptomeIterator` for iterating over all features of the passed type(s).
            If feature_types is None (default), all features will be returned.
        """
        return TranscriptomeIterator(self, region=region, feature_types=feature_types)

    def __iter__(self):
        with self.iterator() as it:
            yield from it

    def get_struct(self):
        """Return a dict mapping feature to child feature types"""
        return self._ft2child_ftype


@dataclass(frozen=True, repr=False)
class Feature(GI_dataclass):
    """
        A (frozen) genomic feature, e.g., a gene, a transcript, an exon, an intron,
        a CDS or a three_prime_UTR/five_prime_UTR. Features are themselves containers of sub-features
        (e.g., a gene contains transcripts, a transcript contains exons, etc.). Sub-features are stored in
        the respective child tuples (e.g., gene.transcript, transcript.exon, etc.).

        Features are built by the Transcriptome class and are typically not instantiated directly. They
        contain a reference to their parent feature (e.g., a transcript's parent is a gene) and a list of
        child features (e.g., a gene's children are transcripts). The parent-child relationships are
        defined by the transcriptome implementation and are typically defined by the GFF file used to build
        the transcriptome. For example, a GFF file may contain a gene feature with a child transcript feature
        and the transcript feature may contain a child exon feature. In this case, the transcriptome
        implementation will define the parent-child relationships as follows:
        gene.parent = None
        gene.children = {'transcript': [transcript1, transcript2, ...]}
        transcript.parent = gene
        transcript.children = {'exon': [exon1, exon2, ...]}
        exon.parent = transcript
        exon.children = {}

        Features will also contain a dynamically created dataclass (see create_sub_class()) that contains
        annotations parsed from the GFF file. For example, a gene feature may contain a 'gene_name' annotation
        that was parsed from the GFF file. Annotations are stored in the feature's anno dict and can be accessed
        via <feature>.<annotation> (e.g., gene.gene_name). Annotations are typically strings but can be of any
        type.

        Equality of features is defined by comparing their genomic coordinates and strand,
        as well as their feature_type (e.g., transcript, exon, five_prime_UTR) and feature_id (which should be unique
        within a transcriptome), as returned by the key() method.

        Features are typically associated with the `Transcriptome` object used to create them and (mutable)
        additional annotations are stored in the respective transcriptome's `anno` dict can be directly accessed via
        <feature>.<annotation>. This hybrid approach allows efficient and transparent access to readonly (immutable)
        annotations parsed from the GFF file as well as to mutable annotations that are added later (e.g, using the
        Transcriptome.annotate() function).

        This also includes dynamically calculated (derived) annotations such as feature sequences that are  sliced from
        a feature's predecessor via the get_sequence() method. Rnalib stores sequences only at the gene level and
        slices the subsequences of child features from those to minimize storage requirements.

        For example, if the sequence of an exon is requested via `exon.sequence` then the Feature implementation will
        search for a 'sequence' annotation in the exons super-features by recursively traversing 'parent' relationships.
        The exon sequence will then be sliced form this sequence by comparing the respective genomic coordinates (which
        works only if parent intervals always envelop their children as asserted by the transcriptome implementation).

        This mechanism can also be used for any other sliceable annotation (e.g., a numpy array) that is stored in the
        transcriptome's anno dict. To access such annotations, use the get(..., slice_from_parent=True) method (e.g.,
        `exon.get('my_array', slice_from_parent=True)`). This will recursively traverse the parent/child
        relationships until the my_array annotation is found and then slice the respective section from there (assuming
        that array indices correspond to genomic coordinates).

        Subfeatures of a feature can be accessed via the features() method which returns a generator over all such
        features that is unsorted by default. To iterate over features in a sorted order, consider using the
        TranscriptomeIterator class.

        To inspect a features methods and attributes, use vars(feature), dir(feature) or help(feature).
    """
    transcriptome: Transcriptome = None  # parent transcriptome
    feature_id: str = None  # unique feature id
    feature_type: str = None  # a feature type (e.g., exon, intron, etc.)
    parent: object = field(default=None, hash=False, compare=False)  # an optional parent
    subfeature_types: tuple = tuple()  # sub-feature types

    def __repr__(self) -> str:
        return f"{self.feature_type}@{self.chromosome}:{self.start}-{self.end}"

    def key(self) -> tuple:
        """ Returns a tuple containing feature_id, feature_type and genomic coordinates including strand """
        return self.feature_id, self.feature_type, self.chromosome, self.start, self.end, self.strand

    def __eq__(self, other):
        """ Compares two features by key. """
        # if issubclass(other.__class__, Feature): # we cannot check for subclass as pickle/unpickle by ref will
        # result in different parent classes
        return self.key() == other.key()

    def __getattr__(self, attr):
        if attr == 'location':
            return self.get_location()
        elif attr == 'rnk':
            return self.get_rnk()
        elif self.transcriptome:  # get value from transcriptome anno dict
            if attr == 'sequence':
                return self.transcriptome.get_sequence(self)
            elif attr == 'spliced_sequence':
                return self.transcriptome.get_sequence(self, mode='spliced')
            elif attr == 'translated_sequence':
                return self.transcriptome.get_sequence(self, mode='translated')
            if attr in self.transcriptome.anno[self]:
                return self.transcriptome.anno[self][attr]
        raise AttributeError(f"{self.feature_type} has no attribute/magic function {attr}")

    def get(self, attr, default_value=None, slice_from_parent=False):
        """ Safe getter supporting default value and slice-from-parent """
        if slice_from_parent and (self.transcriptome is not None):
            return self.transcriptome.slice_from_parent(self, attr, default_value=default_value)
        else:
            return getattr(self, attr, default_value)

    @classmethod
    def from_gi(cls, loc, ):
        """ Init from gi """
        return cls(loc.chromosome, loc.start, loc.end, loc.strand)

    def get_location(self):
        """Returns a genomic interval representing the genomic location of this feature."""
        return gi(self.chromosome, self.start, self.end, self.strand)

    def get_rnk(self):
        """Rank (1-based index) of feature in this feature's parent children list"""
        if not self.parent:
            return None
        return self.parent.__dict__[self.feature_type].index(self) + 1

    def features(self, feature_types=None):
        """ Yields all sub-features (not sorted).
            To get a coordinate-sorted iterator, use sorted(feature.features()). Sorting by chromosome
            is not required as subfeatures are enveloped by their parents by convention.
        """
        for ft in self.subfeature_types:
            for f in self.__dict__[ft]:
                if (not feature_types) or (f.feature_type in feature_types):
                    yield f
                for sf in f.features():  # recursion
                    if (not feature_types) or (sf.feature_type in feature_types):
                        yield sf

    # dynamic feature class creation
    @classmethod
    def create_sub_class(cls, feature_type, annotations: dict = None, child_feature_types: list = None):
        """ Create a subclass of feature with additional fields (as defined in the annotations dict)
            and child tuples
        """
        fields = [('feature_id', str, field(default=None)), ('feature_type', str, field(default=feature_type))]
        fields += [(k, v, field(default=None)) for k, v in annotations.items() if
                   k not in ['feature_id', 'feature_type']]
        if child_feature_types is not None:
            fields += [(k, tuple, field(default=tuple(), hash=False, compare=False)) for k in child_feature_types]
        sub_class = make_dataclass(feature_type, fields=fields, bases=(cls,), frozen=True, repr=False, eq=False)
        return sub_class


class _Feature:
    """
        A mutable genomic (annotation) feature that is used only for building a transcriptome.
    """

    def __init__(self, transcriptome, feature_type, feature_id, loc=None, parent=None, children=None):
        self.transcriptome = transcriptome
        self.loc = loc
        self.feature_type = feature_type
        self.feature_id = feature_id
        self.parent = parent
        self.children = {} if children is None else children
        self.anno = {}
        assert loc is not None

    def get_anno_rec(self):
        """compiles a dict containing all annotations of this feature and all its children per feature_type"""
        a = {self.feature_type: {k: type(v) for k, v in self.anno.items()}}
        t = {self.feature_type: set()}
        s = {self.feature_type}
        if self.children:
            for cat in self.children:
                t[self.feature_type].add(cat)
                s.add(cat)
                for c in self.children[cat]:
                    x, y, z = c.get_anno_rec()
                    a.update(x)
                    t.update(y)
                    s.update(z)
        return a, t, s

    def set_location(self, loc):
        self.loc = loc

    # def __repr__(self):
    #     return feature"{self.feature_type}@{super().__repr__()} ({ {k: len(v) for k, v in self.children.items()}
    #     if self.children else 'NA'})"

    def freeze(self, ft2class):
        """Create a frozen instance (recursively)"""
        f = ft2class[self.feature_type].from_gi(self.loc)
        object.__setattr__(f, 'transcriptome', self.transcriptome)
        object.__setattr__(f, 'feature_id', self.feature_id)
        for k, v in self.anno.items():
            object.__setattr__(f, k, v)
        if self.children:
            object.__setattr__(f, 'subfeature_types', tuple(self.children))
            for k in self.children:
                children = [x.freeze(ft2class) for x in self.children[k]]
                if self.loc.strand == '-':  # reverse order if on neg strand
                    children = list(reversed(children))
                for c in children:
                    object.__setattr__(c, 'parent', f)
                object.__setattr__(f, k, tuple(children))
        return f


class AbstractFeatureFilter(ABC):
    """ For filtering genes/transcripts entries from a GFF when building a transcriptome.
    """

    @abstractmethod
    def filter(self, loc, info):  # -> bool, str:
        """Returns true if the passed feature should be filtered, false otherwise.
            The filter can also return a filter message that will be added to the log.
        """
        pass

    def get_chromosomes(self):  # -> Set[str]:
        """Returns the set of chromosomes to be included or None if all (no filtering applied)."""
        return None


class TranscriptFilter(AbstractFeatureFilter):
    """A transcript filter that can be used to filter genes/transcripts by location and/or feature type
    specific info fields. It can be configured by chaining include_* methods or by passing a dict to the
    constructor. Features will first be filtered by location, then by included and then by excluded feature type
    specific info fields.

    For location filtering, the following rules apply:

    - if included_chrom is not None, only transcripts on these chromosomes will be included
    - if excluded_chrom is not None, transcripts on these chromosomes will be excluded
    - if included_regions is not None, only transcripts overlapping these regions will be included
    - if excluded_regions is not None, transcripts overlapping these regions will be excluded

    For feature type specific filtering, the following rules apply:

    - if included is not None, only transcripts with the specified feature type specific info fields
        will be included. Info fields are parsed to sets of values by splitting on `','`.
        if the feature type specific info field is not present, the feature will be
        filtered unless None is added to the included list.
    - if excluded is not None, transcripts with the specified feature type specific info fields
        will be excluded. Info fields are parsed to sets of values by splitting on `','`.

    Examples
    --------
    >>> # create a filter that includes only protein coding genes
    >>> tf1 = TranscriptFilter().include_gene_types({'protein_coding'})
    >>> # create a filter that includes only protein coding genes and transcripts on chr1
    >>> tf2 = TranscriptFilter().include_gene_types({'protein_coding'}).include_chromosomes({'chr1'})
    >>> # create a filter that includes only canonical genes and transcripts in a given region.
    >>> # Features without tags will be included
    >>> tf3 = TranscriptFilter().include_tags({'Ensembl_canonical', None}).include_regions({'chr1:1-10000'})
    >>> # create a filtered Transcriptome
    >>> config = { ... }
    >>> t = Transcriptome(config, feature_filter=tf3)


    Notes
    -----
    TODO

    * add greater, smaller than filters for feature type specific info fields
    * redesign?
    """

    def __init__(self, config=None):
        self.config = AutoDict() if config is None else config
        self.included_chrom = get_config(self.config, ['location', 'included', 'chromosomes'],
                                         default_value=None)
        self.excluded_chrom = get_config(self.config, ['location', 'excluded', 'chromosomes'],
                                         default_value=None)
        self.included_regions = get_config(self.config, ['location', 'included', 'regions'],
                                           default_value=None)
        self.excluded_regions = get_config(self.config, ['location', 'excluded', 'regions'],
                                           default_value=None)
        if self.included_chrom is not None:
            self.included_chrom = set(self.included_chrom)
        if self.excluded_chrom is not None:
            self.excluded_chrom = set(self.excluded_chrom)
        if self.included_regions is not None:
            self.included_regions = {GI.from_str(s) for s in self.included_regions}
        if self.excluded_regions is not None:
            self.excluded_regions = {GI.from_str(s) for s in self.excluded_regions}

    def filter(self, loc, info):
        # location filtering
        if self.included_chrom is not None and loc.chromosome not in self.included_chrom:
            return True, 'included_chromosome'
        if self.excluded_chrom is not None and loc.chromosome in self.excluded_chrom:
            return True, 'excluded_chromosome'
        if self.included_regions is not None and not any(loc.overlaps(r) for r in self.included_regions):
            return True, 'included_location'
        if self.excluded_regions is not None and any(loc.overlaps(r) for r in self.excluded_regions):
            return True, 'excluded_region'
        # feature type specific info field filtering
        if 'feature_type' not in info:
            return False, 'no_feature_type'
        included = get_config(self.config, [info['feature_type'], 'included'], default_value=None)
        if included is not None:
            for info_field in included:
                if info_field not in info:
                    if None not in included[info_field]:
                        return True, f'missing_{info_field}'
                else:
                    found_values = set(info[info_field].split(','))
                    n_found = len(set(included[info_field]) & found_values)
                    if n_found == 0:  # values not found
                        return True, f'missing_{info_field}_value'
        excluded = get_config(self.config, [info['feature_type'], 'excluded'], default_value=None)
        if excluded is not None:
            for info_field in excluded:
                if info_field not in info:
                    return False, f'found_{info_field}'  # excluded does not support None
                found_values = set(info[info_field].split(','))
                n_found = len(set(excluded[info_field]) & found_values)
                if n_found > 0:  # values found
                    return True, f'found_{info_field}_value'
        return False, 'passed'

    def get_chromosomes(self):
        """Returns the set of chromosomes to be included"""
        if self.included_chrom is None:
            return None
        return self.included_chrom.difference({} if self.excluded_chrom is None else self.excluded_chrom)

    def __repr__(self):
        return json.dumps(self.config, indent=4)

    def include_chromosomes(self, chromosomes: set):
        """Convenience method to add chromosomes to the included_chrom set"""
        chromosomes = to_set(chromosomes)
        if self.included_chrom is None:
            self.included_chrom = chromosomes
        else:
            self.included_chrom.update(chromosomes)
        self.config['location']['included']['chromosomes'] = list(self.included_chrom)
        return self

    def include_regions(self, regions: set):
        """Convenience method to add regions to the included_regions set"""
        if isinstance(regions, str):
            regions = {gi(r) for r in regions.split(',')}
        if self.included_regions is None:
            self.included_regions = regions
        else:
            self.included_regions.update(regions)
        self.config['location']['included']['regions'] = list(self.included_regions)
        return self

    def include_gene_ids(self, ids: set):
        """Convenience method to add included gene ids"""
        self.config['gene']['included']['gene_id'] = to_set(ids)
        return self

    def include_transcript_ids(self, ids: set):
        """Convenience method to add included transcript ids"""
        self.config['transcript']['included']['transcript_id'] = to_set(ids)
        return self

    def include_gene_types(self, gene_types: set, include_missing=True, feature_type='gene'):
        """Convenience method to add included gene_types to gene+transcript inclusion rules. Use, e.g.,
        {'protein_coding'} to load only protein coding genes. If include_missing is True then genes/transcripts
        without gene_type will also be included. """
        gene_types = to_set(gene_types)
        self.config[feature_type]['included']['gene_type'] = list(gene_types) + [
            None] if include_missing else list(gene_types)
        return self

    def include_transcript_types(self, transcript_types: set, include_missing=True, feature_type='transcript'):
        """Convenience method to add included transcript_types. Use, e.g., {'miRNA'} to load only
        miRNA transcripts. If include_missing is True (default) then transcripts without transcript_type will
        also be included. """
        transcript_types = to_set(transcript_types)
        self.config[feature_type]['included']['transcript_type'] = list(transcript_types) + [
            None] if include_missing else list(transcript_types)
        return self

    def include_tags(self, gene_tags: set, include_missing=True, feature_type='transcript'):
        """Convenience method to add included gene tags. Use, e.g., {'Ensembl_canonical'} to load only
            canonical genes. If include_missing is True then genes/transcripts without tags
            will also be included. """
        gene_tags = to_set(gene_tags)
        self.config[feature_type]['included']['tag'] = list(gene_tags) + [
            None] if include_missing else list(gene_tags)
        return self


class RefDict(abc.Mapping[str, int]):
    """
        Named mapping for representing a set of references (contigs) and their lengths.

        Supports aliasing by passing a function (e.g., fun_alias=toggle_chr which will add/remove 'chr' prefixes) to
        easily integrate genomic files that use different (but compatible) reference names. If an aliasing function is
        passed, original reference names are accessible via the orig property. An aliasing function must be reversible,
        i.e., fun_alias(fun_alias(str))==str and support None.

        Note that two reference dicts match if their (aliased) contig dicts match (name of RefDict is not
        compared).
    """

    def __init__(self, d, name=None, fun_alias=None):
        self.d = d
        self.name = name
        self.fun_alias = fun_alias
        if fun_alias is not None:
            self.orig = d.copy()
            self.d = {fun_alias(k): v for k, v in d.items()}  # apply fun to keys
        else:
            self.orig = self

    def __getitem__(self, key):
        return self.d[key]

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def has_len(self, chrom=None):
        """Returns true if the passed chromosome (or all if chrom is None) has an assigned length, false otherwise."""
        if chrom is None:
            return not all(v is None for v in self.values())
        return self.d[chrom] is not None

    def tile(self, tile_size=int(1e6)):
        """
            Iterates in an ordered fashion over the reference dict, yielding non-overlapping genomic intervals of the
            given tile_size (or smaller at chromosome ends).
        """
        for chrom, chrlen in self.d.items():
            chrom_gi = gi(chrom, 1, chrlen)  # will use maxint if chrlen is None!
            for tile in chrom_gi.split_by_maxwidth(tile_size):
                yield tile

    def __repr__(self):
        return (f"RefDict (size: {len(self.d.keys())}): {self.d.keys()}"
                f"{f' (aliased from {self.orig.keys()})' if self.fun_alias else ''}, {self.d.values()} name:"
                f" {self.name} ")

    def chromosomes(self):
        """Returns a list of chromosome names"""
        return list(self.d.keys())

    def chromosomes_orig(self):
        """Returns a list of chromosome names"""
        return list(self.orig.keys())

    def alias(self, chrom):
        if self.fun_alias:
            return self.fun_alias(chrom)
        return chrom

    def index(self, chrom):
        """ Index of the passed chromosome, None if chromosome not in refdict or -1 if None was passed.
            Useful, e.g., for sorting genomic coordinates
        """
        if not chrom:
            return -1
        try:
            return list(self.keys()).index(chrom)
        except ValueError:
            warnings.warn(f"{chrom} not in refdict")
            return None

    @staticmethod
    def merge_and_validate(*refdicts, check_order=False, included_chrom=None):
        """
            Checks whether the passed reference sets are compatible and returns the
            merged reference set containing the intersection of common references.

            Parameters
            ----------
            refdicts:
                list of RefDicts
            check_order:
                if True, the order of the common references is asserted. default: False
            included_chrom:
                if passed, only the passed chromosomes are considered. default: None

            Returns
            -------
            RefDict containing the intersection of common references
        """
        refdicts = [r for r in refdicts if r is not None]
        if len(refdicts) == 0:
            return None
        # intersect all contig lists while preserving order (set.intersection() or np.intersect1d() do not work!)
        shared_ref = {k: None for k in intersect_lists(*[list(r.keys()) for r in refdicts], check_order=check_order) if
                      (included_chrom is None) or (k in included_chrom)}
        # check whether contig lengths match
        for r in refdicts:
            for contig, oldlen in shared_ref.items():
                newlen = r.get(contig)
                if newlen is None:
                    continue
                if oldlen is None:
                    shared_ref[contig] = newlen
                else:
                    assert oldlen == newlen, (f"Incompatible lengths for contig ({oldlen}!={newlen}) when comparing "
                                              f"RefDicts {refdicts}")
        return RefDict(shared_ref, name=','.join([r.name if r.name else "<unnamed refdict>" for r in refdicts]),
                       fun_alias=None)

    @staticmethod
    def load(fh, fun_alias=None, calc_chromlen=False):
        """ Extracts chromosome names, order and (where possible) length from pysam objects.

        Parameters
        ----------
        fh : pysam object or file path (str)
        fun_alias : aliasing functions (see RefDict)
        calc_chromlen : bool, if True, chromosome lengths will be calculated from the file (if required)

        Returns
        -------
        dict: chromosome name to length

        Raises
        ------
        NotImplementedError
            if input type is not supported yet
        """
        was_opened = False
        try:
            if isinstance(fh, str):
                fh = open_file_obj(fh)
                was_opened = True
            if isinstance(fh, pysam.Fastafile):  # @UndefinedVariable
                return RefDict({c: fh.get_reference_length(c) for c in fh.references},
                               name=f'References from FASTA file {fh.filename}', fun_alias=fun_alias)
            elif isinstance(fh, pysam.AlignmentFile):  # @UndefinedVariable
                return RefDict({c: fh.header.get_reference_length(c) for c in fh.references},
                               name=f'References from SAM/BAM file {fh.filename}', fun_alias=fun_alias)
            elif isinstance(fh, pysam.TabixFile):  # @UndefinedVariable
                if calc_chromlen:  # no ref length info in tabix, we need to iterate :-(
                    refdict = {}
                    for c in fh.contigs:
                        line = ('', '', '0')  # default
                        for _, line in enumerate(fh.fetch(c)):
                            pass  # move to last line
                        refdict[c] = int(line.split('\t')[2])
                else:
                    refdict = {c: None for c in fh.contigs}  # no ref length info in tabix...
                return RefDict(refdict, name=f'References from TABIX file {fh.filename}',
                               fun_alias=fun_alias)
            elif isinstance(fh, pysam.VariantFile):  # @UndefinedVariable
                return RefDict({c: fh.header.contigs.get(c).length for c in fh.header.contigs},
                               name=f'References from VCF file {fh.filename}', fun_alias=fun_alias)
            elif isinstance(fh, pyBigWig.pyBigWig):  # @UndefinedVariable
                return RefDict({c: l for c, l in fh.chroms().items()},
                               name=f'References from bigWig/bigBed file {fh}', fun_alias=fun_alias)
            else:
                raise NotImplementedError(f"Unknown input object type {type(fh)}")
        finally:
            if was_opened:
                fh.close()


# class Item(namedtuple('Item',['location','data'])):
#     """ A location, data tuple, returned by an LocationIterator """
#
#     def __len__(self):
#         """ Reports the length of the wrapped location, not the current tuple"""
#         return self.location.__len__()


class Item(NamedTuple):
    """ A location, data tuple, returned by an LocationIterator """
    location: gi
    data: Any

    def __len__(self):
        """ Reports the length of the wrapped location, not the current tuple"""
        return self.location.__len__()


class LocationIterator:
    """
        Superclass for genomic iterators (mostly based on pysam) for efficient, indexed iteration over genomic
        datasets. Most rnalib iterables inherit from this superclass.

        A LocationIterator iterates over a genomic dataset and yields `Item`s, that are tuples of genomic intervals and
        associated data. The data can be any object (e.g., a string, a dict, a list, etc.). Location iterators can be
        restricted to a genomic region.

        LocationIterators keep track of their current genomic location (self.location) as well as statistics about
        the iterated and yielded items (stats()). They can be consumed and converted to lists (self.to_list()),
        pandas dataframes (self.to_dataframe()), interval trees (self.to_intervaltrees()) or bed files (self.to_bed()).

        LocationIterators can be grouped or tiled, see the group() and tile() methods.

        LocationIterators are iterable, implement the context manager protocol ('with' statement) and provide a
        standard interface for region-type filtering (by passing chromosome sets or genomic regions of
        interest). Where feasible and not supported by the underlying (pysam) implementation, they implement chunked
        I/O where for efficient iteration (e.g., FastaIterator).

        The maximum number of iterated items can be queried via the max_items() method which tries to guesstimate this
        number (or an upper bound) from the underlying index data structures.

        Examples
        --------
        >>> locs, data = zip(*it('myfile.bed')) # creates a LocationIterator via factory and consumes items,
        splitting into locs and data lists.


        Attributes
        ----------
        _stats : Counter
            A counter that collects statistics (e.g., yielded items per chromosome) for QC and reporting.
        location : gi
            The current genomic location of this iterator


        Parameters
        ----------
        file : str or pysam file object
            A file path or an open file object. In the latter case, the object will not be closed
        region : GI or str
            genomic region to iterate
        file_format : str
            optional, will be determined from filename if omitted
        per_position : bool
            optional, if True, the iterator will yield per-position items (e.g., FASTA files)
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        refdict : RefDict
            optional, if set, the iterator will use the passed reference dict instead of reading it from the file

        Notes
        -----
        TODOs:

        * strand specific iteration
        * url streaming
        * add is_exhausted flag?
    """

    @abstractmethod
    def __init__(self, file,
                 region: GI = None,
                 file_format: str = None,
                 per_position: bool = False,
                 fun_alias: Callable = None,
                 refdict: RefDict = None):
        self._stats = Counter()  # counter for collecting stats
        self.location = None
        self.per_position = per_position
        if isinstance(file, (str, PathLike)):
            self.file = open_file_obj(file, file_format=file_format)  # open new object
            self._was_opened = True
        else:
            self.file = file
            self._was_opened = False
        self.fun_alias = fun_alias
        # use custom refdict or load from file if possible
        if refdict is None:
            assert file is not None, "No file or refdict passed"
            self.refdict = RefDict.load(self.file, fun_alias=fun_alias)
        else:
            self.refdict = refdict
        self.set_region(region)
        # region is a gi object. List of iterated chromosomes is set in self.chromosomes

    @property
    def stats(self):
        """ Returns the collected stats """
        return self._stats

    def set_region(self, region):
        """ Update the iterated region of this iterator.
            Note that the region's chromosome must be in this anno_its refdict (if any)
        """
        if region is not None:
            self.region = gi(region) if isinstance(region, str) else region
            if (self.refdict is not None) and self.region.chromosome is not None:
                assert self.region.chromosome in self.refdict, (f"{self.region.chromosome} not found in references"
                                                                f" {self.refdict}")
            # all chroms in correct order, as seen in the datafile (not aliased)
            self.chromosomes = [self.refdict.alias(self.region.chromosome)] if self.region.chromosome is not None else (
                self.refdict.orig.keys())
        else:
            self.region = gi()
            self.chromosomes = self.refdict.orig.keys()

    def to_list(self, style='item'):
        """ Convenience method that consumes iterator and returns a list of items, locations or data.
            Parameters
            ----------
            style : str
                'item' (default): all items with be returned
                'location': only the locations of the items will be returned
                 'data': only the data of the items will be returned


            Example
            -------
            >>> it('myfile.bed').to_list()
            >>> locs, data = zip(*it('myfile.bed'))
        """
        if style == "location":
            return [x.location for x in self]
        elif style == "data":
            return [x.data for x in self]
        return list(self)

    def to_bed(self, out,
               fun_anno=lambda idx, loc, item: (f"item{idx}", '.', loc.start - 1, loc.end, "0,0,0", 1, len(loc), 0),
               bed_header=None, disable_progressbar=True, no_header=False, n_col=12
               ):
        """ Consumes iterator and returns results in BED format to the passed output stream.
            out : file-like object
                The output file-like object.
            fun_anno : function
                A function that takes a feature index, a location and a feature as input and returns a tuple of
                strings that will be added to the BED output file.
            bed_header : dict
                A dict containing the BED header fields. If None (default), a default header will be used.
            disable_progressbar : bool
                If true, the progressbar will be disabled.
            no_header : bool
                If true, no header will be written to the output file.

            Example
            -------
            >>> out_file = ...
            >>> with open_file_obj(out_file, 'wt') as out:
            >>>     with LocationIterator(...) as it: # use respective subclass here
            >>>         it.to_bed(out)
            >>> bgzip_and_tabix(out_file)
        """
        if not no_header:
            if bed_header is None:
                bed_header = {'visibility': 1, 'itemRgb': 'On', 'useScore': 1}
            print(f'track {' '.join([f'{x}={bed_header[x]}' for x in bed_header])}', file=out)
        for idx, (loc, item) in tqdm(enumerate(self), desc=f"Writing bed file", disable=disable_progressbar):
            name, score, thickStart, thickEnd, rgb, blockCount, blockSizes, blockStarts = fun_anno(idx, loc, item)
            print(to_str([loc.chromosome, loc.start - 1, loc.end, name, score, loc.strand,
                          thickStart, thickEnd, rgb, blockCount, blockSizes, blockStarts][:n_col], sep='\t', na='.'),
                  file=out)

    def to_dataframe(self,
                     fun=lambda loc, item, fun_col, default_value: [str(item)],  # default: convert item to string repr
                     fun_col=('Value',),
                     coord_inc=(0, 0),
                     coord_colnames=('Chromosome', 'Start', 'End', 'Strand'),
                     excluded_columns=None,
                     included_columns=None,
                     dtypes=None,
                     default_value=None,
                     max_items=None,
                     disable_progressbar=True):
        """ Consumes iterator (up to max_items items) and returns results in a dataframe.
            Start/stop Coordinates will be corrected by the passed coord_inc tuple.

            Parameters
            ----------
            fun : function
                a function that converts the yielded items of this iterator into a tuple of values that represent
                fun_col column values in the created dataframe
            fun_col : tuple
                a tuple of column names for the created dataframe
            coord_inc : tuple
                a tuple of values to be added to start and end coordinates
            coord_colnames : tuple
                a tuple of column names for the created dataframe
            excluded_columns : tuple
                optional, a tuple of column names to be excluded from the created dataframe
            included_columns : tuple
                optional, a tuple of column names to be included in the created dataframe
            dtypes : dict
                optional, a dict of column names and their respective dtypes
            default_value : any
                optional, a default value to be used if a column value is not present in the iterated item
            max_items : int
                maximum number of included items (None: all)
            disable_progressbar : bool
                optional, if True, no progressbar will be shown
        """
        assert fun is not None
        assert (fun_col is not None) and (isinstance(fun_col, tuple))
        # exclude/include columns
        if excluded_columns is not None:
            fun_col = tuple(col for col in fun_col if col not in excluded_columns)
        if included_columns is not None:
            fun_col += included_columns
        if max_items is None:  # fast list comprehension
            df = pd.DataFrame([[loc.chromosome,
                                loc.start,
                                loc.end,
                                '.' if loc.strand is None else loc.strand] + fun(loc, item, fun_col, default_value) for
                               idx, (loc, item) in
                               enumerate(tqdm(self, desc=f"Building dataframe", disable=disable_progressbar))],
                              columns=coord_colnames + fun_col)
        else:  # construct (small) list and then build df
            lst = list()
            for idx, (loc, item) in enumerate(tqdm(self, desc=f"Building dataframe", disable=disable_progressbar)):
                if idx >= max_items:
                    break
                lst.append(
                    [loc.chromosome,
                     loc.start,
                     loc.end,
                     '.' if loc.strand is None else loc.strand] + fun(loc, item, fun_col, None)
                )
            df = pd.DataFrame(lst, columns=coord_colnames + fun_col)
        if dtypes is not None:
            for k, v in dtypes.items():
                df[k] = df[k].astype(v)
        # Correct coordinates if required
        for inc, col in zip(coord_inc, [coord_colnames[1], coord_colnames[2]]):
            if inc != 0:
                df[col] += inc
        return df

    def describe(self,
                 fun=lambda loc, item, fun_col, default_value: [str(item)],  # default: convert item to string repr
                 fun_col=('Value',),
                 coord_inc=(0, 0),
                 coord_colnames=('Chromosome', 'Start', 'End', 'Strand'),
                 excluded_columns=None,
                 included_columns=None,
                 dtypes=None,
                 default_value=None,
                 max_items=None,
                 disable_progressbar=True
                 ) -> Tuple[pd.DataFrame, dict]:
        """ Converts this iterator to a pandas dataframe and calls describe(include='all') """
        df = self.to_dataframe(
            fun=fun,
            fun_col=fun_col,
            coord_inc=coord_inc,
            coord_colnames=coord_colnames,
            excluded_columns=excluded_columns,
            included_columns=included_columns,
            dtypes=dtypes,
            default_value=default_value,
            max_items=max_items,
            disable_progressbar=disable_progressbar)
        # calculate overlap with bioframe and check for empty intervals (end<start)?
        return (df.describe(include='all'),
                {"contains_overlapping": len(df.index) > len(bioframe.merge(df, cols=('Chromosome', 'Start',
                                                                                      'End')).index),
                 "contains_empty": sum((df["End"] - df["Start"]) < 0) > 0})

    def to_intervaltrees(self, disable_progressbar=False):
        """ Consumes iterator and returns results in a dict of intervaltrees.

            .. warning::
                NOTE that this will silently drop empty intervals

            Notes
            -----
            * TODO
                improve performance with from_tuples method
        """
        chr2itree = {}  # a dict mapping chromosome ids to annotation interval trees.
        for loc, item in tqdm(self, desc=f"Building interval trees", disable=disable_progressbar):
            if loc.is_empty():
                continue
            if loc.chromosome not in chr2itree:
                chr2itree[loc.chromosome] = IntervalTree()
            # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
            chr2itree[loc.chromosome].addi(loc.start, loc.end + 1, item)
        return chr2itree

    def group(self, strategy='both'):
        """ Wraps this iterator in a GroupedLocationIterator """
        return GroupedLocationIterator(self, strategy=strategy)

    def tile(self, regions_iterable:Iterable[GI]=None, tile_size=1e8):
        """ Wraps this iterator in a TiledIterator.
            Parameters
            ----------
            regions_iterable : iterable
                an iterable of genomic regions (GI) that defines the tiles. Will be calculated from the refdict if None.
            tile_size : int
                the size of the tiles
        """
        return TiledIterator(location_iterator=self,
                             regions_iterable=regions_iterable,
                             tile_size=tile_size)

    @abstractmethod
    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown.
            Note that this is the upper bound of yielded items but less (or even no) items may be yielded
            based on filter settings, etc. Useful, e.g., for progressbars or time estimates
        """
        return None

    @abstractmethod
    def __iter__(self) -> Item:
        yield None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """ Closes the underlying file object if it was opened by this iterator """
        if self.file and self._was_opened:
            # logging.debug(feature"Closing iterator {self}")
            self.file.close()


class MemoryIterator(LocationIterator):
    """
        A location iterator that iterates over intervals retrieved from one of the following datastructures:

        * {str:gi} dict: yields (gi,str); note that the passed strings must be unique
        * {gi:any} dict: yields (gi,any)
        * iterable of gis:  yields (gi, index in input iterable)

        Regions will be sorted according to a refdict that is created from the data automatically.

        Notes
        -----
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)

    """

    # d = { gi(1, 10,11): 'a', gi(1, 1,10): 'b', gi(2, 1,10): 'c' }
    def __init__(self, d, region=None, fun_alias=None):
        if isinstance(d, dict):
            if len(d) > 0 and isinstance(next(iter(d.values())), str):  # {gi:str}
                d = {name: loc for loc, name in d.items()}
        else:
            d = {name: loc for name, loc in enumerate(d)}
        if fun_alias is not None:  # chrom aliasing
            d = {name: gi(fun_alias(loc.chromosome), loc.start, loc.end, loc.strand) for name, loc in d.items()}
        self._maxitems = len(d)
        # get list of chromosomes
        chromosomes = SortedSet([loc.chromosome for loc in d.values()])
        self.data = {c: dict() for c in chromosomes}  # split by chrom and sort
        for name, loc in d.items():
            self.data[loc.chromosome][name] = loc
        # create refdict
        self.refdict = RefDict({c: max(self.data[c].values()).end for c in self.data.keys()})
        # call super constructor
        super().__init__(self.data, region=region, file_format=None, per_position=False,
                         fun_alias=None,  # already applied! don't set again
                         refdict=self.refdict)

    def to_bed(self, out,
               fun_anno=lambda idx, loc, item: (f"{item}", '.', loc.start - 1, loc.end, "0,0,0", 1, len(loc), 0),
               bed_header=None, disable_progressbar=True, no_header=False, n_col=4
               ):
        super().to_bed(out, fun_anno, bed_header, disable_progressbar, no_header, n_col)

    def __iter__(self) -> Item[gi, object]:
        for chromosome in self.chromosomes:
            for name, self.location in dict(sorted(self.data[chromosome].items(),
                                                   key=lambda item: item[1])).items():
                self._stats['iterated_items', chromosome] += 1
                if self.region is None or self.region.overlaps(self.location):
                    self._stats['yielded_items', chromosome] += 1
                    yield Item(self.location, name)

    def max_items(self):
        return self._maxitems


class TranscriptomeIterator(LocationIterator):
    """
        Iterates over features in a transcriptome object. Yielded items contain the respective feature (location) and
        its transcriptome annotation (data).
        Note that no chromosome aliasing is possible with this iterator as returned features are immutable.

        A transcriptome iterator can be converted to a pandas dataframe via the to_dataframe() method
        which consumes the remaining items. Column names of the dataframe are automatically estimated
        from the transcriptome model. Data columns may be excluded via the 'excluded_columns' parameter;
        by default the reserved 'dna_seq' field is excluded as this field may contain very long sequence strings.
        The default conversion function (fun) extracts values from the iterated location objects (=feature).
        To add custom columns, one needs to provide (i) a list of column names via the 'included_columns' parameter and
        (ii) a custom extraction methof (fun) that return the respective values. Example:

        Examples
        --------
        >>> # advanced usage: convert to pandas dataframe with a custom conversion function
        >>> # here we add a 'feature length' column
        >>> def my_fun(tx, item, fun_col, default_value):
        >>>     return [len(tx) if col == 'feature_len' else tx.get(col, default_value) for col in fun_col] # noqa
        >>> t = Transcriptome(...)
        >>> TranscriptomeIterator(t).to_dataframe(fun=my_fun, included_annotations=['feature_len']).head()

        Parameters
        ----------
        transcriptome : Transcriptome
            The transcriptome object to iterate over
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        feature_types : list
            optional, if set, only features of these types will be iterated

        Notes
        -----
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, transcriptome, region=None, feature_types=None):
        super().__init__(None, region=region,
                         refdict=transcriptome.merged_refdict)
        self.t = transcriptome
        self.feature_types = feature_types

    def max_items(self):
        return len(self.t.anno)

    def to_dataframe(self,
                     fun=lambda loc, item, fun_col, default_value: [loc.get(col, default_value) for col in fun_col],
                     fun_col=None,
                     coord_inc=(0, 0),
                     coord_colnames=('Chromosome', 'Start', 'End', 'Strand'),
                     excluded_columns=('dna_seq',),
                     included_columns=None,
                     dtypes=None,
                     default_value=None,
                     max_items=None,
                     disable_progressbar=True):
        """ Consumes iterator and returns results in a dataframe.
            Start/stop Coordinates will be corrected by the passed coord_inc tuple.

            fun_col is a tuple of column names for the created dataframe.
            fun is a function that converts the yielded items of this iterator into a tuple of
            values that represent fun_col column values in the created dataframe.

            Example
            -------
            >>> t = Transcriptome(...)
            >>> TranscriptomeIterator(t).to_dataframe().head()


        """
        # mandatory fields; we use a dict to keep column order nice
        fun_col = {'feature_id': None, 'feature_type': None}
        # add all annotation keys from the anno dict
        fun_col.update(dict.fromkeys(get_unique_keys(self.t.anno), None))
        # add annotation fields parsed from GFF
        fun_col.update(dict.fromkeys(get_unique_keys(self.t._ft2anno_class), None))  # noqa
        # preserve order
        fun_col = tuple(fun_col.keys())
        # call super method
        return super().to_dataframe(fun=fun,
                                    fun_col=fun_col,
                                    coord_inc=coord_inc,
                                    coord_colnames=coord_colnames,
                                    dtypes=dtypes,
                                    excluded_columns=excluded_columns,
                                    included_columns=included_columns,
                                    default_value=default_value,
                                    max_items=max_items,
                                    disable_progressbar=disable_progressbar)

    def describe(self,
                 fun=lambda loc, item, fun_col, default_value: [loc.get(col, default_value) for col in fun_col],
                 fun_col=('Value',),
                 coord_inc=(0, 0),
                 coord_colnames=('Chromosome', 'Start', 'End', 'Strand'),
                 excluded_columns=('dna_seq',),
                 included_columns=None,
                 dtypes=None,
                 default_value=None,
                 max_items=None,
                 disable_progressbar=True
                 ) -> Tuple[pd.DataFrame, dict]:
        # call super method
        return super().describe(fun=fun,
                                fun_col=fun_col,
                                coord_inc=coord_inc,
                                coord_colnames=coord_colnames,
                                dtypes=dtypes,
                                excluded_columns=excluded_columns,
                                included_columns=included_columns,
                                default_value=default_value,
                                max_items=max_items,
                                disable_progressbar=disable_progressbar)

    def __iter__(self) -> Item:
        for f in self.t.anno.keys():
            try:
                if (not self.feature_types) or (f.feature_type in self.feature_types):
                    self._stats['iterated_items', f.chromosome] += 1
                    # filter by genomic region
                    if (self.region is not None) and (not f.overlaps(self.region)):
                        continue
                    self._stats['yielded_items', f.chromosome] += 1
                    yield Item(f, self.t.anno[f])
            except AttributeError:
                # this happens if the user put non-features in the anno dict
                pass


class FastaIterator(LocationIterator):
    """
    Iterates over a FASTA file yielding sequence strings and keeps track of the covered genomic location.

    The yielded sequence of length <width> will be padded with <fillvalue> characters if padding is True.
    This generator will yield every step_size window, for a tiling window approach set step_size = width.
    FASTA files will be automatically indexed if no existing index file is found and will be read in chunked mode.

    Parameters
    ----------
    fasta_file : str or pysam FastaFile
        A file path or an open FastaFile object. In the latter case, the object will not be closed
    region : gi or str
        optional, if set, only features overlapping with this region will be iterated
    width : int
        sequence window size
    step : int
        increment for each yield
    file_format : str
        optional, will be determined from filename if omitted
    chunk_size : int
        optional, chunk size for reading fasta files. default: 1024
    fill_value : str
        optional, fill value for padding. default: 'N'
    padding : bool
        optional, if True, the sequence will be padded with fill_value characters. default: False
    fun_alias : Callable
        optional, if set, the iterator will use this function for aliasing chromosome names

    Yields
    ------
    sequence: str
        The extracted sequence in including sequence context.
        The core sequence w/o context can be accessed via seq[context_size:context_size+width].

    Notes
    -----
    * reported stats:
        iterated_items, chromosome: (int, str)
            Number of iterated and yielded items
    """

    def __init__(self, fasta_file, region=None, width=1, step=1,
                 file_format=None, chunk_size: int = 1024,
                 fill_value='N', padding=False, fun_alias=None):
        super().__init__(fasta_file, region, file_format, per_position=True,
                         fun_alias=fun_alias)
        self.width = 1 if width is None else width
        self.step = step
        self.fill_value = fill_value
        self.padding = padding
        self.chunk_size = chunk_size

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def iterate_data(self, chromosome):
        """
            Reads pysam data in chunks and yields individual data items
        """
        start_chunk = max(0, self.region.start - 1)  # 0-based coordinates in pysam!
        while True:
            end_chunk = min(self.region.end, start_chunk + self.chunk_size)
            if end_chunk <= start_chunk:
                break
            chunk = self.file.fetch(reference=chromosome, start=start_chunk, end=end_chunk)
            if (chunk is None) or (len(chunk) == 0):  # we are done
                break
            start_chunk += len(chunk)
            for d in chunk:
                yield d

    def __iter__(self) -> Item:
        padding = self.fill_value * (self.width // 2) if self.padding else ""
        for chromosome in self.chromosomes:
            pos1 = max(1, self.region.start)  # 0-based coordinates in pysam!
            pos1 -= len(padding)
            for dat in windowed(chain(padding, self.iterate_data(chromosome), padding),
                                fillvalue=self.fill_value,
                                n=self.width,
                                step=self.step):
                if isinstance(dat, tuple):
                    dat = ''.join(dat)
                chromosome = self.refdict.alias(chromosome)  # chrom aliasing
                end_loc = pos1 + len(dat) - 1
                self.location = gi(chromosome, pos1, end_loc)
                self._stats['iterated_items', chromosome] += 1
                yield Item(self.location, dat)
                pos1 += self.step


class TabixIterator(LocationIterator):
    """ Iterates over a bgzipped + tabix-indexed file and returns location/tuple pairs.
        Genomic locations will be parsed from the columns with given pos_indices and interval coordinates will be
        converted to 1-based inclusive coordinates by adding values from the configured coord_inc tuple to start and end
        coordinates. Note that this class serves as super-class for various file format specific anno_its (e.g.,
        BedIterator, VcfIterator, etc.) which use proper coord_inc/pos_index default values.

        Parameters
        ----------
        tabix_file : str or pysam.TabixFile
            A file path or an open TabixFile object. In the latter case, the object will not be closed
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        per_position : bool
            optional, if True, the iterator will yield per-position items (e.g., FASTA files)
        coord_inc : tuple
            optional, a tuple of values to be added to start and end coordinates
        pos_indices : tuple
            optional, a tuple of column indices for the interval start, end and chromosome
        refdict : RefDict
            optional, if set, the iterator will use the passed reference dict instead of reading it from the file

        Notes
        -----
        * stats:
            iterated_items, chromosome: (int, str)
                Number of iterated/yielded items
        * TODO
            - add slop
            - improve docs
    """

    def __init__(self, tabix_file, region=None, fun_alias=None, per_position=False,
                 coord_inc=(0, 0), pos_indices=(0, 1, 2), refdict=None):
        super().__init__(file=tabix_file, region=region,
                         file_format='tsv', per_position=per_position, fun_alias=fun_alias,  # e.g., toggle_chr
                         refdict=refdict)
        self.coord_inc = coord_inc
        self.pos_indices = pos_indices

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def __iter__(self) -> Item:
        for chromosome in self.chromosomes:
            if chromosome not in self.file.contigs:
                self.stats['empty_chromosomes'] += 1
                continue
            for row in self.file.fetch(reference=chromosome,
                                       start=(self.region.start - 1) if (self.region.start > 0) else None,
                                       # 0-based coordinates in pysam!
                                       end=self.region.end if (self.region.end < MAX_INT) else None,
                                       parser=pysam.asTuple()):  # @UndefinedVariable
                chromosome = self.refdict.alias(row[self.pos_indices[0]])
                start = int(row[self.pos_indices[1]]) + self.coord_inc[0]
                end = int(row[self.pos_indices[2]]) + self.coord_inc[1]
                self.location = gi(chromosome, start, end)
                self._stats['iterated_items', chromosome] += 1
                yield Item(self.location, tuple(row))


class BedGraphIterator(TabixIterator):
    """
        Iterates a bgzipped and indexed bedgraph file and yields float values
        If a strand is passed, all yielded intervals will have this strand assigned.

        Parameters
        ----------
        bedgraph_file : str or pysam.TabixFile
            A file path or an open TabixFile object. In the latter case, the object will not be closed
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        strand : str
            optional, if set, all yielded intervals will have this strand assigned
        refdict : RefDict
            optional, if set, the iterator will use the passed reference dict instead of reading it from the file

        Notes
        -----
        * reported stats:
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, bedgraph_file, region=None, fun_alias=None, strand=None, refdict=None):
        super().__init__(tabix_file=bedgraph_file, region=region, per_position=False, fun_alias=fun_alias,
                         coord_inc=(1, 0), pos_indices=(0, 1, 2), refdict=refdict)
        self.strand = strand

    def __iter__(self) -> Item:
        for loc, t in super().__iter__():
            self._stats['yielded_items', loc.chromosome] += 1
            if self.strand:
                loc = loc.get_stranded(self.strand)
            yield Item(loc, float(t[3]))


@dataclass
class BedRecord:
    """
        Parsed and mutable version of pysam.BedProxy
        See https://genome.ucsc.edu/FAQ/FAQformat.html#format1 for BED format details
    """

    name: str
    score: int
    location: gi
    thick_start: int
    thick_end: int
    item_rgb: str
    block_count: int
    block_sizes: List[int]
    block_starts: List[int]

    def __init__(self, tup, refdict=None):
        super().__init__()
        self.name = tup[3] if len(tup) >= 4 else None
        self.score = int(tup[4]) if len(tup) >= 5 and tup[4] != '.' else None
        strand = tup[5] if len(tup) >= 6 else None
        chromosome = tup[0] if refdict is None else refdict.alias(tup[0])
        self.location = gi(chromosome, int(tup[1]) + 1, int(tup[2]), strand)  # convert -based to 1-based start
        self.thick_start = int(tup[6]) + 1 if len(tup) >= 7 else None
        self.thick_end = int(tup[7]) if len(tup) >= 8 else None
        self.item_rgb = tup[8] if len(tup) >= 9 else None
        self.block_count = int(tup[9]) if len(tup) >= 10 else None
        self.block_sizes = [int(x) for x in tup[10].split(',') if x != ''] if len(tup) >= 11 else None
        self.block_starts = [int(x) for x in tup[11].split(',') if x != ''] if len(tup) >= 12 else None

    def __repr__(self):
        return f"{self.location.chromosome}:{self.location.start}-{self.location.end} ({self.name})"

    def __len__(self):
        """ Reports the length of the wrapped location, not the current tuple"""
        return self.location.__len__()


class BedIterator(TabixIterator):
    """
        Iterates a BED file and yields 1-based coordinates and pysam BedProxy objects
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asBed

        Parameters
        ----------
        bed_file : str or pysam.TabixFile
            A file path or an open TabixFile object. In the latter case, the object will not be closed
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names


        Notes
        -----
        * NOTE that empty intervals (i.e., start==stop coordinate) will not be iterated.
        * reported stats:
                yielded_items, chromosome: (int, str)
                    Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, bed_file, region=None, fun_alias=None):
        assert guess_file_format(
            bed_file) == 'bed', f"expected BED file but guessed file format is {guess_file_format(bed_file)}"
        super().__init__(tabix_file=bed_file, region=region, per_position=False, fun_alias=fun_alias, coord_inc=(1, 0),
                         pos_indices=(0, 1, 2))

    def __iter__(self) -> Item[gi, BedRecord]:
        for chromosome in self.chromosomes:
            if chromosome not in self.file.contigs:
                self.stats['empty_chromosomes'] += 1
                continue
            for bed in self.file.fetch(reference=chromosome,
                                       start=(self.region.start - 1) if (self.region.start > 0) else None,
                                       # 0-based coordinates in pysam!
                                       end=self.region.end if (self.region.end < MAX_INT) else None,
                                       parser=pysam.asTuple()):  # @UndefinedVariable
                rec = BedRecord(tuple(bed), refdict=self.refdict)  # parse bed record
                self.location = rec.location
                self._stats['yielded_items', chromosome] += 1
                yield Item(rec.location, rec)


@dataclass
class BigBedRecord:
    """
        Parsed BigBed record
        See https://github.com/deeptools/pyBigWig/tree/master for format details
    """

    name: str
    score: float
    location: gi
    thick_start: int
    level: float
    signif: float
    score2: int

    def __init__(self, tup, refdict=None):
        super().__init__()
        self.name = tup[3] if len(tup) >= 4 else None
        self.score = float(tup[4]) if len(tup) >= 5 and tup[4] != '.' else None
        strand = tup[5] if len(tup) >= 6 else None
        chromosome = tup[0] if refdict is None else refdict.alias(tup[0])
        self.location = gi(chromosome, int(tup[1]) + 1, int(tup[2]), strand)  # convert -based to 1-based start
        self.level = float(tup[6]) + 1 if len(tup) >= 7 else None
        self.signif = float(tup[7]) if len(tup) >= 8 else None
        self.score2 = int(tup[8]) if len(tup) >= 9 else None

    def __repr__(self):
        return f"{self.location.chromosome}:{self.location.start}-{self.location.end} ({self.name})"

    def __len__(self):
        """ Reports the length of the wrapped location, not the current tuple"""
        return self.location.__len__()


class BigBedIterator(LocationIterator):
    """ Iterates over a BigBed file via pyBigWig.

    Parameters
    ----------
    file : str, PathLike or pyBigWig
        The pyBigWig file to iterate over
    region : gi or str
        optional, if set, only features overlapping with this region will be iterated
    fun_alias : Callable
        optional, if set, the iterator will use this function for aliasing chromosome names
    """

    def __init__(self, file, region=None, fun_alias=None):
        if isinstance(file, (str, PathLike)):
            assert guess_file_format(
                file) == 'bigbed', f"expected BigBed file but guessed file format is {guess_file_format(file)}"
        super().__init__(file=file, region=region, fun_alias=fun_alias,
                         per_position=False)
        assert self.file.isBigBed() == 1, f"Wrong file format. Is this actually a BigWig file?"

    def __iter__(self) -> Item[gi, float]:
        for chromosome in self.chromosomes:
            # set start to 1 if it is zero (i.e. unbounded))
            start = self.region.start if self.region.start > 0 else 1
            # if end is not set, the locationIterator will use MAXINT as end but this does not work with BigWig files.
            # So here, use max chromosome length if end is not set
            end = self.region.end if self.region.end < MAX_INT else self.refdict[chromosome]
            # use pyBigWig's entries fetcher
            for s, e, bed_str in self.file.entries(chromosome, start - 1, end):
                bed_tup = (self.refdict.alias(chromosome), s + 1, e) + tuple(bed_str.split('\t'))
                rec = BigBedRecord(bed_tup, refdict=self.refdict)  # parse bigbed record
                self.location = rec.location
                self._stats['iterated_items', self.location.chromosome] += 1
                yield Item(self.location, rec)

    def header(self):
        """ Returns the header of the underlying BigBed file."""
        return self.file.header()


class BigWigIterator(LocationIterator):
    """ Iterates over a pyBigWig object.
    If per_position is True, the iterator will yield per-position values, otherwise it will yield interval values.

    Parameters
    ----------
    file : str, PathLike or pyBigWig
        The pyBigWig file to iterate over
    region : gi or str
        optional, if set, only features overlapping with this region will be iterated
    fun_alias : Callable
        optional, if set, the iterator will use this function for aliasing chromosome names
    strand : str
        optional, if set, all yielded intervals will have this strand assigned
    per_position : bool
        optional, if True, the iterator will yield per-position values, otherwise it will yield interval values


    Notes
    -----
    Note that you can also access remote files, but this is slow and not recommended unless files are small.
    Example:
    >>> # iterate a remote bigwig file and report intervals. This file is ~100M
    >>> import itertools
    >>> uri = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeCrgMapabilityAlign100mer.bigWig'
    >>> for loc, val in itertools.islice(rna.it(uri),0,10):
    >>>     display(f"{loc}: {val}")
    """

    def __init__(self, file, region=None, fun_alias=None, per_position=False, strand=None):
        if isinstance(file, (str, PathLike)):
            assert guess_file_format(
                file) == 'bigwig', f"expected BigWig file but guessed file format is {guess_file_format(file)}"
        super().__init__(file=file, region=region, fun_alias=fun_alias,
                         per_position=per_position)
        self.strand = strand
        assert self.file.isBigWig() == 1, f"Wrong file format. Is this actually a BigBed file?"

    def __iter__(self) -> Item[gi, float]:
        for chromosome in self.chromosomes:
            # set start to 1 if it is zero (i.e. unbounded))
            start = self.region.start if self.region.start > 0 else 1
            # if end is not set, the locationIterator will use MAXINT as end but this does not work with BigWig files.
            # So here, use max chromosome length if end is not set
            end = self.region.end if self.region.end < MAX_INT else self.refdict[chromosome]
            if self.per_position:
                # use pyBigWig's values fetcher
                for off, value in enumerate(self.file.values(chromosome, start - 1, end, numpy=True)):
                    self.location = gi(self.refdict.alias(chromosome), start + off, start + off, strand=self.strand)
                    self._stats['iterated_items', chromosome] += 1
                    if math.isnan(value):
                        value = None
                    yield Item(self.location, value)
            else:
                # use pyBigWig's interval fetcher
                for s, e, value in self.file.intervals(chromosome, start - 1, end):
                    self.location = gi(self.refdict.alias(chromosome), s + 1, e, strand=self.strand)
                    self._stats['iterated_items', self.location.chromosome] += 1
                    if math.isnan(value):
                        value = None
                    yield Item(self.location, value)

    def header(self):
        """ Returns the header of the underlying BigWig file."""
        return self.file.header()


@dataclass
class VcfRecord:
    """
        Parsed version of `pysam VCFProxy`, no type conversions for performance reasons.

        Attributes
        ----------
        location: gi
            genomic interval representing this record
        pos: int
            1-based genomic (start) position. For deletions, this is the first deleted genomic position
        is_indel: bool
            True if this is an INDEL
        ref/alt: str
            reference/alternate allele string
        qual: float
            variant call quality
        info: dict
            dict of info fields/values
        genotype (per-sample) dicts: for each FORMAT field (including 'GT'), a {sample_id: value} dict will be created
        zyg: dict
            zygosity information per sample. Created by mapping genotypes to zygosity values using gt2zyg()
            (0=nocall, 1=heterozygous call, 2=homozygous call).
        n_calls: int
            number of called alleles (among all considered samples)

        Notes
        -----
        @see https://samtools.github.io/hts-specs/VCFv4.2.pdf
    """
    location: gi
    pos: int
    id: str
    ref: str
    alt: str
    qual: float
    info: dict
    format: List[str]

    def __init__(self, pysam_var, samples, sample_indices, refdict):
        def parse_info(info):
            if info == '.':
                return None
            ret = {}
            for s in info.split(';'):
                s = s.strip().split('=')
                if len(s) == 1:
                    ret[s[0]] = True
                elif len(s) == 2:
                    ret[s[0]] = s[1]
            return ret

        if (len(pysam_var.ref) == 1) and (len(pysam_var.alt) == 1):
            self.is_indel = False
            start, end = pysam_var.pos + 1, pysam_var.pos + 1  # 0-based in pysam
        else:  # INDEL
            self.is_indel = True
            start, end = pysam_var.pos + 2, pysam_var.pos + len(pysam_var.alt)  # 0-based in pysam
        self.pos = start
        self.location = gi(refdict.alias(pysam_var.contig), start, end, None)  # noinspection PyTypeChecker
        self.id = pysam_var.id if pysam_var.id != '.' else None
        self.ref = pysam_var.ref
        self.alt = pysam_var.alt
        self.qual = pysam_var.qual if pysam_var.qual != '.' else None
        self.info = parse_info(pysam_var.info)
        self.format = pysam_var.format.split(":")
        for col, x in enumerate(self.format):  # make faster
            self.__dict__[x] = {k: v for k, v in zip([samples[i] for i in sample_indices],
                                                     [pysam_var[i].split(':')[col] for i in sample_indices])}
        # calc zygosity per call
        if 'GT' in self.__dict__:
            zyg, calls = zip(*map(gt2zyg, self.__dict__['GT'].values()))
            self.zyg = {k: v for k, v in
                        zip(self.__dict__['GT'].keys(), zyg)}  # sample: <0,1,2> (0=nocall, 1=het, 2=hom)
            self.n_calls = sum(calls)

    def __repr__(self):
        return f"{self.location.chromosome}:{self.pos}{self.ref}>{self.alt}"


class VcfIterator(TabixIterator):
    """
        Iterates a VCF file and yields 1-based coordinates and VcfRecord objects (wrapping pysam VcfProxy object)
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asVCF

        Besides coordinate-related filtering, users cam also pass a list of sample ids to be considered (samples).
        The iterator will by default report only records with at least 1 call in one of the configured samples.
        To force reporting of all records, set filter_nocalls=False.

        Parameters
        ----------
        vcf_file : str or pysam.VariantFile
            A file path or an open VariantFile object. In the latter case, the object will not be closed
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        samples : list
            optional, if set, only records with calls in these samples will be reported
        filter_nocalls : bool
            if True (default), the iterator will report only records with at least 1 call in one of the configured
            samples.

        Attributes
        ----------
        header:
            pysam VariantFile header
        allsamples: list
            list of all contained samples
        shownsampleindices:
            indices of all configured samples

        Notes
        -----
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items
            filtered_nocalls, chromosome: (int, str)
                Number of filtered no-calls
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)

    """

    def __init__(self, vcf_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, samples=None, filter_nocalls=True):
        assert guess_file_format(
            vcf_file) == 'vcf', f"expected VCF file but guessed file format is {guess_file_format(vcf_file)}"
        # pass refdict extracted from VCF header, otherwise it is read from tabix index which would contain only the
        # chroms that are contained in the actual file
        super().__init__(tabix_file=vcf_file, region=region, per_position=True, fun_alias=fun_alias, coord_inc=(0, 0),
                         pos_indices=(0, 1, 1), refdict=RefDict.load(vcf_file, fun_alias))
        # get header
        self.header = pysam.VariantFile(vcf_file).header  # @UndefinedVariable
        self.allsamples = list(self.header.samples)  # list of all samples in this VCF file
        self.shownsampleindices = [i for i, j in enumerate(self.header.samples) if j in samples] if (
                samples is not None) else range(len(self.allsamples))  # list of all sammple indices to be considered
        self.filter_nocalls = filter_nocalls

    def __iter__(self) -> Item[gi, VcfRecord]:
        for chromosome in self.chromosomes:
            if chromosome not in self.file.contigs:
                self.stats['empty_chromosomes'] += 1
                continue
            for pysam_var in self.file.fetch(reference=chromosome,
                                             start=(self.region.start - 1) if (self.region.start > 0) else None,
                                             # 0-based coordinates in pysam!
                                             end=self.region.end if (self.region.end < MAX_INT) else None,
                                             parser=pysam.asVCF()):  # @UndefinedVariable
                rec = VcfRecord(pysam_var, self.allsamples, self.shownsampleindices, self.refdict)
                self.location = rec.location
                chromosome = self.location.chromosome
                self._stats['iterated_items', chromosome] += 1
                if ('n_calls' in rec.__dict__) and self.filter_nocalls and (rec.n_calls == 0):
                    self._stats['filtered_nocalls', chromosome] += 1  # filter no-calls
                    continue
                self._stats['yielded_items', chromosome] += 1
                yield Item(self.location, rec)


class GFF3Iterator(TabixIterator):
    """
        Iterates a GTF/GFF3 file and yields 1-based coordinates and dicts containing key/value pairs parsed from
        the respective info sections. The feature_type, source, score and phase fields from the GFF/GTF entries are
        copied to this dict (NOTE: attribute fields with the same name will be overloaded).
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asGTF

        This iterator is used to build a transcriptome object from a GFF3/GTF file.

        Parameters
        ----------
        gtf_file : str or pysam.TabixFile
            A file path or an open TabixFile object. In the latter case, the object will not be closed
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names

        Notes
        -----
        * reported stats:
            yielded_items, chromosome: (int, str)
                Number of iterated/yielded items
    """

    def __init__(self, gtf_file, region=None, fun_alias=None):
        super().__init__(tabix_file=gtf_file, region=region, per_position=False, fun_alias=fun_alias, coord_inc=(0, 0),
                         pos_indices=(0, 1, 2))
        self.file_format = guess_file_format(gtf_file)
        assert self.file_format in ['gtf', 'gff'], \
            f"expected GFF3/GTF file but guessed file format is {self.file_format}"

    def __iter__(self) -> Item[gi, dict]:
        for chromosome in self.chromosomes:
            if chromosome not in self.file.contigs:
                self.stats['empty_chromosomes'] += 1
                continue
            for row in self.file.fetch(reference=chromosome,
                                       start=(self.region.start - 1) if (self.region.start > 0) else None,
                                       # 0-based coordinates in pysam!
                                       end=self.region.end if (self.region.end < MAX_INT) else None,
                                       parser=pysam.asTuple()):  # @UndefinedVariable
                chromosome, source, feature_type, start, end, score, strand, phase, info = row
                self.location = gi(self.refdict.alias(chromosome), int(start) + self.coord_inc[0],
                                   int(end) + self.coord_inc[1], strand)
                info = parse_gff_attributes(info, self.file_format)
                info['feature_type'] = None if feature_type == '.' else feature_type
                info['source'] = None if source == '.' else source
                info['score'] = None if score == '.' else float(score)
                info['phase'] = None if phase == '.' else int(phase)
                self._stats['yielded_items', chromosome] += 1
                yield Item(self.location, info)


class PandasIterator(LocationIterator):
    """ Iterates over a pandas dataframe that contains three columns with chromosome/start/end coordinates.
        Notable subclasses are PyrangesIterator and BioframeIterator.
        If the passed dataframe is not coordinate sorted, is_sorted must be set to False.

        Parameters
        ----------
        df : dataframe
            pandas datafram with at least 4 columns names as in coord_columns and feature parameter. This dataframe
            will be sorted by chromosome and start values unless is_sorted is set to True
        feature : str
            Name of column to yield. If null, the whole row will be yielded
        region : GI or str
            optional, if set, only features overlapping with this region will be iterated
        coord_columns : list
            Names of coordinate columns, default: ['Chromosome', 'Start', 'End', 'Strand']
        is_sorted : bool
            optional, if set, the iterator will assume that the dataframe is already sorted by chromosome and start
        coord_off : list
            Coordinate offsets, default: (1, 0). These offsets will be added to the  read start/end coordinates to
            convert to the rnalib convention.
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        calc_chromlen : bool
            optional, if set, the iterator will calculate the maximum end coordinate for each chromosome and use this
            information to create a reference dictionary
        refdict : RefDict
            optional, if set, the iterator will use the passed reference dict instead of estimating from the data


        Yields
        ------
        location: gi
            Location object describing the current coordinates
        value: any
            The extracted feature value

        Notes
        -----
        * Note, that iteration over dataframes is generally discouraged as there are much more efficient methods for
          data manipulation.
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)

        Examples
        --------
        Here is another (efficient) way to use a pandas iterator:

        >>> gencode_gff = ... # a gff file
        >>> with BioframeIterator(gencode_gff, chromosome='chr2') as it: # create a filtered DF from the passed gff
        >>>     it.df = it.df.query("strand=='-' ") # further filter the dataframe with pandas
        >>>     print(Counter(it.df['feature'])) # count minus strand features
        >>>     # you can also use this code to then iterate the data which may be convenient/readable if the
        >>>     # dataframe is small:
        >>>     for tx, row in it: # now iterate with rnalib iterator
        >>>         # do something with location and pandas data row
    """

    def __init__(self, df, feature: str = None, region: GI = None,
                 coord_columns: tuple = ('Chromosome', 'Start', 'End', 'Strand'),
                 is_sorted=False, coord_off=(0, 0), fun_alias: Callable = None,
                 calc_chromlen=False, refdict=None):
        self._stats = Counter()
        self.location = None
        self.feature = feature
        self.coord_columns = coord_columns
        self.coord_off = coord_off
        # get dataframe and sort if not sorted.
        self.df = df if is_sorted else df.sort_values(
            [coord_columns[0], coord_columns[1], coord_columns[2]]).reset_index(drop=True)
        # add strand column if not included
        if coord_columns[3] not in self.df.columns:
            self.df[coord_columns[3]] = '.'
        # apply chrom aliasing if any
        if fun_alias is not None:
            self.df[coord_columns[0]] = self.df[coord_columns[0]].apply(fun_alias)
        self.chromosomes = list(dict.fromkeys(self.df[coord_columns[0]]))  # unique set with preserved order
        self.refdict = refdict
        if self.refdict is None:
            if calc_chromlen:
                # group by chrom, calc max end coordinate and create refdict
                self.refdict = RefDict(self.df.groupby(coord_columns[0])[coord_columns[2]].max().to_dict())
            else:
                # refdict w/o chrom lengths
                self.refdict = RefDict({c: None for c in self.chromosomes})
        super().__init__(file=None, region=region, per_position=False,
                         fun_alias=None,  # already applied! don't set again
                         refdict=self.refdict)  # reference dict exists
        # filter dataframe for region. TODO: check whether coordinate filtering is exact
        if (self.region is not None) and (not self.region.is_unbounded()):
            logging.debug(f"filtering dataframe for region {self.region}")
            filter_query = [] if self.region.chromosome is None else [f"{coord_columns[0]}==@self.region.chromosome"]
            # overlap check: self.region.start <= other.end and other.start <= self.region.end
            filter_query += [] if self.region.end is None else [f"{coord_columns[1]}<=(@self.region.end-@coord_off[0])"]
            filter_query += [] if self.region.start is None else [
                f"{coord_columns[2]}>=(@self.region.start-@coord_off[1])"]
            self.df = self.df.query('&'.join(filter_query))

    def __iter__(self) -> Item[gi, pd.Series]:
        for row in self.df.itertuples():
            chromosome = self.refdict.alias(getattr(row, self.coord_columns[0]))
            # NOTE in df, coordinates are 0-based. Start is included, End is excluded.
            start = getattr(row, self.coord_columns[1]) + self.coord_off[0]
            end = getattr(row, self.coord_columns[2]) + self.coord_off[1]
            strand = getattr(row, self.coord_columns[3], '.')
            self.location = gi(chromosome, start, end, strand=strand)
            self._stats['iterated_items', chromosome] += 1
            if self.region.overlaps(self.location):
                self._stats['yielded_items', chromosome] += 1
                yield Item(self.location, row if self.feature is None else getattr(row, self.feature, None))

    def to_dataframe(self, **kwargs):
        return self.df

    def max_items(self):
        return len(self.df.index)


class BioframeIterator(PandasIterator):
    """
        Iterates over a [bioframe](https://bioframe.readthedocs.io/) dataframe.
        The genomic coordinates of yielded locations are corrected automatically.

        Parameters
        ----------
        df : dataframe
            bioframe dataframe with at least 4 columns names as in coord_columns and feature parameter. This dataframe
            will be sorted by chromosome and start values unless is_sorted is set to True
        feature : str
            Name of column to yield. If null, the whole row will be yielded
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        coord_columns : list
            Names of coordinate columns, default: ['chrom', 'start', 'end', 'strand']
        is_sorted : bool
            optional, if set, the iterator will assume that the dataframe is already sorted by chromosome and start
        coord_off : list
            Coordinate offsets, default: (1, 0). These offsets will be added to the  read start/end coordinates to
            convert to the rnalib convention.
        calc_chromlen : bool
            optional, if set, the iterator will calculate chromosome lengths from the data (if required)
        schema : str
            optional, if set, it is passed to by bioframe.read_table to parse the dataframe
        fun_alias: Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        refdict: RefDict
            optional, if set, the iterator will use this reference dictionary for chromosome aliasing

    """

    def __init__(self, df, feature: str = None, region: GI = None, is_sorted=False, fun_alias=None, schema=None,
                 coord_columns: tuple = ('chrom', 'start', 'end', 'strand'), coord_off=(1, 0), calc_chromlen=False,
                 refdict=None):
        if isinstance(df, str):
            # assume a filename and read via bioframe read_table method and make sure that dtypes match
            self.file = df
            schema = guess_file_format(self.file) if schema is None else schema
            schema = 'bed' if schema == 'bedgraph' else schema  # map begraph->bed
            df = bioframe.read_table(self.file, schema=guess_file_format(self.file) if schema is None else schema)
            # filter the 'chrom' column for header lines and replace NaN's
            df = df[~df.chrom.str.startswith('#', na=False)].replace(np.nan, ".")
            # ensure proper dtypes
            df = df.astype({coord_columns[0]: str, coord_columns[1]: int, coord_columns[2]: int})
            if coord_columns[3] in df.columns:
                df[coord_columns[3]] = df[coord_columns[3]].astype(str)
        super().__init__(df if is_sorted else bioframe.sort_bedframe(df),
                         feature=feature, region=region,
                         coord_columns=coord_columns,
                         coord_off=coord_off,  # coord correction
                         is_sorted=True,  # we used bioframe for sorting above.
                         fun_alias=fun_alias,  # chrom aliasing will be applied to df
                         calc_chromlen=calc_chromlen,
                         refdict=refdict)


class PyrangesIterator(PandasIterator):
    """
        Iterates over a [pyranges](https://pyranges.readthedocs.io/) object.
        The genomic coordinates of yielded locations are corrected automatically.

        Parameters
        ----------
        probj : pyranges.PyRanges
            A pyranges object
        feature : str
            Name of column to yield. If null, the whole row will be yielded
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        coord_columns : list
            Names of coordinate columns, default: ['Chromosome', 'Start', 'End', 'Strand']
        is_sorted : bool
            optional, if set, the iterator will assume that the dataframe is already sorted by chromosome and start
        fun_alias: Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        coord_off : list
            Coordinate offsets, default: (1, 0). These offsets will be added to the  read start/end coordinates to
            convert to the rnalib convention.
        calc_chromlen : bool
            optional, if set, the iterator will calculate chromosome lengths from the data (if required)
        refdict: RefDict
            optional, if set, the iterator will use this reference dictionary for chromosome aliasing
    """

    def __init__(self, probj, feature=None, region=None, is_sorted=False, fun_alias=None,
                 coord_columns=('Chromosome', 'Start', 'End', 'Strand'), coord_off=(1, 0), calc_chromlen=False,
                 refdict=None):
        if isinstance(probj, str):
            # assume a filename and read via pyranges read_xxx method and make sure that dtypes match
            self.file = probj
            self.file_format = guess_file_format(self.file)
            if self.file_format == 'bed':
                probj = pyranges.read_bed(self.file)
            elif self.file_format == 'gff':
                probj = pyranges.read_gff(self.file)
            elif self.file_format == 'gtf':
                probj = pyranges.read_gtf(self.file)
            elif self.file_format == 'bam':
                probj = pyranges.read_bam(self.file)
            else:
                raise ValueError(f"Unsupported file format {self.file_format}")
        if not is_sorted:
            probj = probj.sort(['Chromosome', 'Start', 'End'])
        self.probj = probj
        super().__init__(probj.df, feature=feature, region=region, coord_columns=coord_columns, coord_off=coord_off,
                         is_sorted=True, fun_alias=fun_alias, calc_chromlen=calc_chromlen,
                         refdict=refdict)


class PybedtoolsIterator(LocationIterator):
    """ Iterates over a pybedtools BedTool

    Parameters
    ----------
    bedtool : str or pybedtools.BedTool
        A file path or an open BedTool object. In the latter case, the object will not be closed
    region : gi or str
        optional, if set, only features overlapping with this region will be iterated
    fun_alias : Callable
        optional, if set, the iterator will use this function for aliasing chromosome names
    calc_chromlen : bool
        optional, if set, the iterator will calculate chromosome lengths from the file (if required)
    refdict: RefDict
        optional, if set, the iterator will use this reference dictionary for chromosome aliasing


    Notes
    -----
    * reported stats:
        yielded_items, chromosome: (int, str)
            Number of yielded items

    """

    def __init__(self, bedtool, region=None, fun_alias=None, calc_chromlen=False, refdict=None):
        self._stats = Counter()
        # instantiate bedtool
        self.bedtool = bedtool if isinstance(bedtool, pybedtools.BedTool) else pybedtools.BedTool(bedtool)
        self.fun_alias = fun_alias
        self.file = self.bedtool.fn if isinstance(self.bedtool.fn, str) else None
        # get ref dict (via pysam)
        self.refdict = refdict
        if self.refdict is None:
            assert self.file is not None, ("refdict cannot be retrieved automatically (missing file reference). "
                                           "Must be set explicitly via the refict parameter.")
            # try to get ref dict which works only for bgzipped+tabixed files
            try:
                self.refdict = RefDict.load(self.bedtool.fn, fun_alias, calc_chromlen=calc_chromlen)
            except Exception as exp:
                logging.error(f"Could not create refdict, is file bgzipped+tabixed? {exp}")
                raise exp
        print(self.refdict)
        self.set_region(region)
        # region is a gi object. List of iterated chromosomes is set in self.chromosomes
        self.location = None

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def __iter__(self) -> Item[gi, pybedtools.Interval]:  # noqa
        current_bedtool = self.bedtool
        # intersect with region if any
        if not self.region.is_unbounded():
            # intersect with -u to report original intervals
            current_bedtool = self.bedtool.intersect([self.region.to_pybedtools()], u=True)
        for iv in current_bedtool:
            self.location = gi(self.refdict.alias(iv.chrom), iv.start + 1, iv.end, strand=iv.strand)
            self._stats['yielded_items', self.location.chromosome] += 1
            yield Item(self.location, iv)

    def close(self):
        pass


# ---------------------------------------------------------
# SAM/BAM anno_its
# ---------------------------------------------------------

class ReadIterator(LocationIterator):
    """ Iterates over a BAM alignment.

        Parameters
        ----------
        bam_file : str
            BAM file name
        region : gi or str
            optional, if set, only features overlapping with this region will be iterated
        file_format : str
            optional, if set, the iterator will assume this file format (e.g., 'bam' or 'sam')
        min_mapping_quality : int
            Minimum mapping quality. If set, reads with lower mapping quality will be filtered.
        flag_filter : int
            Bitwise flag filter. If set, reads with any of the specified flags will be filtered.
        tag_filters : list
            List of TagFilter objects. If set, reads with any of the specified tags will be filtered.
        max_span : int
            Maximum alignment span. If set, reads with a larger alignment span will be filtered.
        report_mismatches : bool
            If set, this iterator will additionally yield (read, mismatches) tuples where 'mismatches' is a list of
            (read_position, genomic_position, ref_allele, alt_allele) tuples that describe differences wrt. the
            aligned reference sequence. This options requires MD BAM tags to be present.
        min_base_quality : int
            Useful only in combination with report_mismatches; filters mismatches based on minimum per-base quality
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names

        Notes
        -----
        * Reported stats
            iterated_items, chromosome: (int, str)
                Number of iterated items
            yielded_items, chromosome: (int, str)
                Number of yielded items
            n_fil_flag, chromosome:
                Number of reads filtered due to a FLAG filter
            n_fil_mq, chromosome:
                Number of reads filtered due to low mapping quality
            n_fil_tag chromosome:
                Number of reads filtered due to a TAG filter
            n_fil_max_span, chromosome:
                Number of reads filtered due to exceeding a configured maximum alignment span

    """

    def __init__(self, bam_file, region=None, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 report_mismatches=False, min_base_quality=0, fun_alias=None):
        super().__init__(bam_file, region=region, file_format=file_format, per_position=False,
                         fun_alias=fun_alias)
        self.min_mapping_quality = min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters = tag_filters
        self.report_mismatches = report_mismatches
        self.min_base_quality = min_base_quality

    def max_items(self):
        """ Returns number of reads retrieved from the SAM/BAM index"""
        return sum([x.total for x in self.file.get_index_statistics()])

    def __iter__(self) -> Item[gi, pysam.AlignedSegment]:
        md_check = False
        for chromosome in self.chromosomes:
            if chromosome not in self.file.references:
                self.stats['empty_chromosomes'] += 1
                continue
            for r in self.file.fetch(contig=chromosome,
                                     start=self.region.start - 1 if (self.region.start > 0) else None,
                                     end=self.region.end if (self.region.end < MAX_INT) else None,
                                     until_eof=True):
                self.location = gi(self.refdict.alias(r.reference_name), r.reference_start + 1, r.reference_end,
                                   '-' if r.is_reverse else '+')
                self._stats['iterated_items', self.location.chromosome] += 1
                if r.flag & self.flag_filter:  # filter based on BAM flags
                    self._stats['n_fil_flag', self.location.chromosome] += 1
                    continue
                if r.mapping_quality < self.min_mapping_quality:  # filter based on mapping quality
                    self._stats['n_fil_mq', self.location.chromosome] += 1
                    continue
                if self.tag_filters is not None:  # filter based on BAM tags
                    is_filtered = False
                    for tf in self.tag_filters:
                        is_filtered = is_filtered | tf.filter(r)
                    if is_filtered:
                        self._stats['n_fil_tag', self.location.chromosome] += 1
                        continue
                # test max_span and drop reads that span larger genomic regions
                if (self.max_span is not None) and (len(self.location) > self.max_span):
                    self._stats['n_fil_max_span', self.location.chromosome] += 1
                    continue
                self._stats['yielded_items', self.location.chromosome] += 1
                # report mismatches
                if self.report_mismatches:
                    if not md_check:
                        assert r.has_tag("MD"), "BAM does not contain MD tag: cannot report mismatches"
                        md_check = True
                    mm = [(off, pos + 1, ref.upper(), r.query_sequence[off]) for (off, pos, ref) in
                          r.get_aligned_pairs(with_seq=True, matches_only=True) if ref.islower() and
                          r.query_qualities[off] >= self.min_base_quality]  # mask bases with low per-base quailty
                    yield Item(self.location, (r, mm))  # yield read/mismatch tuple
                else:
                    yield Item(self.location, r)


class FastPileupIterator(LocationIterator):
    """
        Fast pileup iterator that yields a complete pileup (w/o insertions) over a set of genomic positions. This is
        more lightweight and considerably faster than `pysam's pileup()` but lacks some features (such as
        `ignore_overlaps` or `ignore_orphans`).
        By default, it basically reports what is seen in the default IGV view (using, e.g., the same read flag filter).

        This iterator uses a ReadIterator to iterate the BAM file and then builds a pileup from the reads that overlap
        the reported positions. The pileup is a dict with genomic positions as keys and a Counter object as values. The
        Counter object contains the counts of each base at the respective position.

        Parameters
        ----------
        bam_file : str
            BAM file name
        chromosome : str
            Chromosome name. If set, only this chromosome will be iterated and reported_positions will be derived from
            the respective parameter. Note, that either chromosome and reported_positions or region must be set.
        reported_positions : range or set
            Range or set of genomic positions for which counts will be reported. The chromosome is derived from the
            respective parameter. Note, that either chromosome and reported_positions or region must be set.
        region : gi
            Genomic region. If set, chromosome and reported_positions will be derived from this region.
            Note, that either chromosome and reported_positions or region must be set.
        file_format : str
            optional, if set, the iterator will assume this file format (e.g., 'bam' or 'sam')
        min_mapping_quality : int
            Filters pileup reads based on minimum mapping quality
        flag_filter : int
            Filters pileup reads based on read flags (see utils.BamFlag for details)
        tag_filters : list
            Filters pileup reads based on BAM tags (see utils.TagFilter for details)
        max_span : int
            Restricts maximum pileup depth.
        min_base_quality : int
            Filters pileup based on minimum per-base quality
        fun_alias : function
            Optional function for aliasing chromosome names

        Returns
        -------
        A <base>:<count> Counter object. Base is 'None' for deletions.

        Notes
        -----
        * initial performance tests that used a synchronized iterator over a FASTA and 2 BAMs showed ~50kpos/sec
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items (reads)
            yielded_items, chromosome: (int, str)
                Number of yielded items (positions)

        Examples
        --------
        >>> pileup_data = rna.it("test.bam", style="pileup", region=gi('chr1:1-100')).to_list()
        >>> rna.it("test.bam", style='pileup',
        >>>        region='1:22418244-22418244', # pileup single position
        >>>        min_mapping_quality=30, flag_filter=0, min_base_quality=40, # read filtering
        >>>        tag_filters=[rna.TagFilter('NM', [0], True, inverse=True)], # include only reads w/o mismatches
        >>>        ).to_list()

    """

    def __init__(self, bam_file, chromosome: str = None, reported_positions: set = None, region: GI = None,
                 file_format: str = None, min_mapping_quality: int = 0, flag_filter: int = DEFAULT_FLAG_FILTER,
                 tag_filters: list[TagFilter] = None, max_span: int = None,
                 min_base_quality: int = 0, fun_alias=None):
        self.file = bam_file
        self.location = None
        self.per_position = False
        self._stats = Counter()
        if chromosome is None:  # get from region parameter
            # assert that region is set
            assert (region is not None) and (reported_positions is None), \
                "Either chromosome and reported_positions or region must be set"
            if not isinstance(region, GI):
                region = gi(region)
            reported_positions = {p.start for p in region}  # a set of positions
            chromosome = region.chromosome
        else:
            if isinstance(reported_positions, range):
                reported_positions = set(reported_positions)
            elif not isinstance(reported_positions, set):
                warnings.warn("reported_positions should be a tuple(start, end) or a set() to avoid slow processing")
        # set iterated region and chromosomes
        self.reported_positions = reported_positions
        self.region = gi(chromosome, min(reported_positions), max(reported_positions))
        self.chromosomes = (chromosome,)
        # pileup counter dict
        self.count_dict = defaultdict(Counter)
        # ReadIterator options
        self.file_format = file_format
        self.min_mapping_quality = min_mapping_quality
        self.flag_filter = flag_filter
        self.tag_filters = tag_filters
        self.max_span = max_span
        self.min_base_quality = min_base_quality
        super().__init__(bam_file, region=self.region,
                         file_format=self.file_format,
                         per_position=True, fun_alias=fun_alias)

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return len(self.reported_positions)

    def __iter__(self) -> Item[gi, Counter]:
        # there is only 1 chromosome, no need to iterate
        with ReadIterator(bam_file=self.file, region=self.region, min_mapping_quality=self.min_mapping_quality,
                          flag_filter=self.flag_filter, tag_filters=self.tag_filters, max_span=self.max_span,
                          min_base_quality=self.min_base_quality, fun_alias=self.fun_alias) as rit:
            for loc, r in rit:
                self._stats['iterated_items', loc.chromosome] += 1
                # find reportable positions, skips soft-clipped bases
                gpos = r.reference_start + 1
                rpos = 0
                for op, l in r.cigartuples:
                    if op in [0, 7, 8]:  # M, =, X
                        for _ in range(l):
                            if gpos in self.reported_positions:
                                if r.query_qualities[rpos] >= self.min_base_quality:  # check base qual
                                    if not self.count_dict[gpos]:
                                        self.count_dict[gpos] = Counter()
                                    self.count_dict[gpos][r.query_sequence[rpos]] += 1
                            rpos += 1
                            gpos += 1
                    elif op == 1:  # I
                        rpos += l
                    elif op == 2:  # D
                        for gpos in range(gpos, gpos + l):
                            if gpos in self.reported_positions:
                                if not self.count_dict[gpos]:
                                    self.count_dict[gpos] = Counter()
                                self.count_dict[gpos][None] += 1
                        gpos += 1
                    elif op == 4:  # S
                        rpos += l
                    elif op == 3:  # N
                        gpos += l
                    elif op in [5, 6]:  # H, P
                        pass
                    else:
                        warnings.warn("unsupported CIGAR op %i" % op)
        # yield all reported positions (including uncovered ones)
        for gpos in self.reported_positions:
            self.location = gi(self.region.chromosome, gpos, gpos)
            self._stats['yielded_items', self.region.chromosome] += 1
            yield Item(self.location, self.count_dict[gpos] if gpos in self.count_dict else Counter())


# ---------------------------------------------------------
# grouped anno_its
# ---------------------------------------------------------


class GroupedLocationIterator(LocationIterator):
    """
        Wraps another location iterator and yields groups of items sharing (parts of) the same location
        given a matching strategy  (e.g., same start, same end, same coords, overlapping).
        The iterator yields tuples of (merged) group location and a (locations, items) tuple containing lists of
        locations/items yielded from the wrapped iterator.


        Parameters
        ----------
        it : LocationIterator
            The wrapped location iterator
        strategy : str
            The grouping strategy to use: left (start coordinate match), right (end coordinate match), both (complete
            location match; default), overlap (coordinate overlap).
    """

    def __init__(self, it: LocationIterator, strategy='both'):
        self.orgit = it
        self.it = peekable(it)
        assert strategy in ['start', 'end', 'both', 'overlap'], f"Unsupported grouping strategy {strategy}"
        self.strategy = strategy
        self.per_position = self.orgit.per_position

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return self.orgit.max_items()

    def __iter__(self) -> Item[gi, (tuple, tuple)]:
        for loc, value in self.it:
            mloc = loc.copy()
            values = [value]
            locations = [loc]
            if self.strategy == 'start':
                while self.it.peek(None) and self.it.peek()[0].left_match(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = GI.merge((mloc, loc))
            elif self.strategy == 'end':
                while self.it.peek(None) and self.it.peek()[0].right_match(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = GI.merge((mloc, loc))
            elif self.strategy == 'both':
                while self.it.peek(None) and self.it.peek()[0] == mloc:
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = GI.merge((mloc, loc))
            elif self.strategy == 'overlap':
                while self.it.peek(None) and self.it.peek()[0].overlaps(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = GI.merge((mloc, loc))
            yield Item(mloc, (locations, values))

    def close(self):
        try:
            self.orgit.close()
        except AttributeError:
            pass


@DeprecationWarning
class SyncPerPositionIterator(LocationIterator):
    """ Synchronizes the passed location anno_its by genomic location and yields
        individual genomic positions and overlapping intervals per passed iterator.
        Expects (coordinate-sorted) location anno_its.
        The chromosome order will be determined from a merged refdict or, if not possible,
        by alphanumerical order.

        Examples
        --------
        >>> it1, it2, it3 = ...
        >>> for pos, (i1,i2,i3) in SyncPerPositionIterator([it1, it2, it3]):
        >>>     print(pos,i1,i2,i3)
        >>>     # where i1,...,i3 are lists of tx/data tuples from the passed LocationIterators

        Notes
        -----
        * reported stats
            yielded_items, chromosome: (int, str)
                Number of yielded items (positions)
        * TODOs
            optimize
            documentation and usage scenarios
    """

    def __init__(self, iterables, refdict=None):
        """
        Parameters
        ----------
        iterables : List[LocationIterator]
            a list of location anno_its
        refdict : RefDict
            a reference dict for the passed anno_its. If None, a merged refdict of all anno_its will be used
        """
        self.iterables = iterables
        for it in iterables:
            assert issubclass(type(it), LocationIterator), "Only implemented for LocationIterators"
        self.per_position = True
        self._stats = Counter()
        self.iterators = [peekable(it) for it in iterables]  # make peekable anno_its
        if refdict is None:
            self.refdict = RefDict.merge_and_validate(*[it.refdict for it in iterables])
            if self.refdict is None:
                warnings.warn("Could not determine RefDict from anno_its: using alphanumerical chrom order.")
            else:
                if len(self.refdict) == 0:
                    warnings.warn("RefDict is empty! {self.refdict}")
        else:
            self.refdict = refdict
        self.current = {i: list() for i, _ in enumerate(self.iterators)}
        # get first position if any
        first_positions = SortedList(d[0] for d in [it.peek(default=None) for it in self.iterators] if d is not None)
        if len(first_positions) > 0:
            # defined chrom sort order (fixed list) or create on the fly (sorted set supports addition in iteration)
            self.chroms = SortedSet([first_positions[0].chromosome]) if self.refdict is None else list(
                self.refdict.keys())  # chrom must be in same order as in iterator!
            # get min.max pos
            self.pos, self.maxpos = first_positions[0].start, first_positions[0].end
        else:
            self.chroms = set()  # no data

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def first_pos(self):
        """ Returns the first position of the next item from the anno_its """
        first_positions = SortedList(d[0] for d in [it.peek(default=None) for it in self.iterators] if d is not None)
        if len(first_positions) > 0:
            return first_positions[0]
        return None

    def update(self, chromosome):
        """ Updates the current list of overlapping intervals for the passed chromosome """
        if self.pos is None:
            return
        for i, it in enumerate(self.iterators):
            self.current[i] = [(loc, d) for loc, d in self.current[i] if loc.end >= self.pos]
            while True:
                nxt = it.peek(default=None)
                if nxt is None:
                    break  # exhausted
                loc, dat = nxt
                if loc.chromosome != chromosome:
                    if self.refdict is None:
                        self.chroms.add(loc.chromosome)  # added chr
                    break
                self.maxpos = max(self.maxpos, loc.end)
                if loc.start > self.pos:
                    break
                self.current[i].append(next(it))  # consume interval

    def __iter__(self) -> Item[gi, tuple]:
        """ Yields a tuple of genomic position and a list of overlapping intervals per iterator"""
        for c in self.chroms:
            self.update(c)
            while self.pos <= self.maxpos:
                self._stats['yielded_items', c] += 1
                yield Item(gi(c, self.pos, self.pos), [list(x) for x in self.current.values()])
                self.pos += 1
                self.update(c)
            tmp = self.first_pos()
            if tmp is None:
                break  # exhausted
            self.pos, self.maxpos = tmp.start, tmp.end
            if self.refdict is None:
                self.chroms.add(tmp.chromosome)

    def close(self):
        """ Closes all wrapped anno_its"""
        for it in self.iterables:
            try:
                it.close()
            except AttributeError:
                pass


class AnnotationIterator(LocationIterator):
    """
        Annotates locations in the first iterator with data from the ano_its location anno_its.
        The returned data is a namedtuple with the following fields:
        Item(location=gi, data=Result(anno=dat_from_it, label1=[Item(tx, dat_from_anno_it1)], ..., labeln=[Item(tx,
        dat_from_anno_itn)])

        This enables access to the following data:

        * item.location: gi of the currently annotated location
        * item.data.anno: data of the currently annotated location
        * item.data.<label_n>: list of items from <iterator_n> that overlap the currently annotated position.

        if no labels are provided, the following will be used: it0, it1, ..., itn.

        Parameters
        ----------
        it : LocationIterator
            The main location iterator. The created AnnotationIterator will yield each item from this iterator
            alongside all overlapping items from the configured anno_its anno_its
        anno_its : List[LocationIterator]
            A list of LocationIterators for annotating the main iterator
        labels : List[str]
            A list of labels for storing data from the annotating anno_its
        refdict : RefDict
            A reference dict for the main iterator. If None, the refdict of the main iterator will be used
        disable_progressbar : bool
            If True, disables the tqdm progressbar

        Yields
        ------
        Item(location=gi, data=Result(anno=dat_from_it, label1=[Item(tx, dat_from_anno_it1)], ..., labeln=[Item(tx,
        dat_from_anno_itn)])


    """

    def __init__(self, it, anno_its, labels=None, refdict=None, disable_progressbar=False):
        """
        Parameters
        ----------
            it: the main location iterator. The created AnnotationIterator will yield each item from this iterator
                alongside all overlapping items from the configured anno_its anno_its
            anno_its: a list of location anno_its for annotating the main iterator
            labels: a list of labels for storing data from the annotating anno_its
            refdict: a reference dict for the main iterator. If None, the refdict of the main iterator will be used
            disable_progressbar: if True, disables the tqdm progressbar
        """
        if not isinstance(anno_its, list):
            anno_its = [anno_its]
        if labels is None:
            labels = [f'it{i}' for i in range(len(anno_its))]
        elif not isinstance(labels, list):
            labels = [labels]
        for x in [it] + anno_its:
            assert issubclass(type(x),
                              LocationIterator), f"Only implemented for LocationIterators but not for {type(x)}"
        self._stats = Counter()
        self.it = it
        self.refdict = it.refdict if refdict is None else refdict
        self.anno_its = anno_its
        self.region = it.region
        self.chromosomes = it.chromosomes
        self.disable_progressbar = disable_progressbar
        self.Result = namedtuple(typename='Result',
                                 field_names=['anno'] + labels)  # result type
        self.current = None

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return self.it.max_items()

    def stats(self):
        """Return stats of wrapped main iterator"""
        return self.it.stats

    def update(self, ref, anno_its):
        """ Updates the current buffer of overlapping intervals wrt. the passed reference location """

        def coord_overlaps(a, b):
            return a.start <= b.end and b.start <= a.end

        for i, it in enumerate(anno_its):
            self.buffer[i] = [Item(loc, d) for loc, d in self.buffer[i] if loc.end >= ref.start]  # drop left
            while True:
                nxt = it.peek(default=None)
                if nxt is None:
                    break  # exhausted
                loc, dat = nxt
                if loc.start > ref.end:
                    break  # done
                nxt = next(it)  # consume interval
                if loc.end < ref.start:
                    continue  # skip
                self.buffer[i].append(nxt)
            self.current[i] = [Item(loc, d) for loc, d in self.buffer[i] if coord_overlaps(loc, ref)]
        return anno_its

    def __iter__(self) -> Item[gi, tuple]:
        """ Yields a tuple of the current genomic position and a named results tuple"""
        for chromosome in tqdm(self.chromosomes, total=len(self.chromosomes), disable=self.disable_progressbar):
            self.buffer = [list() for i, _ in enumerate(
                self.anno_its)]  # holds sorted intervals that overlap or are > than the currently annotated interval
            self.current = [list() for i, _ in enumerate(self.anno_its)]
            # set chrom of it
            self.it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [it for it in self.anno_its if chromosome in it.refdict]
            if len(anno_its) == 0:
                # warnings.warn(feature"Skipping chromosome {chromosome} as no annotation data found!")
                for loc, dat in self.it:
                    yield Item(loc, self.Result(dat, *self.current))  # noqa # yield empty results
                continue
            for it in anno_its:  # set current chrom
                it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [peekable(it) for it in anno_its]  # make peekable anno_its
            for loc, dat in self.it:
                anno_its = self.update(loc, anno_its)
                yield Item(loc, self.Result(dat, *self.current))  # noqa

    def close(self):
        """ Closes all wrapped anno_its"""
        for it in [self.it] + self.anno_its:
            try:
                it.close()
            except AttributeError:
                pass


class TiledIterator(LocationIterator):
    """ Wraps a location iterator and reports tuples of items yielded for each genomic tile.
        Tiles are either iterated from the passed regions_iterable or calculated from the reference dict via the
        RefDict.tile(tile_size=...) method.

        The iterator yields the tile location and a tuple of overlapping items from the location iterator.
        During iteration, the locations and data of the last yielded item can be access via it.tile_locations and
        it.tile_items.

        Parameters
        ----------
        location_iterator: LocationIterator
            The location iterator to be wrapped
        regions_iterable: Iterable[GI]
            An iterable of genomic intervals that define the tiles. If None, the tiles will be calculated from the
            reference dict based on the passed tile_size
        tile_size: int
            The size of the tiles in base pairs
    """

    def __init__(self, location_iterator, regions_iterable: Iterable[GI] = None, tile_size=1e8):
        assert issubclass(type(location_iterator), LocationIterator), \
            f"Only implemented for LocationIterators but not for {type(location_iterator)}"
        self.refdict = location_iterator.refdict
        super().__init__(file=None, region=None, file_format=None, per_position=False, fun_alias=None,
                         refdict= self.refdict)
        self.location_iterator = location_iterator
        self.tile_size = tile_size
        if regions_iterable is None:
            assert self.refdict is not None, "Cannot calculate tiles without a reference dict"
            assert self.refdict.has_len(), ("Cannot calculate tiles from refdict without lengths. "
                                                         "Consider creating a refdict with calc_chromlen=True.")
            self.regions_iterable = self.location_iterator.refdict.tile(tile_size=int(self.tile_size))
        else:
            self.regions_iterable = regions_iterable

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return self.location_iterator.max_items()

    def __iter__(self) -> Item:
        for reg in self.regions_iterable:
            self.location = reg
            self.location_iterator.set_region(reg)
            dat = list(self.location_iterator)
            self.tile_locations, self.tile_items = zip(*dat) if len(dat) > 0 else ((), ())
            self._stats['iterated_items', reg.chromosome] += 1
            yield Item(self.location, self.tile_items)


# ---------------------------------------------------------
# Additional, non location-bound anno_its
# ---------------------------------------------------------

class FastqIterator:
    """
    Iterates a fastq file and yields FastqRead objects (named tuples containing sequence name, sequence and quality
    string (omitting line 3 in the FASTQ file)).

    Parameters
    ----------
    fastq_file : str
        The FASTQ file name

    Notes
    -----
    Note that this is not a location iterator.

    reported stats:

    * yielded_items: int
        Number of yielded items
    """

    def __init__(self, fastq_file):
        self.file = fastq_file
        self.per_position = False
        self._stats = Counter()

    @property
    def stats(self):
        return self._stats

    def __len__(self):
        """ Fast read counting """
        i = 0
        with open_file_obj(self.file, file_format='fastq') as fin:
            for i, _ in enumerate(fin):
                pass
        assert (i + 1) % 4 == 0, "Invalid read_count, not divisible by 4: {i+1}"  # fastq complete?
        return (i + 1) // 4

    def __iter__(self) -> FastqRead:
        """
            Iterates FASTQ entries
        """
        with open_file_obj(self.file, file_format='fastq') as fin:
            for d in grouper(fin, 4, ''):
                self._stats['yielded_items'] += 1
                yield FastqRead(d[0].strip(), d[1].strip(), d[3].strip())

    def to_list(self):
        """ Exhausts iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]


def read_aligns_to_loc(loc: gi, read: pysam.AlignedSegment):
    """ Tests whether a read aligns to the passed location by checking the respective alignment block coordinates and
        the read strand. Note that the chromosome is *not* checked.
    """
    if (loc.strand is not None) and (loc.strand != '-' if read.is_reverse else '+'):
        return False  # wrong strand
    # chr is not checked
    for read_start, read_end in read.get_blocks():
        if (loc.start <= read_end) and (read_start < loc.end):
            return True
    return False


def it(obj, **kwargs):
    """ Factory function for creating LocationIterators for the passed object.
        In most cases, the type of the created iterator will be determined by the type of the passed object.
        If there are multiple possibilities, the style parameter can be used to select the desired iterator type.
        For example, if a pandas data frame is passed then the style=... parameter determines whether a
        PandasIterator, a BioframeIterator or a PyrangesIterator is returned.

        The following object types are supported:

        * If the object is a standard python type (dict, list, tuple), a MemoryIterator will be created.
        * If the object is a file or a string, the file format will be guessed from the file extension.
            * The following file formats are supported:
                * fasta: FastaIterator
                * sam, bam: ReadIterator or FastPileupIterator if you pass style=='pileup'
                * tsv: TabixIterator
                * bed: BedIterator
                * vcf, bcf: VcfIterator
                * gff, gtf: GFF3Iterator
                * fastq: FastaIterator
            * if you pass style=='pybedtools', a PybedtoolsIterator will be created
        * If the object is a pandas DataFrame, a PandasIterator will be created.
        * If you pass style='bioframe', a BioframeIterator will be created
        * If you pass style='pyranges', a PyrangesIterator will be created
        * If the object is a pybedtools.BedTool, a PybedtoolsIterator will be created
        * If the object is a pyBigWig.pyBigWig:
            * if style == 'bigbed', a BigBedIterator will be created
            * else a BigWigIterator will be created
        * If the object is a rnalib.LocationIterator, an AnnotationIterator will be created
        * If the object is a rnalib.Transcriptome, a TranscriptomeIterator will be created


        Parameters
        ----------
        obj: any
            The object for which an iterator should be created
        style: str
            The (optional) style of the iterator for disambiguation. This parameter will be removed from kwargs
            before the iterator is created.
        kwargs: dict
            Additional parameters that will be passed to the iterator constructor

        Returns
        -------
        A LocationIterator

        Examples
        --------
        >>> it('test.bed') # creates a BedIterator
        >>> it('test.bed', style='bedgraph') # creates a BedGraphIterator
        >>> it('test.bam', style='pileup', region='chr1:1000-2000', min_mapping_quality=30) # creates a FastPileupIterator
        >>> it(pd.DataFrame(...), style='bioframe') # creates a BioframeIterator
        >>> it(rna.Transcriptome(...), feature_types='exon') # creates a TranscriptomeIterator
        >>> vars(it("test.bam")) # inspect the created iterator
        """
    style = kwargs.get('style', None)
    if style is not None:
        del kwargs['style']
    if isinstance(obj, dict) or isinstance(obj, list) or isinstance(obj, tuple):
        return MemoryIterator(obj, **kwargs)
    elif isinstance(obj, str) or isinstance(obj, (str, PathLike)):
        ff = guess_file_format(obj)
        assert ff is not None, f'Could not guess file format for file {obj}.'
        if style == 'pybedtools':
            return PybedtoolsIterator(obj, **kwargs)
        if ff == 'fasta':
            return FastaIterator(obj, **kwargs)
        elif ff == 'sam' or ff == 'bam':
            if style == 'pileup':
                return FastPileupIterator(obj, **kwargs)
            return ReadIterator(obj, **kwargs)
        elif ff == 'tsv':
            return TabixIterator(obj, **kwargs)
        elif ff == 'bed':
            if style == 'bedgraph':
                return BedGraphIterator(obj, **kwargs)
            return BedIterator(obj, **kwargs)
        elif ff == 'bedgraph':
            return BedGraphIterator(obj, **kwargs)
        elif ff == 'vcf' or ff == 'bcf':
            return VcfIterator(obj, **kwargs)
        elif ff == 'gff':
            return GFF3Iterator(obj, **kwargs)
        elif ff == 'gtf':
            return GFF3Iterator(obj, **kwargs)
        elif ff == 'fastq':
            return FastqIterator(obj)
        elif ff == 'bigwig':
            return BigWigIterator(obj, **kwargs)
        elif ff == 'bigbed':
            return BigBedIterator(obj, **kwargs)
    elif isinstance(obj, pysam.Fastafile):  # @UndefinedVariable
        return FastaIterator(obj, **kwargs)
    elif isinstance(obj, pysam.AlignmentFile):  # @UndefinedVariable
        if style == 'pileup':
            return FastPileupIterator(obj, **kwargs)
        return ReadIterator(obj, **kwargs)
    elif isinstance(obj, pysam.VariantFile):  # @UndefinedVariable
        return VcfIterator(obj, **kwargs)
    elif isinstance(obj, pd.DataFrame):
        if style == 'bioframe':
            return BioframeIterator(obj, **kwargs)
        elif style == 'pyranges':
            return PyrangesIterator(obj, **kwargs)
        return PandasIterator(obj, **kwargs)
    elif isinstance(obj, pybedtools.BedTool):
        return PybedtoolsIterator(obj, **kwargs)
    elif isinstance(obj, pyBigWig.pyBigWig):
        if style == 'bigbed':
            return BigBedIterator(obj, **kwargs)
        return BigWigIterator(obj, **kwargs)
    elif isinstance(obj, LocationIterator):
        return AnnotationIterator(obj, **kwargs)
    elif isinstance(obj, Transcriptome):
        return TranscriptomeIterator(obj, **kwargs)
    raise NotImplementedError(f'Object type {type(obj)} not supported by factory method. Try to create the iterator '
                              f'manually.')
