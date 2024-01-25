"""
    Rnalib is a python utilities library for handling genomics data.

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
import json
import numbers
from abc import abstractmethod, ABC
from collections import Counter, abc
from dataclasses import dataclass, make_dataclass  # type: ignore # import dataclass to avoid PyCharm warnings
from itertools import chain
from os import PathLike
from typing import List, Callable, NamedTuple, Any, Tuple

import bioframe
import pyranges
import dill
import numpy as np
import pybedtools
from intervaltree import IntervalTree
from more_itertools import pairwise, triplewise, windowed, peekable
from sortedcontainers import SortedList, SortedSet

from ._version import __version__
from .constants import *
from .utils import *
from .testdata import get_resource, list_resources

# location of the test data directory. Use the 'RNALIB_TESTDATA' environment variable or monkey patching to set to your
# favourite location, e.g., rnalib.__RNALIB_TESTDATA__ = "your_path'
__RNALIB_TESTDATA__ = os.environ.get('RNALIB_TESTDATA')


# ------------------------------------------------------------------------
# Genomic Interval (gi) model
# ------------------------------------------------------------------------
@dataclass(frozen=True, init=True)
class gi:  # noqa
    """
        Genomic intervals (gi) in rnalib are inclusive, continuous and 1-based.
        Points are represented by intervals with same start and stop coordinate, empty intervals by passing start>end
        coordinates (e.g., gi('chr1', 1,0).is_empty() -> True).

        GIs are implemented as frozen(immutable) dataclasses and can be used, e.g., as keys in a dict.
        They can be instantiated by passing chrom/start/stop coordinates or can be parsed form a string.

        Intervals can be stranded.
        Using None for each component of the coordinates is allowed to represent unbounded intervals

        Chromosomes group intervals and the order of intervals from different groups (chromosomes) is left undefined.
        To sort also by chromosome, one can use a @ReferenceDict which defined the chromosome order:
        sorted(gis, key=lambda x: (refdict.index(x.chromosome), x))
        Note that the index of chromosome 'None' is always 0

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
        >>> gi('chr1', 1, 10)
        >>> gi('chr1', 1, 10, strand='+')
        >>> gi('chr1', 1, 10, strand='-')
        >>> gi('chr1', 1, 10, strand=None)
        >>> gi('chr1', 1, 10, strand='.')
        >>> gi('chr1', 1, 10, strand='u')

    """
    chromosome: str = None
    start: int = 0  # unbounded, ~-inf
    end: int = MAX_INT  # unbounded, ~+inf
    strand: str = None

    def __post_init__(self):
        """ Some sanity checks and default values """
        object.__setattr__(self, 'start', 0 if self.start is None else self.start)
        object.__setattr__(self, 'end', MAX_INT if self.end is None else self.end)
        object.__setattr__(self, 'strand', self.strand if self.strand != '.' else None)
        if self.start > self.end:  # empty interval, set start/end to 0/-1
            object.__setattr__(self, 'start', 0)
            object.__setattr__(self, 'end', -1)
        assert isinstance(self.start, numbers.Number)
        assert isinstance(self.end, numbers.Number)
        assert self.strand in [None, '+', '-']

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
        return cls(chromosome, int(start), int(end), strand)

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
        """ Deep copy """
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
            Note that this will fail on open or empty intervals as those are not supported by rnalib.

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

    def __iter__(self):
        for pos in range(self.start, self.end + 1):
            yield gi(self.chromosome, pos, pos, self.strand)


"""
    Lists valid sub-feature types (e.g., 'exon', 'CDS') and maps their different string representations in various
    GFF3 flavours to the corresponding sequence ontology term (e.g., '3UTR' -> 'three_prime_UTR').
"""

"""
    List of supported gff flavours and the respective GFF field names.
"""


# ------------------------------------------------------------------------
# Transcriptome model
# ------------------------------------------------------------------------


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
            print(f"Using aliasing function for annotation_gff: {self.annotation_fun_alias}")
        self.copied_fields = {'source', 'gene_type'} if copied_fields is None else \
            set(copied_fields) | {'source', 'gene_type'}  # ensure source and gene_type are copied
        self.load_sequence_data = load_sequence_data
        self.calc_introns = calc_introns
        self.disable_progressbar = disable_progressbar
        self.genome_offsets = {} if genome_offsets is None else genome_offsets
        self.feature_filter = TranscriptFilter() if feature_filter is None else TranscriptFilter(
            feature_filter) if isinstance(feature_filter, dict) else feature_filter
        self.log = Counter()
        self.merged_refdict = None
        self.gene = {}  # gid: gene
        self.transcript = {}  # tid: gene
        self.cached = False  # if true then transcriptome was loaded from a pickled file
        self.has_seq = False  # if true, then gene objects are annotated with the respective genomic (dna) sequences
        self.anno = {}  # a dict that holds annotation data for each feature
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
        rd = [] if self.genome_fa is None else [ReferenceDict.load(open_file_obj(self.genome_fa))]
        rd += [ReferenceDict.load(open_file_obj(self.annotation_gff), fun_alias=self.annotation_fun_alias)]
        self.merged_refdict = ReferenceDict.merge_and_validate(*rd,
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
                                print(f"Skipping {self.annotation_flavour} {self.file_format} line {line_number + 1}"
                                      f" ({info['feature_type']}), info:\n\t{info} as no gene_id found.")
                                continue
                            genes[gid] = _Feature(self, 'gene', gid, loc,
                                                  parent=None, children={'transcript': []})
                            for cf in self.copied_fields:
                                genes[gid].anno[cf] = info.get(cf, None)
                            genes[gid].anno['gene_name'] = norm_gn(info.get(fmt['gene_name'], gid), current_symbols,
                                                                   aliases)  # normalized gene symbol/name
                            genes[gid].anno['gff_feature_type'] = info['feature_type']
                except Exception as exc:
                    print(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line {line_number + 1}, "
                          f"info:\n\t{info}")
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
                                print(f"Skipping {self.annotation_flavour} {self.file_format} line {line_number + 1}"
                                      f" ({info['feature_type']}), info:\n\t{info} as no {fmt['tid']} field found.")
                                continue
                            gid = f'gene_{tid}' if fmt['tx_gid'] is None else info.get(fmt['tx_gid'], None)
                            if gid is None:
                                print(f"Skipping {self.annotation_flavour} {self.file_format} line {line_number + 1}"
                                      f" ({info['feature_type']}), info:\n\t{info} as no {fmt['tx_gid']} field found.")
                                continue
                            if gid in filtered_gene_ids:
                                self.log[f"filtered_{info['feature_type']}_parent_gene_filtered"] += 1
                                continue
                            # create transcript object
                            transcripts[tid] = _Feature(self, 'transcript', tid, loc,
                                                        parent=genes.get(gid, None),
                                                        children={k: [] for k in set(fmt['ftype_to_SO'].values())})
                            for cf in self.copied_fields:
                                transcripts[tid].anno[cf] = info.get(cf, None)
                            transcripts[tid].anno['gff_feature_type'] = info['feature_type']
                            # add missing gene annotation (e.g., ucsc, flybase, chess)
                            if gid not in genes:
                                if gid in missing_genes:
                                    newloc = gi.merge([missing_genes[gid].loc, loc])
                                    if newloc is None:
                                        # special case, e.g., in Chess annotation/tx CHS.40038.9 is annotated on the
                                        # opposite strand. We skip this tx and keep the gene annotation.
                                        # In chess 3.0.1 there are 3 such entries, all pseudo genes.
                                        print(f"WARNING: gene {gid} has tx with incompatible coordinates! "
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
                    print(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line {line_number + 1}, "
                          f"info:\n\t{info}")
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
                    print(f"ERROR parsing {self.annotation_flavour} {it.file_format} at line {line_number + 1}, "
                          f"info:\n\t{info}")
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
        self.anno = {f: {} for f in all_features}
        # assert that parents intervals always envelop their children
        for f in self.anno:
            if f.parent is not None:
                assert f.parent.envelops(
                    f), f"parents intervals must envelop their child intervals: {f.parent}.envelops({f})==False"
        # build some auxiliary dicts
        self.transcripts = [f for f in self.anno.keys() if f.feature_type == 'transcript']
        self.gene = {f.feature_id: f for f in self.anno.keys() if f.feature_type == 'gene'}
        self.gene.update({f.gene_name: f for f in self.anno.keys() if f.feature_type == 'gene'})
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
                # print(start,end, len(prefix), len(self.anno[g]['dna_seq']), len(g))
        self.has_seq = True

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
            query : GenomicInterval
                Query interval
            feature_types : str or List[str]
                Feature types to query. If None, all feature types will be queried.
            envelop : bool
                If true, only features fully contained in the query interval are returned.
            sort : bool
                If true, the returned features will be sorted by chromosome and start coordinate.
        """
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

    def annotate(self, iterators, fun_anno, labels=None, chromosome=None, start=None, end=None, region=None,
                 feature_types=None, disable_progressbar=True):
        """ Annotates all features of the configured type and in the configured genomic region using the passed fun_anno
            function.
            NOTE: consider removing previous annotations with the clear_annotations() functions before (re-)annotating
            a transcriptome.
        """
        with AnnotationIterator(
                TranscriptomeIterator(self, chromosome=chromosome, start=start, end=end, region=region,
                                      feature_types=feature_types),
                iterators, labels) as it:
            for item in (pbar := tqdm(it, disable=disable_progressbar)):
                pbar.set_description(f"buffer_size={[len(x) for x in it.buffer]}")
                fun_anno(item)
        # # which chroms to consider?
        # chroms=self.merged_refdict if chromosome is None else ReferenceDict({chromosome:None})
        # for chrom in chroms:
        #     with AnnotationIterator(
        #             TranscriptomeIterator(self, chromosome=chrom, start=start, end=end, region=region,
        #             feature_types=feature_types, description=chrom  ),
        #             iterators, labels) as it:
        #         for item in it:
        #             fun_anno(item)

    def save(self, out_file):
        """
            Stores this transcriptome and all annotations as dill (pickle) object.
            Note that this can be slow for large-scaled transcriptomes and will produce large ouput files.
            Consider using save_annotations()/load_annotations() to save/load only the annotation dictionary.
        """
        print(f"Storing {self} to {out_file}")
        with open(out_file, 'wb') as out:
            dill.dump(self, out, recurse=True)
            # byref=True cannot vbe used as dynamically created dataclasses are not supported yet

    @classmethod
    def load(cls, in_file):
        """Load transcriptome from pickled file"""
        print(f"Loading transcriptome model from {in_file}")
        import gc
        gc.disable()  # disable garbage collector
        with open(in_file, 'rb') as infile:
            obj = dill.load(infile)
        gc.enable()
        obj.cached = True
        print(f"Loaded {obj}")
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
        print(f"Storing annotations of {self} to {out_file}")
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
        print(f"Loading annotations from {in_file}")
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
                o.strand,
                to_str(o.phase if hasattr(o, 'phase') else None, na='.'),
                to_str([f'{k}={v}' for k, v in data_dict.items()], sep=';')
            ]]), file=out_stream)

        copied_fields = [x for x in self.copied_fields if x not in ['score', 'phase']]
        with open(out_file, 'w') as out:
            with self.iterator(feature_types=feature_types) as it:
                for f, dat in it:
                    if f.feature_type == 'gene':
                        info = {'ID': f.feature_id, 'gene_name': f.gene_name}
                    else:
                        info = {'ID': f.feature_id,
                                'Parent': f.parent.feature_id}
                    info.update({k: getattr(f, k) for k in copied_fields})  # add copied fields
                    write_line(f, f.feature_type, info, out)
        if bgzip:
            bgzip_and_tabix(out_file)
            return out_file + '.gz'
        return out_file

    def __len__(self):
        return len(self.anno)

    def __repr__(self):
        return f"Transcriptome with {len(self.genes)} genes and {len(self.transcripts)} tx" + (
            " (+seq)" if self.has_seq else "") + (" (cached)" if self.cached else "")

    def iterator(self, chromosome=None, start=None, end=None, region=None, feature_types=None):
        """ returns a :class:`.TranscriptomeIterator` for iterating over all features of the passed type(s).
            If feature_types is None (default), all features will be returned.
        """
        return TranscriptomeIterator(self, chromosome=chromosome, start=start, end=end, region=region,
                                     feature_types=feature_types)

    def __iter__(self):
        with self.iterator() as it:
            yield from it

    def get_struct(self):
        """Return a dict mapping feature to child feature types"""
        return self._ft2child_ftype


@dataclass(frozen=True, repr=False)
class Feature(gi):
    """
        A (frozen) genomic feature, e.g., a gene, a transcript, an exon, an intron,
        a CDS or a three_prime_UTR/_prime_UTR. Features are themselves containers of sub-features
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
    specific info fields.

    The filter is configured via a config dict that can be passed to the constructor.
    Alternatively, the config can be built via the include_* methods.

    The config dict has the following structure:
    >>>    {
    >>>     'location': {
    >>>            'included': {
    >>>                'chromosomes': ['chr1', 'chr2', ...],
    >>>                'regions': ['chr1:100-200', ...]
    >>>            },
    >>>            'excluded': {
    >>>                    'chromosomes': ['chr3', ...],
    >>>                    'regions': ['chr5:100-200', ...]
    >>>                }
    >>>        },
    >>>        'gene': {
    >>>            'included': {
    >>>                'gene_id': ['ENSG000001', ...],
    >>>                'gene_type': ['protein_coding', ...]
    >>>                'myannotation': ['value1', 'value2', ...]
    >>>            },
    >>>            'excluded': {
    >>>                'gene_id': ['ENSG000002', ...],
    >>>                'gene_type': ['lincRNA', ...]
    >>>                'myannotation': ['value3', 'value4', ...]
    >>>            }
    >>>        },
    >>>        'transcript': {
    >>>            'included': {
    >>>                'transcript_id': ['ENST000001', ...],
    >>>                'myannotation': ['value1', 'value2', ...]
    >>>            },
    >>>            'excluded': {
    >>>                'transcript_id': ['ENST000002', ...],
    >>>                'myannotation': ['value3', 'value4', ...]
    >>>            }
    >>>        }
    >>>    }

    Features will first be filtered by location, then by included and then by excluded feature type specific
    info fields.

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
            self.included_regions = {gi.from_str(s) for s in self.included_regions}
        if self.excluded_regions is not None:
            self.excluded_regions = {gi.from_str(s) for s in self.excluded_regions}

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
        if not isinstance(chromosomes, set):
            if isinstance(chromosomes, str):
                chromosomes = {chromosomes}
            else:
                chromosomes = set(chromosomes)
        if self.included_chrom is None:
            self.included_chrom = chromosomes
        else:
            self.included_chrom.update(chromosomes)
        self.config['location']['included']['chromosomes'] = list(self.included_chrom)
        return self

    def include_regions(self, regions: set):
        """Convenience method to add regions to the included_regions set"""
        if self.included_regions is None:
            self.included_regions = regions
        else:
            self.included_regions.update(regions)
        self.config['location']['included']['regions'] = list(self.included_regions)
        return self

    def include_gene_ids(self, ids: set):
        """Convenience method to add included gene ids"""
        self.config['gene']['included']['gene_id'] = list(ids)
        return self

    def include_transcript_ids(self, ids: set):
        """Convenience method to add included transcript ids"""
        self.config['transcript']['included']['transcript_id'] = list(ids)
        return self

    def include_gene_types(self, gene_types: set, include_missing=True):
        """Convenience method to add included gene_types to gene+transcript inclusion rules. Use, e.g.,
        {'protein_coding'} to load only protein coding genes. If include_missing is True then genes/transcripts
        without gene_type will also be included. """
        self.config['gene']['included']['gene_type'] = list(gene_types) + [
            None] if include_missing else list(gene_types)
        self.config['transcript']['included']['gene_type'] = list(gene_types) + [
            None] if include_missing else list(gene_types)
        return self

    def include_transcript_types(self, transcript_types: set, include_missing=True):
        """Convenience method to add included transcript_types. Use, e.g., {'miRNA'} to load only
        miRNA transcripts. If include_missing is True (default) then transcripts without transcript_type will
        also be included. """
        self.config['transcript']['included']['transcript_type'] = list(transcript_types) + [
            None] if include_missing else list(transcript_types)
        return self

    def include_tags(self, gene_tags: set, include_missing=True):
        """Convenience method to add included gene tags. Use, e.g., {'Ensembl_canonical'} to load only
            canonical genes. If include_missing is True then genes/transcripts without tags
            will also be included. """
        self.config['gene']['included']['tag'] = list(gene_tags) + [
            None] if include_missing else list(gene_tags)
        self.config['transcript']['included']['tag'] = list(gene_tags) + [
            None] if include_missing else list(gene_tags)
        return self


class ReferenceDict(abc.Mapping[str, int]):
    """
        Named mapping for representing a set of references (contigs) and their lengths.

        Supports aliasing by passing a function (e.g., fun_alias=toggle_chr which will add/remove 'chr' prefixes) to
        easily integrate genomic files that use different (but compatible) reference names. If an aliasing function is
        passed, original reference names are accessible via the orig property. An aliasing function must be reversible,
        i.e., fun_alias(fun_alias(str))==str and support None.

        Note that two reference dicts match if their (aliased) contig dicts match (name of ReferenceDict is not
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

    def iter_blocks(self, block_size=int(1e6)):
        """
            Iterates in an ordered fashion over the reference dict, yielding genomic intervals of the given block_size
            (or smaller at chromosome ends).
        """
        for chrom, chrlen in self.d.items():
            chrom_gi = gi(chrom, 1, chrlen)  # will use maxint if chrlen is None!
            for block in chrom_gi.split_by_maxwidth(block_size):
                yield block

    def __repr__(self):
        return (f"RefSet (size: {len(self.d.keys())}): {self.d.keys()}"
                f"{f' (aliased from {self.orig.keys()})' if self.fun_alias else ''}, {self.d.values()} name:"
                f" {self.name} ")

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
            print(f"{chrom} not in refdict")
            return None

    @staticmethod
    def merge_and_validate(*refsets, check_order=False, included_chrom=None):
        """
            Checks whether the passed reference sets are compatible and returns the
            merged reference set containing the intersection of common references.

            Parameters
            ----------
            refsets:
                list of ReferenceDicts
            check_order:
                if True, the order of the common references is asserted. default: False
            included_chrom:
                if passed, only the passed chromosomes are considered. default: None

            Returns
            -------
            ReferenceDict containing the intersection of common references
        """
        refsets = [r for r in refsets if r is not None]
        if len(refsets) == 0:
            return None
        # intersect all contig lists while preserving order (set.intersection() or np.intersect1d() do not work!)
        shared_ref = {k: None for k in intersect_lists(*[list(r.keys()) for r in refsets], check_order=check_order) if
                      (included_chrom is None) or (k in included_chrom)}
        # check whether contig lengths match
        for r in refsets:
            for contig, oldlen in shared_ref.items():
                newlen = r.get(contig)
                if newlen is None:
                    continue
                if oldlen is None:
                    shared_ref[contig] = newlen
                else:
                    assert oldlen == newlen, (f"Incompatible lengths for contig ({oldlen}!={newlen}) when comparing "
                                              f"RefSets {refsets}")
        return ReferenceDict(shared_ref, name=','.join([r.name if r.name else "<unnamed refdict>" for r in refsets]),
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
                was_opened = True
                fh = open_file_obj(fh)
            if isinstance(fh, pysam.Fastafile):  # @UndefinedVariable
                return ReferenceDict({c: fh.get_reference_length(c) for c in fh.references},
                                     name=f'References from FASTA file {fh.filename}', fun_alias=fun_alias)
            elif isinstance(fh, pysam.AlignmentFile):  # @UndefinedVariable
                return ReferenceDict({c: fh.header.get_reference_length(c) for c in fh.references},
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
                return ReferenceDict(refdict, name=f'References from TABIX file {fh.filename}',
                                     fun_alias=fun_alias)
            elif isinstance(fh, pysam.VariantFile):  # @UndefinedVariable
                return ReferenceDict({c: fh.header.contigs.get(c).length for c in fh.header.contigs},
                                     name=f'References from VCF file {fh.filename}', fun_alias=fun_alias)
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
        datasets. Most rnalib iterables inherit from this suoperclass and yield named tuples containing data and its
        respective genomic location.

        A LocationIterator iterates over a genomic dataset and yields tuples of genomic intervals and
        associated data. The data can be any object (e.g., a string, a dict, a list, etc.). The iterator can be
        restricted to genomic regions (e.g., chromosomes or regions) and/or strand.

        LocationIterators keep track of their current genomic location (self.location) as well as statistics about
        the iterated and yielded items. They can be consumed and converted to lists (self.to_list()), pandas dataframes
        (self.to_dataframe()) or interval trees (self.to_intervaltrees()).

        LocationIterators are iterable and implement the context manager protocol (with statement) and provide a
        standard interface for region-type filtering (by passing chromosome sets or genomic regions of
        interest). Where feasible and not supported by the underlying (pysam) implementation, they implement chunked
        I/O where for efficient iteration (e.g., FastaIterator).

        The maximum number of itereated items can be queried via the max_items() method which tries to guesstimate this
        number from the underlying index data structures (if available).

        Examples
        --------
        >>> with LocationIterator(...) as it:
        >>>    ...


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
        chromosome, start, end, region : coordinates
        region : str
            genomic region to iterate; overrides chromosome/start/end/strand params
        file_format : str
            optional, will be determined from filename if omitted
        chunk_size : int
            optional, chunk size for chunked I/O (e.g., FASTA files)
        per_position : bool
            optional, if True, the iterator will yield per-position items (e.g., FASTA files)
        fun_alias : Callable
            optional, if set, the iterator will use this function for aliasing chromosome names
        refdict : ReferenceDict
            optional, if set, the iterator will use the passed reference dict instead of reading it from the file
        calc_chromlen : bool
            optional, if set, the iterator will calculate chromosome lengths from the file (if required)

        Notes
        -----
        TODOs:

        * strand specific iteration
        * url streaming
        * remove self.chromosome and get from self.location
        * add is_exhausted flag?
    """

    @abstractmethod
    def __init__(self, file,
                 chromosome: str = None,
                 start: int = None,
                 end: int = None,
                 region: gi = None,
                 strand: str = None,
                 file_format: str = None,
                 chunk_size: int = 1024,
                 per_position: bool = False,
                 fun_alias: Callable = None,
                 refdict: ReferenceDict = None,
                 calc_chromlen: bool = False):
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
        self.calc_chromlen = calc_chromlen
        self.refdict = refdict if refdict is not None else ReferenceDict.load(self.file,
                                                                              self.fun_alias,
                                                                              self.calc_chromlen) if self.file else None
        self.chunk_size = chunk_size
        self.strand = strand
        if region is not None:
            self.region = gi.from_str(region) if isinstance(region, str) else region
        else:
            self.region = gi(chromosome, start, end, strand)
        self.chromosome, self.start, self.end = self.region.split_coordinates()
        assert (self.refdict is None) or (self.chromosome is None) or (
                self.chromosome in self.refdict), f"{self.chromosome} not found in references {self.refdict}"
        if self.chromosome is None:
            self.chromosomes = self.refdict.keys() if self.refdict is not None else None  # all chroms in correct order
        else:
            self.chromosomes = [self.chromosome]

    @property
    def stats(self):
        """ Returns the collected stats """
        return self._stats

    def set_region(self, region):
        """ Update the iterated region of this iterator.
            Note that the region's chromosome must be in this iterators refdict (if any)
        """
        self.region = gi.from_str(region) if isinstance(region, str) else region
        self.chromosome, self.start, self.end = self.region.split_coordinates()
        if self.refdict is not None and self.chromosome is not None:
            assert self.chromosome in self.refdict, f"Invalid chromosome {self.chromosome} not in \
            refddict {self.refdict}"

    def to_list(self):
        """ Consumes iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]

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
                                '.' if loc.strand is None else loc.strand] + fun(loc, item, fun_col, None) for
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
        stats = {"contains_overlapping": len(df.index) > len(bioframe.merge(df, cols=('Chromosome', 'Start',
                                                                                      'End')).index),
                 "contains_empty": sum((df["End"] - df["Start"]) < 0) > 0}
        return df.describe(include='all'), stats

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
            # print(feature"Closing iterator {self}")
            self.file.close()


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
        >>> def my_fun(loc, item, fun_col, default_value):
        >>>     return [len(loc) if col == 'feature_len' else loc.get(col, default_value) for col in fun_col] # noqa
        >>> t = Transcriptome(...)
        >>> TranscriptomeIterator(t).to_dataframe(fun=my_fun, included_annotations=['feature_len']).head()

        Parameters
        ----------
        transcriptome : Transcriptome
            The transcriptome object to iterate over
        chromosome, start, end, region : coordinates
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

    def __init__(self, transcriptome, chromosome=None, start=None, end=None, region=None, feature_types=None):
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region,
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
            if (not self.feature_types) or (f.feature_type in self.feature_types):
                self.chromosome = f.chromosome
                self._stats['iterated_items', self.chromosome] += 1
                # filter by genomic region
                if (self.region is not None) and (not f.overlaps(self.region)):
                    continue
                self._stats['yielded_items', self.chromosome] += 1
                yield Item(f, self.t.anno[f])


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
    chromosome, start, end, region : coordinates
    width : int
        sequence window size
    step : int
        increment for each yield
    file_format : str
        optional, will be determined from filename if omitted

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
    * TODO:
        * support chromosome=None
    """

    def __init__(self, fasta_file, chromosome=None, start=None, end=None, region=None, width=1, step=1,
                 file_format=None,
                 fill_value='N', padding=False, fun_alias=None):
        super().__init__(fasta_file, chromosome, start, end, region, file_format, per_position=True,
                         fun_alias=fun_alias, calc_chromlen=False)
        self.width = 1 if width is None else width
        self.step = step
        self.fill_value = fill_value
        self.padding = padding

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def read_chunk(self, chromosome, start, end):
        """ Fetch a chunk from the FASTA file """
        return self.file.fetch(reference=self.refdict.alias(chromosome), start=start, end=end)  # @UndefinedVariable

    def iterate_data(self):
        """
            Reads pysam data in chunks and yields individual data items
        """
        start_chunk = 0 if (self.start is None) else max(0, self.start - 1)  # 0-based coordinates in pysam!
        while True:
            end_chunk = (start_chunk + self.chunk_size) if (self.end is None) else min(self.end,
                                                                                       start_chunk + self.chunk_size)
            if end_chunk <= start_chunk:
                break
            chunk = self.read_chunk(self.chromosome, start_chunk, end_chunk)
            if (chunk is None) or (len(chunk) == 0):  # we are done
                break
            start_chunk += len(chunk)
            for d in chunk:
                yield d

    def __iter__(self) -> Item:
        padding = self.fill_value * (self.width // 2) if self.padding else ""
        pos1 = 1 if (self.start is None) else max(1, self.start)  # 0-based coordinates in pysam!
        pos1 -= len(padding)
        for dat in windowed(chain(padding, self.iterate_data(), padding),
                            fillvalue=self.fill_value,
                            n=self.width,
                            step=self.step):
            if isinstance(dat, tuple):
                dat = ''.join(dat)
            end_loc = pos1 + len(dat) - 1
            self.location = gi(self.chromosome, pos1, end_loc)
            self._stats['iterated_items', self.chromosome] += 1
            yield Item(self.location, dat)
            pos1 += self.step


class TabixIterator(LocationIterator):
    """ Iterates over a bgzipped + tabix-indexed file and returns location/tuple pairs.
        Genomic locations will be parsed from the columns with given pos_indices and interval coordinates will be
        converted to 1-based inclusive coordinates by adding values from the configured coord_inc tuple to start and end
        coordinates. Note that this class serves as super-class for various file format specific iterators (e.g.,
        BedIterator, VcfIterator, etc.) which use proper coord_inc/pos_index default values.


        Notes
        -----
        * stats:
            iterated_items, chromosome: (int, str)
                Number of iterated/yielded items
        * TODO
            - add slop
            - improve docs
    """

    def __init__(self, tabix_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, per_position=False,
                 coord_inc=(0, 0), pos_indices=(0, 1, 2), refdict=None, calc_chromlen=False):
        super().__init__(file=tabix_file, chromosome=chromosome, start=start, end=end, region=region,
                         file_format='tsv', per_position=per_position, fun_alias=fun_alias,
                         refdict=refdict, calc_chromlen=calc_chromlen)  # e.g., toggle_chr
        self.coord_inc = coord_inc
        self.pos_indices = pos_indices

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def __iter__(self) -> Item:
        # we need to check whether chrom exists in tabix contig list!
        chrom = self.refdict.alias(self.chromosome)
        if (chrom is not None) and (chrom not in self.file.contigs):
            # print(feature"{chrom} not in {self.file.contigs}")
            return
        for row in self.file.fetch(reference=chrom,
                                   start=(self.start - 1) if (self.start > 0) else None,
                                   # 0-based coordinates in pysam!
                                   end=self.end if (self.end < MAX_INT) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome = self.refdict.alias(row[self.pos_indices[0]])
            start = int(row[self.pos_indices[1]]) + self.coord_inc[0]
            end = int(row[self.pos_indices[2]]) + self.coord_inc[1]
            self.location = gi(chromosome, start, end)
            self._stats['iterated_items', self.chromosome] += 1
            yield Item(self.location, tuple(row))


class BedGraphIterator(TabixIterator):
    """
        Iterates a bgzipped and indexed bedgraph file and yields float values
        If a strand is passed, all yielded intervals will have this strand assigned.

        Notes
        -----
        * reported stats:
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, bedgraph_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, strand=None, calc_chromlen=False):
        super().__init__(tabix_file=bedgraph_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(1, 0), pos_indices=(0, 1, 2),
                         calc_chromlen=calc_chromlen)
        self.strand = strand

    def __iter__(self) -> Item:
        for loc, t in super().__iter__():
            self._stats['yielded_items', self.chromosome] += 1
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

    def __init__(self, tup):
        super().__init__()
        self.name = tup[3] if len(tup) >= 4 else None
        self.score = int(tup[4]) if len(tup) >= 5 and tup[4] != '.' else None
        strand = tup[5] if len(tup) >= 6 else None
        self.location = gi(tup[0], int(tup[1]) + 1, int(tup[2]), strand)  # convert -based to 1-based start
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

        Notes
        -----
        * NOTE that empty intervals (i.e., start==stop coordinate) will not be iterated.
        * reported stats:
                yielded_items, chromosome: (int, str)
                    Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, bed_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, calc_chromlen=False):
        assert guess_file_format(
            bed_file) == 'bed', f"expected BED file but guessed file format is {guess_file_format(bed_file)}"
        super().__init__(tabix_file=bed_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(1, 0), pos_indices=(0, 1, 2),
                         calc_chromlen=calc_chromlen)

    def __iter__(self) -> Item[gi, BedRecord]:
        chrom = self.refdict.alias(self.chromosome)  # check whether chrom exists in tabix contig list
        if (chrom is not None) and (chrom not in self.file.contigs):
            return
        for bed in self.file.fetch(reference=chrom,
                                   start=(self.start - 1) if (self.start > 0) else None,
                                   # 0-based coordinates in pysam!
                                   end=self.end if (self.end < MAX_INT) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            rec = BedRecord(tuple(bed))  # parse bed record
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(rec.location, rec)


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
        super().__init__(tabix_file=vcf_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=True, fun_alias=fun_alias, coord_inc=(0, 0), pos_indices=(0, 1, 1),
                         refdict=ReferenceDict.load(vcf_file, fun_alias), calc_chromlen=False)
        # get header
        self.header = pysam.VariantFile(vcf_file).header  # @UndefinedVariable
        self.allsamples = list(self.header.samples)  # list of all samples in this VCF file
        self.shownsampleindices = [i for i, j in enumerate(self.header.samples) if j in samples] if (
                samples is not None) else range(len(self.allsamples))  # list of all sammple indices to be considered
        self.filter_nocalls = filter_nocalls

    def __iter__(self) -> Item[gi, VcfRecord]:
        # check whether chrom exists in tabix contig list!
        chrom = self.refdict.alias(self.chromosome)
        if (chrom is not None) and (chrom not in self.file.contigs):
            return
        for pysam_var in self.file.fetch(reference=chrom,
                                         start=(self.start - 1) if (self.start > 0) else None,
                                         # 0-based coordinates in pysam!
                                         end=self.end if (self.end < MAX_INT) else None,
                                         parser=pysam.asVCF()):  # @UndefinedVariable
            rec = VcfRecord(pysam_var, self.allsamples, self.shownsampleindices, self.refdict)
            self.location = rec.location
            self.chromosome = self.location.chromosome
            self._stats['iterated_items', self.chromosome] += 1
            if ('n_calls' in rec.__dict__) and self.filter_nocalls and (rec.n_calls == 0):
                self._stats['filtered_nocalls', self.chromosome] += 1  # filter no-calls
                continue
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(self.location, rec)


class GFF3Iterator(TabixIterator):
    """
        Iterates a GTF/GFF3 file and yields 1-based coordinates and dicts containing key/value pairs parsed from
        the respective info sections. The feature_type, source, score and phase fields from the GFF/GTF entries are
        copied to this dict (NOTE: attribute fields with the same name will be overloaded).
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asGTF

        This iterator is used to build a transcriptome object from a GFF3/GTF file.

        Notes
        -----
        * reported stats:
            yielded_items, chromosome: (int, str)
                Number of iterated/yielded items
    """

    def __init__(self, gtf_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, calc_chromlen=False):
        super().__init__(tabix_file=gtf_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(0, 0), pos_indices=(0, 1, 2),
                         calc_chromlen=calc_chromlen)
        self.file_format = guess_file_format(gtf_file)
        assert self.file_format in ['gtf',
                                    'gff'], f"expected GFF3/GTF file but guessed file format is {self.file_format}"

    def __iter__(self) -> Item[gi, dict]:
        # check whether chrom exists in tabix contig list!
        chrom = self.refdict.alias(self.chromosome)
        if (chrom is not None) and (chrom not in self.file.contigs):
            return
        for row in self.file.fetch(reference=chrom,
                                   start=(self.start - 1) if (self.start > 0) else None,
                                   # 0-based coordinates in pysam!
                                   end=self.end if (self.end < MAX_INT) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome, source, feature_type, start, end, score, strand, phase, info = row
            self.location = gi(self.refdict.alias(chromosome), int(start) + self.coord_inc[0],
                               int(end) + self.coord_inc[1], strand)
            info = parse_gff_attributes(info, self.file_format)
            info['feature_type'] = None if feature_type == '.' else feature_type
            info['source'] = None if source == '.' else source
            info['score'] = None if score == '.' else float(score)
            info['phase'] = None if phase == '.' else int(phase)
            self._stats['yielded_items', self.chromosome] += 1
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
        coord_columns : list
            Names of coordinate columns, default: ['Chromosome', 'Start', 'End', 'Strand']
        coord_off : list
            Coordinate offsets, default: (1, 0). These offsets will be added to the  read start/end coordinates to
            convert to the rnalib convention.
        feature : str
            Name of column to yield. If null, the whole row will be yielded

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
        >>>     for loc, row in it: # now iterate with rnalib iterator
        >>>         # do something with location and pandas data row
    """

    def __init__(self, df, feature=None, chromosome=None, start=None, end=None, region=None, strand=None,
                 coord_columns=('Chromosome', 'Start', 'End', 'Strand'), is_sorted=False, per_position=False,
                 coord_off=(0, 0), fun_alias: Callable = None, calc_chromlen=False, refdict=None):
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
                self.refdict = ReferenceDict(self.df.groupby(coord_columns[0])[coord_columns[2]].max().to_dict())
            else:
                # refdict w/o chrom lengths
                self.refdict = ReferenceDict({c: None for c in self.chromosomes})
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, strand=strand,
                         per_position=False, fun_alias=None,  # already applied!
                         refdict=self.refdict)  # reference dict exists
        if (self.region is not None) and (
                not self.region.is_unbounded()):  # filter dataframe for region. TODO: check exact coords
            print(f"INFO: filtering dataframe for region {self.region}")
            filter_query = [] if self.region.chromosome is None else [f"{coord_columns[0]}==@self.region.chromosome"]
            # overlap check: self.start <= other.end and other.start <= self.end
            filter_query += [] if self.region.end is None else [f"{coord_columns[1]}<=(@self.region.end-@coord_off[0])"]
            filter_query += [] if self.region.start is None else [
                f"{coord_columns[2]}>=(@self.region.start-@coord_off[1])"]
            # print(feature"DF filter string: {'&'.join(filter_query)}, region: {self.region}")
            self.df = self.df.query('&'.join(filter_query))
        self.per_position = per_position

    def __iter__(self) -> Item[gi, pd.Series]:
        for row in self.df.itertuples():
            self.chromosome = getattr(row, self.coord_columns[0])
            if self.fun_alias is not None:
                self.chromosome = self.fun_alias(self.chromosome)
            # NOTE in df, coordinates are 0-based. Start is included, End is excluded.
            start = getattr(row, self.coord_columns[1]) + self.coord_off[0]
            end = getattr(row, self.coord_columns[2]) + self.coord_off[1]
            strand = getattr(row, self.coord_columns[3], '.')
            self.location = gi(self.chromosome, start, end, strand=strand)
            self._stats['iterated_items', self.chromosome] += 1
            if self.region.overlaps(self.location):
                self._stats['yielded_items', self.chromosome] += 1
                yield Item(self.location, row if self.feature is None else getattr(row, self.feature, None))

    def to_dataframe(self, **kwargs):
        return self.df

    def max_items(self):
        return len(self.df.index)


class BioframeIterator(PandasIterator):
    """
        Iterates over a [bioframe](https://bioframe.readthedocs.io/) dataframe.
        The genomic coordinates of yielded locations are corrected automatically.
    """

    def __init__(self, df, feature: str = None, chromosome: str = None, start: int = None, end: int = None,
                 region: gi = None,
                 strand: str = None, is_sorted=False, fun_alias=None, schema=None,
                 coord_columns: tuple = ('chrom', 'start', 'end', 'strand'),
                 calc_chromlen=False, refdict=None):
        if isinstance(df, str):
            # assume a filename and read via bioframe read_table method and make sure that dtypes match
            self.file = df
            df = bioframe.read_table(self.file, schema=guess_file_format(self.file) if schema is None else schema)
            # filter the 'chrom' column for header lines and replace NaN's
            df = df[~df.chrom.str.startswith('#', na=False)].replace(np.nan, ".")
            # ensure proper dtypes
            df = df.astype({coord_columns[0]: str, coord_columns[1]: int, coord_columns[2]: int})
            if coord_columns[3] in df.columns:
                df[coord_columns[3]] = df[coord_columns[3]].astype(str)
        super().__init__(df if is_sorted else bioframe.sort_bedframe(df),
                         feature, chromosome, start, end, region, strand,
                         coord_columns=coord_columns,
                         coord_off=(1, 0),  # coord correction
                         per_position=False,
                         is_sorted=True,  # we used bioframe for sorting above.
                         calc_chromlen=calc_chromlen,
                         refdict=refdict)


class PyrangesIterator(PandasIterator):
    """
        Iterates over a [pyranges](https://pyranges.readthedocs.io/) object.
        The genomic coordinates of yielded locations are corrected automatically.
    """

    def __init__(self, probj, feature=None, chromosome=None, start=None, end=None, region=None, strand=None,
                 is_sorted=False, fun_alias=None, coord_columns=('Chromosome', 'Start', 'End', 'Strand'),
                 calc_chromlen=False, refdict=None):
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
        super().__init__(probj.df,
                         feature, chromosome, start, end, region, strand,
                         coord_columns=coord_columns,
                         coord_off=(1, 0),  # coord correction
                         per_position=False,
                         is_sorted=True,  # we use bioframe for sorting above.
                         calc_chromlen=calc_chromlen,
                         refdict=refdict)


class MemoryIterator(LocationIterator):
    """
        A location iterator that iterates over intervals retrieved from one of the following datastructures:

        * {str:gi} dict: yields (gi,str); note that the passed strings must be unique
        * {gi:any} dict: yields (gi,any)
        * iterable of gis:  yields (gi, index in input iterable)

        regions will be sorted according to the passed refdict

        Notes
        -----
        * reported stats:
            iterated_items, chromosome: (int, str)
                Number of iterated items
            yielded_items, chromosome: (int, str)
                Number of yielded items (remaining are filtered, e.g., due to region constraints)

    """

    # d = { gi(1, 10,11): 'a', gi(1, 1,10): 'b', gi(2, 1,10): 'c' }
    def __init__(self, d, chromosome=None, start=None, end=None, region=None, fun_alias=None, calc_chromlen=False):
        if isinstance(d, dict):
            if len(d) > 0 and isinstance(next(iter(d.values())), str):  # {gi:str}
                d = {y: x for x, y in d.items()}
        else:
            d = {i: loc for i, loc in enumerate(d)}
        self._maxitems = len(d)
        # chrom aliasing
        if fun_alias is not None:
            for k, v in d.items():
                d[k] = gi(fun_alias(v.chromosome), v.start, v.end, strand=v.strand)
        # get list of chromosomes (aliased)
        self.chromosomes = []
        for loc in d.values():
            if loc.chromosome not in self.chromosomes:
                self.chromosomes.append(loc.chromosome)
        # split by chrom and sort
        self.data = {c: dict() for c in self.chromosomes}
        for name, loc in d.items():
            self.data[loc.chromosome][name] = loc
        # create refdict
        self.refdict = ReferenceDict(
            {c: max(self.data[c].values()).end if calc_chromlen else None for c in self.chromosomes})

        super().__init__(self.data,
                         chromosome=chromosome, start=start, end=end, region=region,
                         strand=None, file_format=None, chunk_size=1024, per_position=False,
                         fun_alias=fun_alias, refdict=self.refdict, calc_chromlen=False)

    def __iter__(self) -> Item[gi, object]:
        for self.chromosome in self.chromosomes:
            for name, self.location in dict(
                    sorted(self.data[self.chromosome].items(), key=lambda item: item[1])).items():
                self._stats['iterated_items', self.chromosome] += 1
                if self.region.overlaps(self.location):
                    self._stats['yielded_items', self.chromosome] += 1
                    yield Item(self.location, name)

    def max_items(self):
        return self._maxitems


class PybedtoolsIterator(LocationIterator):
    """ Iterates over a pybedtools BedTool

        Notes
        -----
        * reported stats:
            yielded_items, chromosome: (int, str)
                Number of yielded items

    """

    def __init__(self, bedtool, chromosome=None, start=None, end=None, strand=None, region=None, fun_alias=None,
                 calc_chromlen=False):
        self._stats = Counter()
        # instantiate bedtool
        self.bedtool = bedtool if isinstance(bedtool, pybedtools.BedTool) else pybedtools.BedTool(bedtool)
        # get ref dict (via pysam)
        self.fun_alias = fun_alias
        self.file = self.bedtool.fn if isinstance(self.bedtool.fn, str) else None
        if self.file is not None:
            # try to get ref dict which works only for bgzipped+tabixed files
            try:
                self.refdict = ReferenceDict.load(self.bedtool.fn, fun_alias,
                                                  calc_chromlen=calc_chromlen) if self.file else None
            except Exception as exp:
                print(f"WARN: Could not create refdict, is file bgzipped+tabixed? {exp}")
                self.refdict = None
        # intersect with region if any
        if region is not None:
            self.region = gi.from_str(region) if isinstance(region, str) else region
        else:
            self.region = gi(chromosome, start, end, strand)
        self.chromosome = chromosome
        # iterated chromosomes
        if self.chromosome is None:
            self.chromosomes = self.refdict.keys() if self.refdict is not None else None  # all chroms in correct order
        else:
            self.chromosomes = [self.chromosome]
        self.location = None

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return None

    def __iter__(self) -> Item[gi, pybedtools.Interval]:  # noqa
        current_bedtool = self.bedtool
        # intersect with region if any
        if (self.region is not None) and (not self.region.is_unbounded()):
            # intersect with -u to report original intervals
            current_bedtool = self.bedtool.intersect([self.region.to_pybedtools()], u=True)
        for iv in current_bedtool:
            self.location = gi(iv.chrom, iv.start + 1, iv.end, strand=iv.strand)
            self.chromosome = iv.chrom
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(self.location, iv)

    def close(self):
        pass


# ---------------------------------------------------------
# SAM/BAM iterators
# ---------------------------------------------------------

class ReadIterator(LocationIterator):
    """ Iterates over a BAM alignment.

        Parameters
        ----------
        bam_file : str
            BAM file name
        chromosome : str
            Chromosome name. If set, only this chromosome will be iterated.
        start : int
            Start position. If set, only reads overlapping this position will be iterated.
        end : int
            End position. If set, only reads overlapping this position will be iterated.
        region : str
            Genomic region. If set, only reads overlapping this region will be iterated.
        file_format : str
            File format, default: 'bam'
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
        fun_alias : function
            Function for aliasing chromosome names

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

    def __init__(self, bam_file, chromosome=None, start=None, end=None, region=None, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 report_mismatches=False, min_base_quality=0, fun_alias=None):
        super().__init__(bam_file, chromosome, start, end, region, file_format, per_position=False,
                         fun_alias=fun_alias, calc_chromlen=False)
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
        for r in self.file.fetch(contig=self.refdict.alias(self.chromosome),
                                 start=self.start if (self.start > 0) else None,
                                 end=self.end if (self.end < MAX_INT) else None,
                                 until_eof=True):
            self.location = gi(self.refdict.alias(r.reference_name), r.reference_start + 1, r.reference_end,
                               '-' if r.is_reverse else '+')
            self._stats['iterated_items', self.chromosome] += 1
            if r.flag & self.flag_filter:  # filter based on BAM flags
                self._stats['n_fil_flag', self.location.chromosome] += 1
                continue
            if r.mapping_quality < self.min_mapping_quality:  # filter based on mapping quality
                self._stats['n_fil_mq', self.location.chromosome] += 1
                continue
            if self.tag_filters is not None:  # filter based on BAM tags
                is_filtered = False
                for tf in self.tag_filters:
                    # print("test", tf, r.get_tag("MD"), tf.filter(r), type(r.get_tag('MD')) )
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
        `ignore_overlaps` or `ignore_orphans`). By default, it basically reports what is seen in the default IGV view.

        Parameters
        ----------
        bam_file : str
            BAM file name
        chromosome : str
            Chromosome name. IF set, only this chromosome will be iterated/
        reported_positions : range or set
            Range or set of genomic positions for which counts will be reported.
        file_format : str
            File format, default: 'bam'
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

    """

    def __init__(self, bam_file, chromosome, reported_positions, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 min_base_quality=0, max_depth=100000, fun_alias=None):
        self.reported_positions = reported_positions
        if isinstance(reported_positions, range):
            self.start, self.end = reported_positions.start, reported_positions.stop
        elif isinstance(reported_positions, set):
            self.start = min(reported_positions) - 1
            self.end = max(reported_positions) + 1
        else:
            print("reported_positions should be a tuple(start, end) or a set() to avoid slow processing")
            self.start = min(reported_positions) - 1
            self.end = max(reported_positions) + 1
        super().__init__(bam_file, chromosome, self.start, self.end, file_format, per_position=True,
                         fun_alias=fun_alias)
        self.min_mapping_quality = min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters = tag_filters
        self.min_base_quality = min_base_quality
        self.max_depth = max_depth
        self.count_dict = defaultdict(Counter)

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown."""
        return len(self.reported_positions)

    def __iter__(self) -> Item[gi, Counter]:
        self.rit = ReadIterator(self.file, self.chromosome, self.start, self.end,
                                min_mapping_quality=self.min_mapping_quality,
                                flag_filter=self.flag_filter, max_span=self.max_span,
                                fun_alias=self.fun_alias)
        for _, r in self.rit:
            self._stats['iterated_items', self.chromosome] += 1
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
                    print("unsupported CIGAR op %i" % op)
        # yield all reported positions (including uncovered ones)
        for gpos in self.reported_positions:
            self.location = gi(self.chromosome, gpos, gpos)
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(self.location, self.count_dict[gpos] if gpos in self.count_dict else Counter())


# ---------------------------------------------------------
# grouped iterators
# ---------------------------------------------------------


class BlockLocationIterator(LocationIterator):
    """ Returns locations and lists of values that share the same location wrt. a
        given matching strategy (e.g., same start, same end, same coords, overlapping).

        Expects a coordinate-sorted location iterator!

    """

    def __init__(self, it, strategy=BlockStrategy.LEFT):
        self.orgit = it
        self.it = peekable(it)
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
            if self.strategy == BlockStrategy.LEFT:
                while self.it.peek(None) and self.it.peek()[0].left_match(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = gi.merge((mloc, loc))
            elif self.strategy == BlockStrategy.RIGHT:
                while self.it.peek(None) and self.it.peek()[0].right_match(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = gi.merge((mloc, loc))
            elif self.strategy == BlockStrategy.BOTH:
                while self.it.peek(None) and self.it.peek()[0] == mloc:
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = gi.merge((mloc, loc))
            elif self.strategy == BlockStrategy.OVERLAP:
                while self.it.peek(None) and self.it.peek()[0].overlaps(mloc):
                    loc, v = next(self.it)
                    locations += [loc]
                    values += [v]
                    mloc = gi.merge((mloc, loc))
            yield Item(mloc, (locations, values))

    def close(self):
        try:
            self.orgit.close()
        except AttributeError:
            pass


class SyncPerPositionIterator(LocationIterator):
    """ Synchronizes the passed location iterators by genomic location and yields
        individual genomic positions and overlapping intervals per passed iterator.
        Expects (coordinate-sorted) location iterators.
        The chromosome order will be determined from a merged refdict or, if not possible,
        by alphanumerical order.

        Examples
        --------
        >>> it1, it2, it3 = ...
        >>> for pos, (i1,i2,i3) in SyncPerPositionIterator([it1, it2, it3]):
        >>>     print(pos,i1,i2,i3)
        >>>     # where i1,...,i3 are lists of loc/data tuples from the passed LocationIterators

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
            a list of location iterators
        refdict : ReferenceDict
            a reference dict for the passed iterators. If None, a merged refdict of all iterators will be used
        """
        self.iterables = iterables
        for it in iterables:
            assert issubclass(type(it), LocationIterator), "Only implemented for LocationIterators"
        self.per_position = True
        self._stats = Counter()
        self.iterators = [peekable(it) for it in iterables]  # make peekable iterators
        if refdict is None:
            self.refdict = ReferenceDict.merge_and_validate(*[it.refdict for it in iterables])
            if self.refdict is None:
                print("WARNING: could not determine refdict from iterators: using alphanumerical chrom order.")
            else:
                print("Iterating merged refdict:", self.refdict)
                if len(self.refdict) == 0:
                    print("WARNING refdict is empty!")
        else:
            self.refdict = refdict
            print("Iterating refdict:", self.refdict)
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
        """ Returns the first position of the next item from the iterators """
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
        """ Closes all wrapped iterators"""
        for it in self.iterables:
            try:
                it.close()
            except AttributeError:
                pass


class AnnotationIterator(LocationIterator):
    """
        Annotates locations in the first iterator with data from the ano_its location iterators.
        The returned data is a namedtuple with the following fields:
        Item(location=gi, data=Result(anno=dat_from_it, label1=[Item(loc, dat_from_anno_it1)], ..., labeln=[Item(loc,
        dat_from_anno_itn)])

        This enables access to the following data:

        * item.location: gi of the currently annotated location
        * item.data.anno: data of the currently annotated location
        * item.data.<label_n>: list of items from <iterator_n> that overlap the currently annotated position.

        if no labels are provided, the following will be used: it0, it1, ..., itn.

        Yields
        ------
        Item(location=gi, data=Result(anno=dat_from_it, label1=[Item(loc, dat_from_anno_it1)], ..., labeln=[Item(loc,
        dat_from_anno_itn)])


    """

    def __init__(self, it, anno_its, labels=None, refdict=None, disable_progressbar=False):
        """
        Parameters
        ----------
            it: the main location iterator. The created AnnotationIterator will yield each item from this iterator
                alongside all overlapping items from the configured anno_its iterators
            anno_its: a list of location iterators for annotating the main iterator
            labels: a list of labels for storing data from the annotating iterators
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
                # print(feature"Skipping chromosome {chromosome} as no annotation data found!")
                for loc, dat in self.it:
                    yield Item(loc, self.Result(dat, *self.current))  # noqa # yield empty results
                continue
            for it in anno_its:  # set current chrom
                it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [peekable(it) for it in anno_its]  # make peekable iterators
            for loc, dat in self.it:
                anno_its = self.update(loc, anno_its)
                yield Item(loc, self.Result(dat, *self.current))  # noqa

    def close(self):
        """ Closes all wrapped iterators"""
        for it in [self.it] + self.anno_its:
            try:
                it.close()
            except AttributeError:
                pass


class TiledIterator(LocationIterator):
    """ Wraps a location iterator and reports tuples of items yielded for each genomic tile.
        Tiles are either iterated from the passed regions_iterable or calculated from the reference dict via the
        refdict.iter_blocks(block_size=tile_size) method.

        The iterator yields the tile location and a tuple of overlapping items from the location iterator.
        The locations of the respective items can be access via it.tile_locations

        Parameters
        ----------
        location_iterator: LocationIterator
            The location iterator to be wrapped
        regions_iterable: Iterable[gi]
            An iterable of genomic intervals that define the tiles. If None, the tiles will be calculated from the
            reference dict based on the passed tile_size
        fun_alias: function
            Optional function for aliasing chromosome names
        tile_size: int
            The size of the tiles in base pairs
        calc_chromlen: bool
            If True, the chromlen will be calculated from the location iterator.
    """

    def __init__(self, location_iterator, regions_iterable=None, fun_alias=None, tile_size=1e8, calc_chromlen=False):
        assert issubclass(type(location_iterator), LocationIterator), \
            f"Only implemented for LocationIterators but not for {type(location_iterator)}"
        super().__init__(None, None, None, None, None, None, per_position=False, fun_alias=fun_alias,
                         calc_chromlen=calc_chromlen)
        self.location_iterator = location_iterator
        self.tile_size = tile_size
        self.regions_iterable = self.location_iterator.refdict.iter_blocks(block_size=int(self.tile_size)) \
            if regions_iterable is None else regions_iterable

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
# Additional, non location-bound iterators
# ---------------------------------------------------------

class FastqIterator:
    """
    Iterates a fastq file and yields FastqRead objects (named tuples containing sequence name, sequence and quality
    string (omitting line 3 in the FASTQ file)).

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
