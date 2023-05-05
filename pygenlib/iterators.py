"""
Genomic itererables for efficient iteration over genomic data.
All iterables inherit from LocationIterator and yield data alongside the respective genomic coordinates.
Supports chunked I/O where feasible and not supported by the underlying (pysam) implementation (e.g., FastaIterator)

TODO
- docs
- fastqiterator
"""
from enum import Enum
from typing import overload

import pysam
from abc import abstractmethod
from more_itertools import windowed, peekable
from itertools import chain, islice
from pygenlib.utils import open_pysam_obj, BAM_FLAG, DEFAULT_FLAG_FILTER, get_reference_dict
from pygenlib.genemodel import loc_obj
import pyranges as pr
from collections import Counter
from os import PathLike
def merge_yields(l) -> (loc_obj, tuple):
    """ Takes an enumeration of (loc,payload) tuples and returns a tuple (merged location, payloads) """
    l1, l2 = zip(*l)
    mloc = loc_obj.merge(list(l1))
    return mloc, l2



class LocationIterator:
    """Superclass"""

    def __init__(self, file, chromosome=None, start=None, end=None, region=None, file_format=None, chunk_size=1024):
        self.stats=Counter()
        self.location=None
        if isinstance(file, str) or isinstance(file, PathLike):
            self.file = open_pysam_obj(file, file_format=file_format)  # open new object
            print(self.file)
            self.was_opened = True
        else:
            self.file=file
            self.was_opened=False
        self.refdict = get_reference_dict(self.file)
        self.chunk_size = chunk_size
        if region is not None:
            location = loc_obj.from_str(region) if isinstance(region, str) else region
            self.chromosome, self.start, self.end = location.split_coordinates()
        else:
            self.chromosome = chromosome
            self.start = start
            self.end = end
        assert (self.chromosome is None) or (self.chromosome in self.refdict), f"{chromosome} not found in references {self.refdict}"
        return self

    def take(self):
        """ Exhausts iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.file and self.was_opened:
            print(f"Closing iterator {self}")
            self.file.close()


class FastaIterator(LocationIterator):
    """ A generator that iterates over a FASTA file yielding sequence strings and keepd track of the covered
        genomic location.

    The yielded sequence of length <width> will be padded with <fillvalue> characters if padding is True.
    This generator will yield every step_size window, for a tiling window approach set step_size = width.
    FASTA files will be automatically indexed if no existing index file is found and will be read in chunked mode.

    Parameters
    ----------
    fasta_file : str or pysam FastaFile
        A file path or an open FastaFile object. In the latter case, the object will not be closed

    chromosome, start, end, location : coordinates
    width : int
        sequence window size
    step : int
        increment for each yield
    file_format : str
        optional, will be determined from filename if omitted

    State
    ------
    location: loc_obj
        Location object describing the returned sequence.

    Yields
    ------
    sequence: str
        The extracted sequence in including sequence context. The core sequence w/o context can be accessed via seq[context_size:context_size+width].

    """

    def __init__(self, fasta_file, chromosome, start=None, end=None, location=None, width=1, step=1, file_format=None,
                 fillvalue='N', padding=False):
        super().__init__(fasta_file, chromosome, start, end, location, file_format)
        self.width = 1 if width is None else width
        self.step = step
        self.fillvalue = fillvalue
        self.padding = padding

    def read_chunk(self, chromosome, start, end):
        """ Fetch a chunk from the FASTA file """
        return self.file.fetch(reference=chromosome, start=start, end=end)  # @UndefinedVariable

    def iterate_data(self):
        """
            Reads pysam data in chunks and yields individual data items
        """
        start_chunk = 0 if (self.start is None) else max(0, self.start - 1)  # 0-based coordinates in pysam!
        while True:
            end_chunk = (start_chunk + self.chunk_size) if self.end is None else min(self.end,
                                                                                     start_chunk + self.chunk_size)
            if end_chunk <= start_chunk:
                break
            chunk = self.read_chunk(self.chromosome, start_chunk, end_chunk)
            if (chunk is None) or (len(chunk) == 0):  # we are done
                break
            start_chunk += len(chunk)
            for d in chunk:
                yield d

    def __iter__(self) -> (loc_obj, str):
        padding = self.fillvalue * (self.width // 2) if self.padding else ""
        pos1 = 1 if (self.start is None) else max(1, self.start)  # 0-based coordinates in pysam!
        pos1 -= len(padding)
        for dat in windowed(chain(padding, self.iterate_data(), padding), fillvalue=self.fillvalue, n=self.width,
                            step=self.step):
            if isinstance(dat, tuple):
                dat = ''.join(dat)
            end_loc = pos1 + len(dat) - 1
            self.location = loc_obj(self.chromosome, pos1, end_loc)
            self.stats['n_seq', self.chromosome]+=1
            yield self.location, dat
            pos1 += self.step

class CHR_PREFIX_OPERATOR(Enum):
    ADD = 1 # add chr prefix
    DEL = 2 # remove chr prefix

class TabixIterator(LocationIterator):
    """ Iterates over tabix-indexed files and returns location/tuple pairs.

        Requires a tabix-indexed file. Genomic locations will be parsed from the columns with given pos_indices.
        For 'known' tabix based file formats (e.g., BED) this will be set automatically (and will overload any configured values).

        For convenience, 'chr' prefixes can be added to or removed from the read file if chr_prefix_operator is set to
        'add' or 'del' respectively. Use with care to avoid incompatible genomic datasets.

        Example:
                BED: coord_inc = [1, 0]
                VCF: pos_indices=[0, 1, 1]

        FIXME
        - add slop
        - fill/fill_value
        - improve docs
    """

    def __init__(self, tabix_file, chromosome=None, start=None, end=None,
                 region=None, chr_prefix_operator=None,
                 coord_inc=[0, 0],
                 pos_indices=[0, 1, 2]):

        super().__init__(file=tabix_file, chromosome=chromosome, start=start, end=end, region=region, file_format='tsv')  # open or retain object
        self.chr_prefix_operator = chr_prefix_operator  # chr_prefix_operator can be 'add_chr', 'drop_chr' or None
        self.coord_inc = coord_inc
        self.pos_indices = pos_indices
        self.start = 1 if (self.start is None) else max(1, self.start)  # 0-based coordinates in pysam!

    def escape_chr(self, reference):
        if not self.chr_prefix_operator:
            return(reference)
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.ADD:
            return 'chr' + reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.DEL:
            return reference[3:]

    def unescape_chr(self, reference):
        if not self.chr_prefix_operator:
            return(reference)
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.DEL:
            return 'chr' + reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR:
            return reference[3:]

    def __iter__(self) -> (loc_obj, str):
        for row in self.file.fetch(reference=self.escape_chr(self.chromosome),
                                   start=self.start - 1, # 0-based coordinates in pysam!
                                   end=self.end,
                                   parser=pysam.asTuple()): # @UndefinedVariable
            chromosome = self.unescape_chr(row[self.pos_indices[0]])
            start = int(row[self.pos_indices[1]]) + self.coord_inc[0]
            end = int(row[self.pos_indices[2]]) + self.coord_inc[1]
            self.location = loc_obj(chromosome, start, end)
            self.stats['n_rows', chromosome] += 1
            yield self.location, tuple(row)

class PandasIterator(LocationIterator):
    """ Iterates over a pandas dataframe that contains three columns with chromosome/start/end coordinates.
        Compatible with pyranges (@see https://github.com/biocore-ntnu/pyranges)
        The dataframe MUST be coordinate sorted!

        Parameters
        ----------
        df : dataframe
            pandas datafram with at least 4 columns names as in coord_columns and feature parameter.
            This dataframe will be sorted by chromosome and start values unless is_sorted is set to True

        coord_columns : list
            Names of coordinate columns, default: ['Chromosome', 'Start', 'End']

        feature : str
            Name of column to yield

        Yields
        ------
        location: Location
            Location object describing the current coordinates
        value: object
            The extracted feature value

    """
    def __init__(self, df, feature, coord_columns=['Chromosome', 'Start', 'End'], is_sorted=False):
        self.stats=Counter()
        self.file=None
        self.location=None
        self.df = df if is_sorted else df.sort_values(['Chromosome', 'Start'])
        self.feature=feature
        self.coord_columns = coord_columns

    def __iter__(self) -> (loc_obj, str):
        for _, row in self.df.iterrows():
            chromosome=row[self.coord_columns[0]]
            start=row[self.coord_columns[1]]
            end=row[self.coord_columns[2]]
            self.location = loc_obj(chromosome, start, end)
            self.stats['n_rows', chromosome] += 1
            yield self.location, row[self.feature]
    def close(self):
        pass

# ---------------------------------------------------------
# SAM/BAM iterators
# ---------------------------------------------------------
class ReadIterator(LocationIterator):
    """ Iterates over a BAM alignment.
        NOTE: bam_file can be a file path (which will open a new pysam.AlignmentFile) or an existing pysam.AlignmentFile object
    """
    def __init__(self, bam_file, chromosome=None, start=None, end=None, location=None, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 report_mismatches=False, min_base_quality=0):
        super().__init__(bam_file, chromosome, start, end, location, file_format)
        self.min_mapping_quality=min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters=tag_filters
        self.report_mismatches=report_mismatches
        self.min_base_quality = min_base_quality
    def __iter__(self) -> (loc_obj, str):
        md_check=False
        for r in self.file.fetch(contig=self.chromosome,
                                    start=self.start,
                                    end=self.end,
                                    until_eof=True):
            self.location = loc_obj(r.reference_name, r.reference_start+1, r.reference_end, '-' if r.is_reverse else '+')
            if r.flag & self.flag_filter: # filter based on BAM flags
                self.stats['n_fil_flag', self.location.chromosome]+=1
                continue
            if r.mapping_quality < self.min_mapping_quality: # filter based on mapping quality
                self.stats['n_fil_mq', self.location.chromosome]+=1
                continue
            if self.tag_filters is not None: # filter based on BAM tags
                is_filtered=False
                for tf in self.tag_filters:
                    #print("test", tf, r.get_tag("MD"), tf.filter(r), type(r.get_tag('MD')) )
                    is_filtered=is_filtered | tf.filter(r)
                if is_filtered:
                    self.stats['n_fil_tag', self.location.chromosome] += 1
                    continue
            # test max_span and drop reads that span larger genomic regions
            if (self.max_span is not None) and (len(self.location) > self.max_span):
                self.stats['n_fil_max_span', self.location.chromosome] += 1
                continue
            self.stats['n_reads', self.location.chromosome] += 1
            # report mismatches
            if self.report_mismatches:
                if not md_check:
                    assert r.has_tag("MD"), "BAM does not contain MD tag: cannot report mismatches"
                mm = [(off, pos + 1, ref.upper(), r.query_sequence[off]) for (off, pos, ref) in
                      r.get_aligned_pairs(with_seq=True, matches_only=True) if ref.islower() and
                      r.query_qualities[off] > self.min_base_quality] # mask bases with low per-base quailty
                yield self.location, r, mm
            else:
                yield self.location, r

class FastPileupIterator(LocationIterator):
    """
        Fast pileup iterator that yields a complete pileup (no INDELs) over a set of genomic positions.
        Avoids the heavy pysam pileup command.
        reported_positions: either a range (start/end) or a set of genomic positions for which counts will be reported.
        count_dict: a Counter containing (pos, base): count tuples.

        max_depth restricts maximum pileup depth.
    """
    def __init__(self, bam_file, chromosome, reported_positions, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 min_base_quality=0, max_depth=100000):
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
        super().__init__(bam_file, chromosome, self.start, self.end, file_format)
        self.min_mapping_quality=min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters=tag_filters
        self.min_base_quality = min_base_quality
        self.max_depth = max_depth
        self.count_dict = Counter()

    def __iter__(self) -> (loc_obj, str):
        self.rit = ReadIterator(self.file, self.chromosome, self.start, self.end,
                                min_mapping_quality=self.min_mapping_quality,
                                flag_filter=self.flag_filter, max_span=self.max_span)
        for _, r in self.rit:
            # find reportable positions, skips softclipped bases
            gpos = r.reference_start + 1
            rpos = 0
            for op, l in r.cigartuples:
                if op in [0, 7, 8]:  # M, =, X
                    for _ in range(l):
                        if gpos in self.reported_positions:
                            if r.query_qualities[rpos] >= self.min_base_quality:  # check base qual
                                if not self.count_dict[gpos]:
                                    self.count_dict[gpos]=Counter()
                                self.count_dict[gpos][r.query_sequence[rpos]] += 1
                        rpos += 1
                        gpos += 1
                elif op == 1:  # I
                    rpos += l
                elif op == 2:  # D
                    for _ in range(l):
                        if gpos in self.reported_positions:
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
        for gpos in self.count_dict:
            self.location = loc_obj(self.chromosome, gpos, gpos)
            yield self.location, self.count_dict[gpos]


# ---------------------------------------------------------
# grouped iterators
# ---------------------------------------------------------
class BlockStrategy(Enum):
    LEFT = 1  # same start coord
    RIGHT = 2  # same end coord
    BOTH = 3  # same start and end
    OVERLAP = 4  # overlapping


class BlockLocationIterator(LocationIterator):
    """ Returns locations and lists of values that share the same location wrt. a
        given matching strategy (e.g., same start, same end, same coords, overlapping).

        Expects a coordinate-sorted location iterator!
    """

    def __init__(self, it, strategy=BlockStrategy.LEFT, stranded=True):
        self.orgit = it
        self.it = peekable(it)
        self.strategy = strategy

    def __iter__(self) -> (loc_obj, (tuple, tuple)):
        values = None
        locations = None
        for l, value in self.it:
            mloc = l.copy()
            values = [value]
            locations = [l]
            if self.strategy == BlockStrategy.LEFT:
                while self.it.peek(None) and self.it.peek()[0].left_match(mloc):
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = loc_obj.merge((mloc, l))
            elif self.strategy == BlockStrategy.RIGHT:
                while self.it.peek(None) and self.it.peek()[0].right_match(mloc):
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = loc_obj.merge((mloc, l))
            elif self.strategy == BlockStrategy.BOTH:
                while self.it.peek(None) and self.it.peek()[0] == mloc:
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = loc_obj.merge((mloc, l))
            elif self.strategy == BlockStrategy.OVERLAP:
                while self.it.peek(None) and self.it.peek()[0].overlaps(mloc):
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = loc_obj.merge((mloc, l))
            yield mloc, (locations, values)

    def close(self):
        try:
            self.orgit.close()
        except AttributeError:
            pass


#
#
# if __name__ == '__main__':
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=1).take())
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=2).take())
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=10).take())
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=1, context_size=1).take())
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=2, context_size=1).take())
#     # print(FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', 1, 10, width=10, context_size=1).take())
#     #print('\n'.join(str(x) for x in FastaIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta','1', width=10, step_size=10).take()))
#
#     # dict_chr2idx, dict_idx2chr, dict_chr2len = get_chrom_dicts('/Users/niko.popitsch/git/genomic_iterators/tests/data/reference1.fasta')
#     # print(TabixIterator('/Users/niko.popitsch/git/genomic_iterators/tests/data/test_snps.vcf.gz',
#     #                     dict_chr2idx, dict_chr2len, '1', 1, 10, file_format='tsv', coord_inc=[0,0], pos_indices=[0,1,1]).take())
#
#     # to test interactively in PyDev console:
#     # from genomic_iterators.iterators import FastaIterator
#     # [...] modify source + save
#     # import imp
#     # imp.reload(genomic_iterators.iterators)
#     pass
#
