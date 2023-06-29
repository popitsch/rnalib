"""
    Genomic itererables for efficient iteration over genomic data.

    - Most iterables inherit from LocationIterator and yield named tuples containing data and its respective genomic
        location.
    - Supports chunked I/O where feasible and not supported by the underlying (pysam) implementation (e.g.,
        FastaIterator)

    TODO
    - improve docs
    - strand specific iteration
    - url streaming

    @LICENSE
"""
import math
from collections import Counter, namedtuple
from enum import Enum
from itertools import chain
from os import PathLike

import pysam
from more_itertools import windowed, peekable
from sortedcontainers import SortedSet, SortedList
from tqdm import tqdm
import pandas as pd

from pygenlib.utils import gi, open_file_obj, DEFAULT_FLAG_FILTER, get_reference_dict, grouper, ReferenceDict, \
    guess_file_format, parse_gff_attributes

""" A location, data tuple, returned by an locationiteraor """
Item = namedtuple('Item', 'location data')


class LocationIterator:
    """
        Superclass.

        :param region genomic region to iterate; overrides chromosome/strat/end/strand params
    """

    def __init__(self, file, chromosome=None, start=None, end=None, region=None, strand=None, file_format=None,
                 chunk_size=1024, per_position=False,
                 fun_alias=None, refdict=None):
        self.stats = Counter()
        self.location = None
        self.per_position = per_position
        if isinstance(file, str) or isinstance(file, PathLike):
            self.file = open_file_obj(file, file_format=file_format)  # open new object
            self.was_opened = True
        else:
            self.file = file
            self.was_opened = False
        self.fun_alias = fun_alias
        self.refdict = refdict if refdict is not None else get_reference_dict(self.file, self.fun_alias) if self.file else None
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
            self.chromosomes = self.refdict.keys() if self.refdict is not None else None # all chroms in correct order
        else:
            self.chromosomes = [self.chromosome]
        return self

    def set_region(self, region):
        """ Update the iterated region of this iterator.
            Note that the region's chromosome must be in this iterators refdict (if any)
        """
        self.region = gi.from_str(region) if isinstance(region, str) else region
        self.chromosome, self.start, self.end = self.region.split_coordinates()
        if self.refdict is not None and self.chromosome is not None:
            assert self.chromosome in self.refdict, f"Invalid chromosome {self.chromosome} not in refddict {self.refdict}"

    def take(self):
        """ Exhausts iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]

    def max_items(self):
        """ Maximum numebr of items yielded by this iterator or None if unknown.
            Note that this is the upper boudn of yielded items but less (or even no) items may be yielded
            based on filter setings, etc.
            Useful, e.g., for progressbars or time estimates
        """
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.file and self.was_opened:
            print(f"Closing iterator {self}")
            self.file.close()


class TranscriptomeIterator(LocationIterator):
    """
        Iterates over features in a transcriptome object.
        Note that no chromosome aliasing is possible with this iterator as returned features are immutable.
    """

    def __init__(self, t, chromosome=None, start=None, end=None, region=None, feature_types=None, description='Transcriptome elements'):
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, refdict=t.merged_refdict)
        self.t = t
        self.feature_types = feature_types

    def __iter__(self) -> Item:
        for f in self.t.__iter__(feature_types=self.feature_types):
            # filter by genomic region
            if (self.region is not None) and (not f.overlaps(self.region)):
                self.stats[f'filtered_features'] += 1
                continue
            yield Item(f, self.t.anno[f])


class DictIterator(LocationIterator):
    """
        A simple location iterator that iterates over entries in a {name:gi} dict.
        Mainly for debugging purposes.
    """

    def __init__(self, d, chromosome=None, start=None, end=None, region=None, fun_alias=None):
        self.d = {n: gi(fun_alias(l.chromosome), l.start, l.end, l.strand) for n, l in
                  d.items()} if fun_alias else d
        self.refdict = ReferenceDict({l.chromosome: None for l in self.d.values()}, name='DictIterator',
                                     fun_alias=None)
        print(self.refdict)
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, per_position=False,
                         fun_alias=fun_alias, refdict=self.refdict)

    def max_items(self):
        return len(self.d)

    def __iter__(self) -> Item:
        for n, l in self.d.items():
            if self.region.overlaps(l):
                yield Item(l, n)


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
    location: gi
        Genomic source interval of the returned sequence.

    Yields
    ------
    sequence: str
        The extracted sequence in including sequence context.
        The core sequence w/o context can be accessed via seq[context_size:context_size+width].

    TODO: support chromosome=None; support max_items

    """

    def __init__(self, fasta_file, chromosome, start=None, end=None, location=None, width=1, step=1, file_format=None,
                 fillvalue='N', padding=False, fun_alias=None):
        super().__init__(fasta_file, chromosome, start, end, location, file_format, per_position=True,
                         fun_alias=fun_alias)
        self.width = 1 if width is None else width
        self.step = step
        self.fillvalue = fillvalue
        self.padding = padding

    def read_chunk(self, chromosome, start, end):
        """ Fetch a chunk from the FASTA file """
        return self.file.fetch(reference=self.refdict.alias(chromosome), start=start, end=end)  # @UndefinedVariable

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

    def __iter__(self) -> Item:
        padding = self.fillvalue * (self.width // 2) if self.padding else ""
        pos1 = 1 if (self.start is None) else max(1, self.start)  # 0-based coordinates in pysam!
        pos1 -= len(padding)
        for dat in windowed(chain(padding, self.iterate_data(), padding), fillvalue=self.fillvalue, n=self.width,
                            step=self.step):
            if isinstance(dat, tuple):
                dat = ''.join(dat)
            end_loc = pos1 + len(dat) - 1
            self.location = gi(self.chromosome, pos1, end_loc)
            self.stats['n_seq', self.chromosome] += 1
            yield Item(self.location, dat)
            pos1 += self.step


class TabixIterator(LocationIterator):
    """ Iterates over a bgzipped + tabix-indexed file and returns location/tuple pairs.
        Genomic locations will be parsed from the columns with given pos_indices and interval coordinates will be
        converted to 1-based inclusive coordinates by adding values from the configured coord_inc tuple to start and end
        coordinates. Note that this class serves as super-class for various file format specific iterators (e.g.,
        BedIterator, VcfIterator, etc.) which use proper coord_inc/pos_index default values.



        FIXME
        - add slop
        - improve docs
    """

    def __init__(self, tabix_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, per_position=False,
                 coord_inc=(0, 0),
                 pos_indices=(0, 1, 2)):
        super().__init__(file=tabix_file, chromosome=chromosome, start=start, end=end, region=region,
                         file_format='tsv', per_position=per_position, fun_alias=fun_alias)  # e.g., toggle_chr
        self.coord_inc = coord_inc
        self.pos_indices = pos_indices

    def __iter__(self) -> Item:
        for row in self.file.fetch(reference=self.refdict.alias(self.chromosome),
                                   start=(self.start - 1) if (self.start > 0) else None,  # 0-based coordinates in pysam!
                                   end=self.end if (self.end<math.inf) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome = self.refdict.alias(row[self.pos_indices[0]])
            start = int(row[self.pos_indices[1]]) + self.coord_inc[0]
            end = int(row[self.pos_indices[2]]) + self.coord_inc[1]
            self.location = gi(chromosome, start, end)
            self.stats['n_rows', chromosome] += 1
            yield Item(self.location, tuple(row))


class BedGraphIterator(TabixIterator):
    """
        Iterates a bgzipped and indexed bedgraph file and yields float values
        If a strand is passed, all yielded intervals will have this strand assigned.
    """

    def __init__(self, bedgraph_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, strand=None):
        super().__init__(tabix_file=bedgraph_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(1, 0), pos_indices=(0, 1, 2))
        self.strand = strand

    def __iter__(self) -> Item(gi, float):
        for loc, t in super().__iter__():
            if self.strand:
                loc = loc.get_stranded(self.strand)
            yield Item(loc, float(t[3]))


class BedRecord():
    """Parsed (mutable) version of pysam BedProxy"""

    def __init__(self, tup):
        self.name = tup[3] if len(tup) >= 4 else None
        self.score = tup[4] if len(tup) >= 5 else None
        strand = tup[5] if len(tup) >= 6 else None
        self.loc = gi(tup[0], int(tup[1]) + 1, int(tup[2]), strand)

    def __repr__(self):
        return f"{self.loc.chromosome}:{self.loc.start}-{self.loc.end} ({self.name})"


class BedIterator(TabixIterator):
    """
        Iterates a BED file and yields 1-based coordinates and pysam BedProxy objects
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asBed
    """

    def __init__(self, bed_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None):
        assert guess_file_format(
            bed_file) == 'bed', f"expected BED file but guessed file format is {guess_file_format(bed_file)}"
        super().__init__(tabix_file=bed_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(1, 0), pos_indices=(0, 1, 2))

    def __iter__(self) -> Item(gi, BedRecord):
        for bed in self.file.fetch(reference=self.refdict.alias(self.chromosome),
                                   start=(self.start - 1) if (self.start > 0) else None,  # 0-based coordinates in pysam!
                                   end=self.end if (self.end<math.inf) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            rec = BedRecord(tuple(bed))
            self.stats['n_rows', rec.loc.chromosome] += 1
            yield Item(rec.loc, rec)


# VCF no-call genotypes
gt2zyg_dict = {
    './.': None,
    '0/0': 0,
    './0': 0,
    '0/.': 0,
    './1': 1,
    '1/.': 1,
    '0/1': 1,
    '1/0': 1,
    '1/1': 2,
    './2': 1,
    '2/.': 1,
    '0/2': 1,
    '2/0': 1,
    '2/2': 2}


def gt2zyg(gt):
    if gt in gt2zyg_dict:
        return gt2zyg_dict[gt]
    dat = gt.split('/')
    return 2 if dat[0] == dat[1] else 1


class VcfRecord():
    """Parsed version of pysam VCFProxy, no type conversions for performance reasons"""
    location: gi = None

    def parse_info(self, info):
        if info == '.':
            return None
        ret = {}
        for x in info.split(';'):
            s = x.strip().split('=')
            if len(s) == 1:
                ret[s[0]] = True
            elif len(s) == 2:
                ret[s[0]] = s[1]
        return ret

    def __init__(self, pysam_var, samples, sample_indices, refdict):
        if (len(pysam_var.ref) == 1) and (len(pysam_var.alt) == 1):
            self.is_indel=False
            start, end = pysam_var.pos + 1, pysam_var.pos + 1 # 0-based in pysam
        else:  # INDEL
            self.is_indel = True
            start, end = pysam_var.pos + 2, pysam_var.pos + len(pysam_var.alt) # 0-based in pysam
        self.pos = start
        self.location = gi(refdict.alias(pysam_var.contig), start, end, None)
        self.id = pysam_var.id if pysam_var.id != '.' else None
        self.ref = pysam_var.ref
        self.alt = pysam_var.alt
        self.qual = pysam_var.qual if pysam_var.qual != '.' else None
        self.info = self.parse_info(pysam_var.info)
        self.format = pysam_var.format.split(":")
        for col, x in enumerate(self.format):  # make faster
            self.__dict__[x] = {k: v for k, v in zip([samples[i] for i in sample_indices],
                                                     [pysam_var[i].split(':')[col] for i in sample_indices])}
        # calc zygosity per call
        if 'GT' in self.__dict__:
            self.zyg = {k: gt2zyg(v) for k, v in self.__dict__['GT'].items()}
            self.n_calls = sum([0 if (x is None) or (x == 0) else 1 for x in self.zyg.values()])

    def __repr__(self):
        return f"{self.location.chromosome}:{self.pos}{self.ref}>{self.alt}"


class VcfIterator(TabixIterator):
    """
        Iterates a VCF file and yields 1-based coordinates and pysam VcfProxy objects
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asVCF
    """

    def __init__(self, vcf_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, samples=None, filter_nocalls=True ):
        assert guess_file_format(
            vcf_file) == 'vcf', f"expected VCF file but guessed file format is {guess_file_format(vcf_file)}"
        super().__init__(tabix_file=vcf_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=True, fun_alias=fun_alias, coord_inc=(0, 0), pos_indices=(0, 1, 1))
        # get header
        self.header = pysam.VariantFile(vcf_file).header  # @UndefinedVariable
        self.allsamples = list(self.header.samples) # list of all samples in this VCF file
        self.shownsampleindices = [i for i, j in enumerate(self.header.samples) if j in samples] if (
                samples is not None) else range(len(self.allsamples)) # list of all sammple indices to be considered
        self.filter_nocalls = filter_nocalls

    def __iter__(self) -> Item(gi, VcfRecord):
        for pysam_var in self.file.fetch(reference=self.refdict.alias(self.chromosome),
                                   start=(self.start - 1) if (self.start > 0) else None,  # 0-based coordinates in pysam!
                                   end=self.end if (self.end<math.inf) else None,
                                   parser=pysam.asVCF()):  # @UndefinedVariable
            rec = VcfRecord(pysam_var, self.allsamples, self.shownsampleindices, self.refdict)
            if ('n_calls' in rec.__dict__) and (self.filter_nocalls) and (rec.n_calls == 0):
                self.stats['filtered_nocalls'] += 1 # filter no-calls
                continue
            self.location=rec.location
            self.stats['n_rows', self.location.chromosome] += 1
            yield Item(self.location, rec)


class GFF3Iterator(TabixIterator):
    """
        Iterates a GTF/GFF3 file and yields 1-based coordinates and dicts containing key/value pairs parsed from
        the respective info sections. The feature_type, source, score and phase fields from the GFF/GTF entries are
        copied to this dict (NOTE: attribute fields with the same name will be overloaded).
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asGTF
    """

    def __init__(self, gtf_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None):
        super().__init__(tabix_file=gtf_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(0, 0), pos_indices=(0, 1, 2))
        self.file_format = guess_file_format(gtf_file)
        assert self.file_format in ['gtf',
                                    'gff'], f"expected GFF3/GTF file but guessed file format is {self.file_format}"

    def __iter__(self) -> Item(gi, dict):
        for row in self.file.fetch(reference=self.refdict.alias(self.chromosome),
                                   start=(self.start - 1) if (self.start>0) else None,  # 0-based coordinates in pysam!
                                   end=self.end if (self.end<math.inf) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome, source, feature_type, start, end, score, strand, phase, info = row
            self.location = gi(self.refdict.alias(chromosome), int(start) + self.coord_inc[0],
                               int(end) + self.coord_inc[1], strand)
            info = parse_gff_attributes(info, self.file_format)
            info['feature_type'] = None if feature_type == '.' else feature_type
            info['source'] = None if source == '.' else source
            info['score'] = None if score == '.' else float(score)
            info['phase'] = None if phase == '.' else int(phase)
            self.stats['n_rows', chromosome] += 1
            yield Item(self.location, info)


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

    def __init__(self, df, feature, chromosome=None, start=None, end=None, region=None, strand=None,coord_columns=['Chromosome', 'Start', 'End'], is_sorted=False, per_position=False,
                 fun_alias=None):
        self.stats = Counter()
        self.location = None
        self.df = df if is_sorted else df.sort_values(['Chromosome', 'Start'])
        self.chromosomes=list(dict.fromkeys(df['Chromosome'])) # unique set with preserved order
        self.refdict=ReferenceDict({c:None for c in self.chromosomes})
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, per_position=False,
                         fun_alias=fun_alias, refdict=self.refdict)
        self.feature = feature
        self.coord_columns = coord_columns
        self.per_position = per_position

    def __iter__(self) -> Item(gi, pd.Series):
        for _, row in self.df.iterrows():
            chromosome = row[self.coord_columns[0]] if self.fun_alias is None else self.fun_alias(
                row[self.coord_columns[0]])
            start = row[self.coord_columns[1]]
            end = row[self.coord_columns[2]]
            self.location = gi(chromosome, start, end)
            if self.region.overlaps(self.location):
                self.stats['n_rows', chromosome] += 1
                yield Item(self.location, row[self.feature])

    def close(self):
        pass


# ---------------------------------------------------------
# SAM/BAM iterators
# ---------------------------------------------------------
class ReadIterator(LocationIterator):
    """ Iterates over a BAM alignment.

        :parameter flag_filter for filtering reads based on read flags
        :parameter tag_filters for filtering reads based on BAM tag values
        :parameter max_span for filtering reads based on maximum alignment span (end_pos-start_pos+1)
        :parameter report_mismatches if set, this iterator will additionally yield (read, mismatches) tuples where
            'mismatches' is a list of (read_position, genomic_position, ref_allele, alt_allele) tuples that describe
            differences wrt. the aligned reference sequence. This options requires MD BAM tags to be present.
        :parameter min_base_quality useful only in combination with report_mismatches; filters mismatches based on
            minimum per-base quality

    """

    def __init__(self, bam_file, chromosome=None, start=None, end=None, location=None, file_format=None,
                 min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER, tag_filters=None, max_span=None,
                 report_mismatches=False, min_base_quality=0, fun_alias=None):
        super().__init__(bam_file, chromosome, start, end, location, file_format, per_position=False,
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

    def __iter__(self) -> Item(gi, pysam.AlignedSegment):
        md_check = False
        for r in self.file.fetch(contig=self.refdict.alias(self.chromosome),
                                 start=self.start if (self.start>0) else None,
                                 end=self.end if (self.end<math.inf) else None,
                                 until_eof=True):
            self.location = gi(self.refdict.alias(r.reference_name), r.reference_start + 1, r.reference_end, '-' if r.is_reverse else '+')
            if r.flag & self.flag_filter:  # filter based on BAM flags
                self.stats['n_fil_flag', self.location.chromosome] += 1
                continue
            if r.mapping_quality < self.min_mapping_quality:  # filter based on mapping quality
                self.stats['n_fil_mq', self.location.chromosome] += 1
                continue
            if self.tag_filters is not None:  # filter based on BAM tags
                is_filtered = False
                for tf in self.tag_filters:
                    # print("test", tf, r.get_tag("MD"), tf.filter(r), type(r.get_tag('MD')) )
                    is_filtered = is_filtered | tf.filter(r)
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
                      r.query_qualities[off] > self.min_base_quality]  # mask bases with low per-base quailty
                yield Item(self.location, (r, mm))  # yield read/mismatch tuple
            else:
                yield Item(self.location, r)


class FastPileupIterator(LocationIterator):
    """
        Fast pileup iterator that yields a complete pileup (w/o insertions) over a set of genomic positions. This is
        more lightweight and considerably faster than pysams pileup() but lacks some features (such as 'ignore_overlaps'
        or 'ignore_orphans'). By default, it basically reports what is seen in the default IGV view.


        :parameter reported_positions  either a range (start/end) or a set of genomic positions for which counts will be reported.
        :parameter min_base_quality filters pileup based on minimum per-base quality
        :parameter max_depth restricts maximum pileup depth.

        :return A base:count Counter object. Base is 'None' for deletions.
        Note that this implementation does not support 'ignore_overlaps' or 'ignore_orphans' options like pysam.pileup()
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
        self.count_dict = Counter()

    def __iter__(self) -> Item(gi, Counter):
        self.rit = ReadIterator(self.file, self.chromosome, self.start, self.end,
                                min_mapping_quality=self.min_mapping_quality,
                                flag_filter=self.flag_filter, max_span=self.max_span,
                                fun_alias=self.fun_alias)
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
            yield Item(self.location, self.count_dict[gpos] if gpos in self.count_dict else Counter())


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
        self.per_position = self.orgit.per_position

    def __iter__(self) -> Item(gi, (tuple, tuple)):
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
                    mloc = gi.merge((mloc, l))
            elif self.strategy == BlockStrategy.RIGHT:
                while self.it.peek(None) and self.it.peek()[0].right_match(mloc):
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = gi.merge((mloc, l))
            elif self.strategy == BlockStrategy.BOTH:
                while self.it.peek(None) and self.it.peek()[0] == mloc:
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = gi.merge((mloc, l))
            elif self.strategy == BlockStrategy.OVERLAP:
                while self.it.peek(None) and self.it.peek()[0].overlaps(mloc):
                    l, v = next(self.it)
                    locations += [l]
                    values += [v]
                    mloc = gi.merge((mloc, l))
            yield Item(mloc, (locations, values))

    def close(self):
        try:
            self.orgit.close()
        except AttributeError:
            pass



class SyncPerPositionIterator(LocationIterator):
    """ Synchronizes the passed location iterators by genomic location and yields
        individual genomic positions and overlapping intervals per passed iterator
        Expects coordinate-sorted location iterators!
        The chromosome order will be determined from a merged refdict or, if not possible,
        by alphanumerical order.

        Example:
            for pos, (i1,i2,i3) in SyncPerPositionIterator([it1, it2, it3]):
                print(pos,i1,i2,i3)
                ...
            where i1,..,i3 are lists of loc/data tuples from the passed LocationIterators

        TODO:
        - optimize
        - documentation and usage scenarios
    """

    def __init__(self, iterables, refdict=None):
        """
        :parameter iterables a list of location iterators
        """
        self.iterables = iterables
        for it in iterables:
            assert issubclass(type(it), LocationIterator), "Only implemented for LocationIterators"
        self.per_position = True
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
            # defined chrom sort order (fixed list) or create on the fly (sorted set that supports addition in iteration)
            self.chroms = SortedSet([first_positions[0].chromosome]) if self.refdict is None else list(
                self.refdict.keys())  # chrom must be in same order as in iterator!
            # get min.max pos
            self.pos, self.maxpos = first_positions[0].start, first_positions[0].end
        else:
            self.chroms = set()  # no data

    def first_pos(self):
        first_positions = SortedList(d[0] for d in [it.peek(default=None) for it in self.iterators] if d is not None)
        if len(first_positions) > 0:
            return first_positions[0]
        return None

    def update(self, chromosome):
        if self.pos is None:
            return
        for i, it in enumerate(self.iterators):
            self.current[i] = [(l, d) for l, d in self.current[i] if l.end >= self.pos]
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

    def __iter__(self) -> Item(gi, tuple):
        for c in self.chroms:
            self.update(c)
            while self.pos <= self.maxpos:
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
        for it in self.iterables:
            try:
                it.close()
            except AttributeError:
                pass


# ---------------------------------------------------------
# non location iterators
# ---------------------------------------------------------
FastqRead = namedtuple('FastqRead', 'name seq qual')


class FastqIterator():
    """
    Iterates a fastq file and yields FastqRead objects (named tuples containing sequence name, sequence and quality
    string (omitting line 3 in the FASTQ file)).

    Note that this is not a location iterator.
    """

    def __init__(self, fastq_file):
        self.file = fastq_file
        self.per_position = False

    def __len__(self):
        """ Fast read counting """
        i = 0
        with open_file_obj(self.file, file_format='fastq') as fin:
            for i, _ in enumerate(fin):
                pass
        assert (i + 1) % 4 == 0, "Invalid readcount, not divisible by 4: {i+1}"  # fastq complete?
        return (i + 1) // 4

    def __iter__(self) -> FastqRead:
        """
            Iterates FASTQ entries
        """
        with open_file_obj(self.file, file_format='fastq') as fin:
            for d in grouper(fin, 4, ''):
                yield FastqRead(d[0].strip(), d[1].strip(), d[3].strip())

    def take(self):
        """ Exhausts iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]

    def close(self):
        if self.file and self.was_opened:
            print(f"Closing iterator {self}")
            self.file.close()

class AnnotationIterator(LocationIterator):
    """
        Annotates locations in the first iterator with data from the ano_its location iterators.
        Yields the following data:
            Item(location=gi, data=Result(anno=dat_from_it,
                            label1=[Item(loc, dat_from_anno_it1)], ...,
                            labeln=[Item(loc, dat_from_anno_itn)])
            Which enables access to the following data
                item.location: gi of the currently annotated location
                item.data.anno: data of the currently annotated location
                item.data.<label_n>: list of items from <iterator_n> that overlap the currentluy annotated position.

            if no labels are provided, the following will be used: it0, it1, ..., itn.


    """

    def __init__(self, it, anno_its, labels=None, refdict=None):
        """
        :parameter it a location iterator; the created AnnotationIterator will yield each item from this iterator
            alongside all overlapping items from the configured anno_its iterators
        :parameter anno_its a list of location iterators for annotating the main iterator
        """
        if not isinstance(anno_its, list):
            anno_its = [anno_its]
        if labels is None:
            labels = [f'it{i}' for i in range(len(anno_its))]
        elif not isinstance(labels, list):
            labels = [labels]
        for x in [it] + anno_its:
            assert issubclass(type(x), LocationIterator), f"Only implemented for LocationIterators {type(x)}"
        self.it = it
        self.refdict = it.refdict if refdict is None else refdict
        self.anno_its=anno_its
        self.region=it.region
        self.chromosomes = it.chromosomes
        self.Result = namedtuple('Result', ['anno'] + labels)  # result type

    def coord_overlaps(self, a, b):
        return a.start <= b.end and b.start <= a.end
    def update(self, ref, anno_its):
        for i, it in enumerate(anno_its):
            self.buffer[i] = [Item(l, d) for l, d in self.buffer[i] if l.end >= ref.start]  # drop left
            while True:
                nxt = it.peek(default=None)
                if nxt is None:
                    break  # exhausted
                loc, dat = nxt
                if loc.start > ref.end:
                    break  # done
                nxt = next(it)  # consume interval
                if loc.end < ref.start:
                    continue # skip
                self.buffer[i].append(nxt)
            self.current[i] = [Item(l, d) for l, d in self.buffer[i] if self.coord_overlaps(l,ref)]
        return anno_its

    def __iter__(self) -> Item(gi, tuple):
        for chromosome in tqdm(self.chromosomes, total=len(self.chromosomes)):
            self.buffer = [list() for i, _ in enumerate(
                self.anno_its)]  # holds sorted interval lists that overlap or are > than the currently annotated interval
            self.current = [list() for i, _ in enumerate(self.anno_its)]
            # set chrom of it
            self.it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [it for it in self.anno_its if chromosome in it.refdict]
            if len(anno_its)==0:
                #print(f"Skipping chromosome {chromosome} as no annotation data found!")
                for loc, dat in self.it:
                    yield Item(loc, self.Result(dat, *self.current)) # yield empty results
                continue
            for it in anno_its: # set current chrom
                it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its=[peekable(it) for it in anno_its]  # make peekable iterators
            for loc, dat in self.it:
                anno_its=self.update(loc, anno_its)
                yield Item(loc, self.Result(dat, *self.current))

    def close(self):
        for it in [self.it] + self.anno_its:
            try:
                it.close()
            except AttributeError:
                pass
class AnnotationIterator_OLD(LocationIterator):
    """
        Annotates locations in the first iterator with data from the ano_its location iterators.
        Yields the following data:
            Item(location=gi, data=Result(anno=dat_from_it,
                            label1=[Item(loc, dat_from_anno_it1)], ...,
                            labeln=[Item(loc, dat_from_anno_itn)])
            Which enables access to the following data
                item.location: gi of the currently annotated location
                item.data.anno: data of the currently annotated location
                item.data.<label_n>: list of items from <iterator_n> that overlap the currentluy annotated position.

            if no labels are provided, the following will be used: it0, it1, ..., itn.


    """

    def __init__(self, it, anno_its, labels=None, refdict=None):
        """
        :parameter it a location iterator; the created AnnotationIterator will yield each item from this iterator
            alongside all overlapping items from the configured anno_its iterators
        :parameter anno_its a list of location iterators for annotating the main iterator
        """
        if not isinstance(anno_its, list):
            anno_its = [anno_its]
        if labels is None:
            labels = [f'it{i}' for i in range(len(anno_its))]
        elif not isinstance(labels, list):
            labels = [labels]
        for x in [it] + anno_its:
            assert issubclass(type(x), LocationIterator), f"Only implemented for LocationIterators {type(x)}"
        self.it = peekable(it)
        self.anno_its = [peekable(it) for it in anno_its]  # make peekable iterators
        self.refdict = it.refdict if refdict is None else refdict
        #print("Iterating refdict:", self.refdict)
        self.chroms = list(self.refdict.keys())
        self.buffer = [list() for i, _ in enumerate(self.anno_its)] # holds sorted interval lists that overlap or are > than the currently annotated interval
        self.current = [list() for i, _ in enumerate(self.anno_its)]
        self.Result = namedtuple('Result', ['anno'] + labels)  # result type

    def update(self, ref):
        for i, it in enumerate(self.anno_its):
            #self.current[i] = [Item(l, d) for l, d in self.current[i] if l.overlaps(ref)]  # drop non overlapping
            self.buffer[i] = [Item(l, d) for l, d in self.buffer[i] if l.overlaps(ref) or l > ref]  # drop non overlapping
            while True:
                nxt = it.peek(default=None)
                if nxt is None:
                    break  # exhausted
                loc, dat = nxt
                if loc.chromosome != ref.chromosome:
                    if (loc.chromosome not in self.chroms) or (
                            self.chroms.index(loc.chromosome) < self.chroms.index(ref.chromosome)):
                        next(it)  # consume and skip
                        continue
                    else:
                        break  # chrom comes after current one
                over = loc.overlaps(ref)
                if (loc > ref) and not (over):
                    break  # done
                nxt = next(it)  # consume interval
                if over:
                    self.buffer[i].append(nxt)
            self.current[i] = [Item(l, d) for l, d in self.buffer[i] if l.overlaps(ref)]

    def __iter__(self) -> Item(gi, tuple):
        for loc, dat in self.it:
            if self.refdict and loc.chromosome not in self.refdict:
                continue  # skip chrom
            self.update(loc)
            yield Item(loc, self.Result(dat, *self.current))

    def close(self):
        for it in [self.it] + self.anno_its:
            try:
                it.close()
            except AttributeError:
                pass
