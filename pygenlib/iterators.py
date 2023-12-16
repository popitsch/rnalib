"""
    Genomic iterables for efficient iteration over genomic data.

    - Most iterables inherit from LocationIterator and yield named tuples containing data and its respective genomic
        location.
    - Supports chunked I/O where feasible and not supported by the underlying (pysam) implementation (e.g.,
        FastaIterator)
    - Standard interface for region-type filtering
    - Iterators keep track of their current genomic position and number of iterated/yielded items per chromosome.

    TODO
    - improve docs
    - strand specific iteration
    - url streaming
    - remove self.chromosome and get from self.location
    - add is_exhausted flag?

    @LICENSE
"""
import math
from collections import Counter, namedtuple
from enum import Enum
from itertools import chain
from os import PathLike
from more_itertools import windowed, peekable
from sortedcontainers import SortedSet, SortedList
from tqdm import tqdm
from intervaltree import IntervalTree

import pybedtools
import bioframe
import pysam
import pandas as pd
import numpy as np

from pygenlib.utils import MAX_INT, gi, open_file_obj, DEFAULT_FLAG_FILTER, get_reference_dict, grouper, ReferenceDict, \
    guess_file_format, parse_gff_attributes, get_unique_keys


class Item(namedtuple('Item', 'location data')):
    """ A location, data tuple, returned by an LocationIterator """

    def __len__(self):
        """ Reports the length of the wrapped location, not the current tuple"""
        return self.location.__len__()


class LocationIterator:
    """
        Superclass.

        _stats is a counter that collects statistics (e.g., yielded items per chromosome) for QC and reporting.

        Parameters
        ----------

        region genomic region to iterate; overrides chromosome/start/end/strand params
    """

    def __init__(self, file, chromosome=None, start=None, end=None, region=None, strand=None, file_format=None,
                 chunk_size=1024, per_position=False,
                 fun_alias=None, refdict=None, calc_chromlen=False):
        self._stats = Counter()  # counter for collecting stats
        self.location = None
        self.per_position = per_position
        if isinstance(file, str) or isinstance(file, PathLike):
            self.file = open_file_obj(file, file_format=file_format)  # open new object
            self.was_opened = True
        else:
            self.file = file
            self.was_opened = False
        self.fun_alias = fun_alias
        self.calc_chromlen = calc_chromlen
        self.refdict = refdict if refdict is not None else get_reference_dict(self.file,
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
        return self._stats

    def set_region(self, region):
        """ Update the iterated region of this iterator.
            Note that the region's chromosome must be in this iterators refdict (if any)
        """
        self.region = gi.from_str(region) if isinstance(region, str) else region
        self.chromosome, self.start, self.end = self.region.split_coordinates()
        if self.refdict is not None and self.chromosome is not None:
            assert self.chromosome in self.refdict, f"Invalid chromosome {self.chromosome} not in refddict {self.refdict}"

    def to_list(self):
        """ Consumes iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]

    def to_dataframe(self,
                     fun=lambda loc, item, fun_col, default_value: [str(item)],  # default: convert item to string repr
                     fun_col=['Value'],
                     coord_inc=(0, 0),
                     coord_colnames=['Chromosome', 'Start', 'End', 'Strand'],
                     excluded_columns=None,
                     included_columns=None,
                     dtypes=None,
                     default_value=None):
        """ Consumes iterator and returns results in a dataframe.
            Start/stop Coordinates will be corrected by the passed coord_inc tuple.

            fun is a function that converts the yielded items of this iterator into a tuple of values that represent
            fun_col column values in the created dataframe
        """
        assert fun is not None
        assert (fun_col is not None) and (isinstance(fun_col, list))
        # exclude/include columns
        if excluded_columns is not None:
            fun_col = [col for col in fun_col if col not in excluded_columns]
        if included_columns is not None:
            fun_col += list(included_columns)
        df = pd.DataFrame([[loc.chromosome,
                            loc.start,
                            loc.end,
                            '.' if loc.strand is None else loc.strand] + fun(loc, item, fun_col, None) for loc, item in
                           tqdm(self, desc=f"Building dataframe")],
                          columns=coord_colnames + fun_col)
        if dtypes is not None:
            for k, v in dtypes.items():
                df[k] = df[k].astype(v)
        # Correct coordinates if required
        for inc, col in zip(coord_inc, [coord_colnames[1], coord_colnames[2]]):
            if inc != 0:
                df[col] += inc
        return df

    def to_intervaltrees(self):
        """ Consumes iterator and returns results in a dict of intervaltrees.
            NOTE that this will silently drop empty intervals
            TODO improve performance with from_tuples method
        """
        chr2itree = {}  # a dict mapping chromosome ids to annotation interval trees.
        for loc, item in tqdm(self, desc=f"Building interval trees"):
            if loc.is_empty():
                continue
            if loc.chromosome not in chr2itree:
                chr2itree[loc.chromosome] = IntervalTree()
            # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
            chr2itree[loc.chromosome].addi(loc.start, loc.end + 1, item)
        return chr2itree

    def max_items(self):
        """ Maximum number of items yielded by this iterator or None if unknown.
            Note that this is the upper bound of yielded items but less (or even no) items may be yielded
            based on filter settings, etc. Useful, e.g., for progressbars or time estimates
        """
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.file and self.was_opened:
            # print(f"Closing iterator {self}")
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

        # advanced usage: convert to pandas dataframe with a custom conversion function
        # here we add a 'feature length' column
        def my_fun(loc, item, fun_col, default_value):
            return [len(loc) if col=='feature_len' else loc.get(col, default_value) for col in fun_col]
        TranscriptomeIterator(t).to_dataframe(fun=my_fun, included_annotations=['feature_len']).head()

        Stats
        ------
        iterated_items, chromosome: (int, str)
            Number of iterated items
        yielded_items, chromosome: (int, str)
            Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, t, chromosome=None, start=None, end=None, region=None, feature_types=None):
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, refdict=t.merged_refdict)
        self.t = t
        self.feature_types = feature_types

    def to_dataframe(self,
                     fun=lambda loc, item, fun_col, default_value: [loc.get(col, default_value) for col in fun_col],
                     excluded_columns={'dna_seq'},
                     included_columns=None,
                     coord_inc=(0, 0), coord_colnames=['Chromosome', 'Start', 'End', 'Strand'],
                     dtypes=None,
                     default_value=None):
        """ Consumes iterator and returns results in a dataframe.
            Start/stop Coordinates will be corrected by the passed coord_inc tuple.

            fun is a function that converts the yielded items of this iterator into a tuple of values that represent
            fun_col column values in the created dataframe
        """
        # mandatory fields; we use a dict to keep column order nice
        fun_col = {'feature_id': None, 'feature_type': None}
        # add all annotation keys from the anno dict
        fun_col.update(dict.fromkeys(get_unique_keys(self.t.anno), None))
        # add annotation fields parsed from GFF
        fun_col.update(dict.fromkeys(get_unique_keys(self.t._ft2anno_class), None))
        # preserve order
        fun_col = list(fun_col.keys())
        # call super method
        return super().to_dataframe(fun=fun,
                                    fun_col=fun_col,
                                    coord_inc=coord_inc,
                                    coord_colnames=coord_colnames,
                                    dtypes=dtypes,
                                    excluded_columns=excluded_columns,
                                    included_columns=included_columns,
                                    default_value=default_value)

    def __iter__(self) -> Item:
        for f in self.t.__iter__(feature_types=self.feature_types):
            self.chromosome = f.chromosome
            self._stats['iterated_items', self.chromosome] += 1
            # filter by genomic region
            if (self.region is not None) and (not f.overlaps(self.region)):
                continue
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(f, self.t.anno[f])


class FastaIterator(LocationIterator):
    """ Generator that iterates over a FASTA file yielding sequence strings and keeps track of the covered
        genomic location.

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

    Stats
    -----
    iterated_items, chromosome: (int, str)
        Number of iterated and yielded items

    TODO: support chromosome=None; support max_items; support region filering + stats

    """

    def __init__(self, fasta_file, chromosome, start=None, end=None, region=None, width=1, step=1, file_format=None,
                 fillvalue='N', padding=False, fun_alias=None):
        super().__init__(fasta_file, chromosome, start, end, region, file_format, per_position=True,
                         fun_alias=fun_alias, calc_chromlen=False)
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
            self._stats['iterated_items', self.chromosome] += 1
            yield Item(self.location, dat)
            pos1 += self.step


class TabixIterator(LocationIterator):
    """ Iterates over a bgzipped + tabix-indexed file and returns location/tuple pairs.
        Genomic locations will be parsed from the columns with given pos_indices and interval coordinates will be
        converted to 1-based inclusive coordinates by adding values from the configured coord_inc tuple to start and end
        coordinates. Note that this class serves as super-class for various file format specific iterators (e.g.,
        BedIterator, VcfIterator, etc.) which use proper coord_inc/pos_index default values.


        Stats
        ------
        iterated_items, chromosome: (int, str)
            Number of iterated/yielded items



        FIXME
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

    def __iter__(self) -> Item:
        # we need to check whether chrom exists in tabix contig list!
        chrom = self.refdict.alias(self.chromosome)
        if (chrom is not None) and (chrom not in self.file.contigs):
            # print(f"{chrom} not in {self.file.contigs}")
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

        Stats
        ------
        yielded_items, chromosome: (int, str)
            Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    def __init__(self, bedgraph_file, chromosome=None, start=None, end=None,
                 region=None, fun_alias=None, strand=None, calc_chromlen=False):
        super().__init__(tabix_file=bedgraph_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=False, fun_alias=fun_alias, coord_inc=(1, 0), pos_indices=(0, 1, 2),
                         calc_chromlen=calc_chromlen)
        self.strand = strand

    def __iter__(self) -> Item(gi, float):
        for loc, t in super().__iter__():
            self._stats['yielded_items', self.chromosome] += 1
            if self.strand:
                loc = loc.get_stranded(self.strand)
            yield Item(loc, float(t[3]))


class BedRecord:
    """
        Parsed and mutable version of pysam.BedProxy
        TODO add BED12 support
    """

    def __init__(self, tup):
        self.name = tup[3] if len(tup) >= 4 else None
        self.score = tup[4] if len(tup) >= 5 else None
        strand = tup[5] if len(tup) >= 6 else None
        self.loc = gi(tup[0], int(tup[1]) + 1, int(tup[2]), strand)  # convert -based to 1-based start

    def __repr__(self):
        return f"{self.loc.chromosome}:{self.loc.start}-{self.loc.end} ({self.name})"


class BedIterator(TabixIterator):
    """
        Iterates a BED file and yields 1-based coordinates and pysam BedProxy objects
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asBed

        NOTE that empty intervals (i.e., start==stop coordinate) will not be iterated.

        Stats
        ------
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

    def __iter__(self) -> Item(gi, BedRecord):
        chrom = self.refdict.alias(self.chromosome)  # check whether chrom exists in tabix contig list
        if (chrom is not None) and (chrom not in self.file.contigs):
            return
        for bed in self.file.fetch(reference=chrom,
                                   start=(self.start - 1) if (self.start > 0) else None,
                                   # 0-based coordinates in pysam!
                                   end=self.end if (self.end < MAX_INT) else None,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            rec = BedRecord(tuple(bed))
            self._stats['yielded_items', self.chromosome] += 1
            yield Item(rec.loc, rec)


def gt2zyg(gt) -> (int, int):
    """
    Parameters
    ----------
    gt genotype

    Returns
    -------
    zygosity of GT and a flag if called.
    zygosity: 2: all called alleles are the same, 1: mixed called alleles, 0: no call
    call: 0 if no-call or homref, 1 otherwise
    """
    dat = gt.split('/') if '/' in gt else gt.split('|')
    if set(dat) == {'.'}:  # no call
        return 0, 0
    dat_clean = [x for x in dat if x != '.']  # drop no-calls
    if set(dat_clean) == {'0'}:  # homref in all called samples
        return 2, 0
    return 2 if len(set(dat_clean)) == 1 else 1, 1


class VcfRecord:
    """
        Parsed version of pysam VCFProxy, no type conversions for performance reasons.
        features:
        - is_indel: true if this is an INDEL
        - pos: 1-based genomic (start) position. For deletions, this is the first deleted genomic position
        - location: genomic interval representing this record
        - ref/alt: reference/alternate allele string
        - qual: variant call quality
        - info: dict of info fields/values
        - genotype (per-sample) dicts: for each FORMAT field (including 'GT'), a {sample_id: value} dict will be created.
        - zyg: zygosity information per sample. Created by mapping genotypes to zygosity values using gt2zyg()
            (0=nocall, 1=heterozygous call, 2=homozygous call).
        - n_calls: number of calles alleles (among all considered samples)

        @see https://samtools.github.io/hts-specs/VCFv4.2.pdf
    """
    location: gi = None

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
        self.location = gi(refdict.alias(pysam_var.contig), start, end, None)
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


        Features:
        - header: the pysam VariantFile header
        - allsamples: list of all comtained samples
        - shownsampleindices: indices of all configured samples

        Stats
        ------
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
                         refdict=get_reference_dict(vcf_file, fun_alias), calc_chromlen=False)
        # get header
        self.header = pysam.VariantFile(vcf_file).header  # @UndefinedVariable
        self.allsamples = list(self.header.samples)  # list of all samples in this VCF file
        self.shownsampleindices = [i for i, j in enumerate(self.header.samples) if j in samples] if (
                samples is not None) else range(len(self.allsamples))  # list of all sammple indices to be considered
        self.filter_nocalls = filter_nocalls

    def __iter__(self) -> Item(gi, VcfRecord):
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

        Stats
        ------
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

    def __iter__(self) -> Item(gi, dict):
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

        NOTE that iteration over dataframes is generally discouraged as there are much more efficient methods
        for data manipulation. Here is another (efficient) way to use a pandas iterator:

        with BioframeIterator(gencode_gff, chromosome='chr2') as it: # create a filtered dataframe from the passed gff
            it.df = it.df.query("strand=='-' ") # further filter the dataframe with pandas
            print(Counter(it.df['feature'])) # counte - strand features
            # you can also use this code to then iterate the data which may be convenient/readable if the
            # dataframe is small:
            for loc, row in it: # now iterate with pygenlib iterator
                # do something with location and pandas data row

        Parameters
        ----------
        df : dataframe
            pandas datafram with at least 4 columns names as in coord_columns and feature parameter.
            This dataframe will be sorted by chromosome and start values unless is_sorted is set to True

        coord_columns : list
            Names of coordinate columns, default: ['Chromosome', 'Start', 'End', 'Strand']

        coord_off : list
            Coordinate offsets, default: [1, 0]
            These offsets will be added to the  read start/end coordinates to convert to the pygenlib convention.

        feature : str
            Name of column to yield. If null, the whole row will be yielded

        Yields
        ------
        location: Location
            Location object describing the current coordinates
        value: object
            The extracted feature value

        Stats
        ------
        iterated_items, chromosome: (int, str)
            Number of iterated items
        yielded_items, chromosome: (int, str)
            Number of yielded items (remaining are filtered, e.g., due to region constraints)

    """

    def __init__(self, df, feature=None, chromosome=None, start=None, end=None, region=None, strand=None,
                 coord_columns=['Chromosome', 'Start', 'End', 'Strand'], is_sorted=False, per_position=False,
                 coord_off=(0, 0), fun_alias=None, calc_chromlen=False, refdict=None):
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
            # print(f"DF filter string: {'&'.join(filter_query)}, region: {self.region}")
            self.df = self.df.query('&'.join(filter_query))
        self.per_position = per_position

    def __iter__(self) -> Item(gi, pd.Series):
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

    def to_dataframe(self):
        return self.df

    def max_items(self):
        return len(self.df.index)


class BioframeIterator(PandasIterator):
    """
        Iterates over a [bioframe](https://bioframe.readthedocs.io/) dataframe.
        The genomic coordinates of yielded locations are corrected automatically.
    """

    def __init__(self, df, feature=None, chromosome=None, start=None, end=None, region=None, strand=None,
                 is_sorted=False, fun_alias=None, schema=None, coord_columns=['chrom', 'start', 'end', 'strand'],
                 calc_chromlen=False, refdict=None):
        if isinstance(df, str):
            # assume a filename and read via bioframe read_table method and make sure that dtypes match
            self.file = df
            df = bioframe.read_table(self.file, schema=guess_file_format(self.file) if schema is None else schema). \
                astype({coord_columns[0]: str, coord_columns[1]: int, coord_columns[2]: int}, errors='ignore')
            # filter the 'chrom' column for header lines and replace NaN's
            df = df[~df.chrom.str.startswith('#', na=False)].replace(np.nan, ".")
            if coord_columns[3] in df.columns:
                df[coord_columns[3]] = df[coord_columns[3]].astype(str)
        super().__init__(df if sorted else bioframe.sort_bedframe(df),
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
        - {str:gi} dict: yields (gi,str); note that the passed strings must be unique
        - {gi:any} dict: yields (gi,any)
        - iterable of GIs:  yields (gi, index in input iterable)
        Regions will be sorted according to the passed refdict

        Stats
        ------
        iterated_items, chromosome: (int, str)
            Number of iterated items
        yielded_items, chromosome: (int, str)
            Number of yielded items (remaining are filtered, e.g., due to region constraints)
    """

    # d = { gi(1, 10,11): 'a', gi(1, 1,10): 'b', gi(2, 1,10): 'c' }
    def __init__(self, d, chromosome=None, start=None, end=None, region=None, fun_alias=None, calc_chromlen=False):
        if isinstance(d, dict):
            if len(d) > 0 and isinstance(next(iter(d.values())), str):  # {gi:str}
                d = { y : x for x, y in d.items()}
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
        self.refdict = ReferenceDict({c: max(self.data[c].values()).end if calc_chromlen else None for c in self.chromosomes})

        super().__init__(self.data,
                         chromosome=chromosome, start=start, end=end, region=region,
                         strand=None, file_format=None, chunk_size=1024, per_position=False,
                         fun_alias=fun_alias, refdict=self.refdict, calc_chromlen=False)

    def __iter__(self) -> Item[gi, object]:
        for self.chromosome in self.chromosomes:
            for name, self.location in dict(sorted(self.data[self.chromosome].items(), key=lambda item: item[1])).items():
                self._stats['iterated_items', self.chromosome] += 1
                if self.region.overlaps(self.location):
                    self._stats['yielded_items', self.chromosome] += 1
                    yield Item(self.location, name)
    def max_items(self):
        return self._maxitems

# class PyrangesIterator(PandasIterator):
#     def __init__(self, pyrangesobject, feature, chromosome=None, start=None, end=None, region=None, strand=None,
#                  is_sorted=False, fun_alias=None):
#         super().__init__(df, feature, chromosome, start, end, region, strand, coord_columns=['Chromosome', 'Start', 'End', 'Strand'], coord_off=[1, 0], per_position=False)

class PybedtoolsIterator(LocationIterator):
    """ Iterates over a pybedtools BedTool

        Stats
        ------
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
                self.refdict = get_reference_dict(self.bedtool.fn, fun_alias,
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

    def __iter__(self) -> Item[gi, pybedtools.Interval]:
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

        :parameter flag_filter for filtering reads based on read flags
        :parameter tag_filters for filtering reads based on BAM tag values
        :parameter max_span for filtering reads based on maximum alignment span (end_pos-start_pos+1)
        :parameter report_mismatches if set, this iterator will additionally yield (read, mismatches) tuples where
            'mismatches' is a list of (read_position, genomic_position, ref_allele, alt_allele) tuples that describe
            differences wrt. the aligned reference sequence. This options requires MD BAM tags to be present.
        :parameter min_base_quality useful only in combination with report_mismatches; filters mismatches based on
            minimum per-base quality

        Stats
        ------
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
        more lightweight and considerably faster than pysams pileup() but lacks some features (such as 'ignore_overlaps'
        or 'ignore_orphans'). By default, it basically reports what is seen in the default IGV view.

        Performance:
        - initial performance tests that used a synchronized iterator over a FASTA and 2 BAMs showed ~50kpos/sec


        :parameter reported_positions  either a range (start/end) or a set of genomic positions for which counts will be reported.
        :parameter min_base_quality filters pileup based on minimum per-base quality
        :parameter max_depth restricts maximum pileup depth.

        :return A base:count Counter object. Base is 'None' for deletions.
        Note that this implementation does not support 'ignore_overlaps' or 'ignore_orphans' options like pysam.pileup()

        Stats
        ------
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
        self.count_dict = Counter()

    def __iter__(self) -> Item[gi, Counter]:
        self.rit = ReadIterator(self.file, self.chromosome, self.start, self.end,
                                min_mapping_quality=self.min_mapping_quality,
                                flag_filter=self.flag_filter, max_span=self.max_span,
                                fun_alias=self.fun_alias)
        for _, r in self.rit:
            self._stats['iterated_items', self.chromosome] += 1
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
            self._stats['yielded_items', self.chromosome] += 1
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

    def __init__(self, it, strategy=BlockStrategy.LEFT):
        self.orgit = it
        self.it = peekable(it)
        self.strategy = strategy
        self.per_position = self.orgit.per_position

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
        individual genomic positions and overlapping intervals per passed iterator
        Expects coordinate-sorted location iterators!
        The chromosome order will be determined from a merged refdict or, if not possible,
        by alphanumerical order.

        Example:
            for pos, (i1,i2,i3) in SyncPerPositionIterator([it1, it2, it3]):
                print(pos,i1,i2,i3)
                ...
            where i1,..,i3 are lists of loc/data tuples from the passed LocationIterators

        Stats
        ------
        yielded_items, chromosome: (int, str)
            Number of yielded items (positions)


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
        for it in self.iterables:
            try:
                it.close()
            except AttributeError:
                pass


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
            assert issubclass(type(x),
                              LocationIterator), f"Only implemented for LocationIterators but not for {type(x)}"
        self._stats = Counter()
        self.it = it
        self.refdict = it.refdict if refdict is None else refdict
        self.anno_its = anno_its
        self.region = it.region
        self.chromosomes = it.chromosomes
        self.Result = namedtuple('Result', ['anno'] + labels)  # result type

    @LocationIterator.stats.getter
    def stats(self):
        """Return stats of wrapped main iterator"""
        return self.it.stats

    def update(self, ref, anno_its):
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
        for chromosome in tqdm(self.chromosomes, total=len(self.chromosomes)):
            self.buffer = [list() for i, _ in enumerate(
                self.anno_its)]  # holds sorted interval lists that overlap or are > than the currently annotated interval
            self.current = [list() for i, _ in enumerate(self.anno_its)]
            # set chrom of it
            self.it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [it for it in self.anno_its if chromosome in it.refdict]
            if len(anno_its) == 0:
                # print(f"Skipping chromosome {chromosome} as no annotation data found!")
                for loc, dat in self.it:
                    yield Item(loc, self.Result(dat, *self.current))  # yield empty results
                continue
            for it in anno_its:  # set current chrom
                it.set_region(gi(chromosome, self.region.start, self.region.end))
            anno_its = [peekable(it) for it in anno_its]  # make peekable iterators
            for loc, dat in self.it:
                anno_its = self.update(loc, anno_its)
                yield Item(loc, self.Result(dat, *self.current))

    def close(self):
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

        TODO compare to annotation iterator
    """

    def __init__(self, location_iterator, regions_iterable=None, fun_alias=None, tile_size=1e8, calc_chromlen=False):
        assert issubclass(type(location_iterator),
                          LocationIterator), f"Only implemented for LocationIterators but not for {type(location_iterator)}"
        super().__init__(None, None, None, None, None, None, per_position=False, fun_alias=fun_alias,
                         calc_chromlen=calc_chromlen)
        self.location_iterator = location_iterator
        self.tile_size = tile_size
        self.regions_iterable = self.location_iterator.refdict.iter_blocks(block_size=int(self.tile_size)) \
            if regions_iterable is None else regions_iterable

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
FastqRead = namedtuple('FastqRead', 'name seq qual')


class FastqIterator:
    """
    Iterates a fastq file and yields FastqRead objects (named tuples containing sequence name, sequence and quality
    string (omitting line 3 in the FASTQ file)).

    Note that this is not a location iterator.

    Stats
    ------
    yielded_items: int
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

    def take(self):
        """ Exhausts iterator and returns results in a list.
            For debugging/testing only
        """
        return [x for x in self]
