"""
    Genomic itererables for efficient iteration over genomic data.

    - Most iterables inherit from LocationIterator and yield data alongside the respective genomic coordinates
    - Supports chunked I/O where feasible and not supported by the underlying (pysam) implementation (e.g., FastaIterator)

    TODO
    - docs
    - strand specific iteration
    - url streaming

    @LICENSE
"""
from enum import Enum
from typing import overload

import pysam
from abc import abstractmethod
from more_itertools import windowed, peekable
from itertools import chain, islice
import pyranges as pr
from collections import Counter, namedtuple
from os import PathLike
from sortedcontainers import SortedSet, SortedList
from itertools import pairwise

from pygenlib.utils import gi, open_file_obj, BAM_FLAG, DEFAULT_FLAG_FILTER, get_reference_dict, grouper, ReferenceDict, \
    guess_file_format, parse_gff_info

def merge_yields(l) -> (gi, tuple):
    """ Takes an enumeration of (loc,payload) tuples and returns a tuple (merged location, payloads) """
    l1, l2 = zip(*l)
    mloc = gi.merge(list(l1))
    return mloc, l2


class LocationIterator:
    """Superclass"""

    def __init__(self, file, chromosome=None, start=None, end=None, region=None, strand=None, file_format=None, chunk_size=1024, per_position=False):
        self.stats = Counter()
        self.location = None
        self.per_position=per_position
        if isinstance(file, str) or isinstance(file, PathLike):
            self.file = open_file_obj(file, file_format=file_format)  # open new object
            self.was_opened = True
        else:
            self.file = file
            self.was_opened = False
        self.refdict = get_reference_dict(self.file) if self.file else None
        self.chunk_size = chunk_size
        self.strand = strand
        if region is not None:
            location = gi.from_str(region) if isinstance(region, str) else region
            self.chromosome, self.start, self.end = location.split_coordinates()
        else:
            self.chromosome = chromosome
            self.start = start
            self.end = end
        assert (self.refdict is None) or (self.chromosome is None) or (self.chromosome in self.refdict), f"{chromosome} not found in references {self.refdict}"
        self.region=gi(self.chromosome, self.start, self.end, self.strand)
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

    def annotate(self, anno_its, field_names):
        if not isinstance(anno_its, list):
            anno_its=[anno_its]
            field_names=[field_names]
        for p, (loc, idx, data) in SyncPerPositionIterator([self]+anno_its, self.refdict):
            # pass refdict to iterate also intervals that are not covered in the annotation
            for l in [loc[i] for i,x in enumerate(idx) if x==0]:
                for a in range(len(anno_its)):
                    anno_val = [data[i] for i, x in enumerate(idx) if x == a + 1]
                    if not hasattr(l, field_names[a]):
                        l.__setattr__(field_names[a], [])
                    for v in anno_val:
                        getattr(l, field_names[a]).append(v)
                if p.start==l.end:
                #     l.mean_value=np.mean(l.mean_value) if len(l.mean_value)>0 else None
                     yield l

class DictIterator(LocationIterator):
    """
        A simple location iterator that iterates over entries in a {name:gi} dict.
        Mainly for debugging purposes.
    """
    def __init__(self, d, chromosome=None, start=None, end=None, region=None):
        super().__init__(None, chromosome=chromosome, start=start, end=end, region=region, per_position=False)
        self.d=d

    def __iter__(self) -> (gi, str):
        for n,l in self.d.items():
            if self.region.overlaps(l):
                yield l,n


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

    TODO: support chromosome=None

    """

    def __init__(self, fasta_file, chromosome, start=None, end=None, location=None, width=1, step=1, file_format=None,
                 fillvalue='N', padding=False):
        super().__init__(fasta_file, chromosome, start, end, location, file_format, per_position=True)
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

    def __iter__(self) -> (gi, str):
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
            yield self.location, dat
            pos1 += self.step


class CHR_PREFIX_OPERATOR(Enum):
    ADD = 1  # add chr prefix
    DEL = 2  # remove chr prefix


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
                 pos_indices=[0, 1, 2],
                 per_position=False):

        super().__init__(file=tabix_file, chromosome=chromosome, start=start, end=end, region=region,
                         file_format='tsv', per_position=per_position)
        self.chr_prefix_operator = chr_prefix_operator  # chr_prefix_operator can be 'add_chr', 'drop_chr' or None
        self.coord_inc = coord_inc
        self.pos_indices = pos_indices
        self.start = 1 if (self.start is None) else max(1, self.start)  # 0-based coordinates in pysam!

    def escape_chr(self, reference):
        if (reference is None) or self.chr_prefix_operator is None:
            return reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.ADD:
            return 'chr' + reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.DEL:
            return reference[3:]

    def unescape_chr(self, reference):
        if (reference is None) or self.chr_prefix_operator is None:
            return reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR.DEL:
            return 'chr' + reference
        elif self.chr_prefix_operator == CHR_PREFIX_OPERATOR:
            return reference[3:]

    def __iter__(self) -> (gi, str):
        for row in self.file.fetch(reference=self.escape_chr(self.chromosome),
                                   start=self.start - 1,  # 0-based coordinates in pysam!
                                   end=self.end,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome = self.unescape_chr(row[self.pos_indices[0]])
            start = int(row[self.pos_indices[1]]) + self.coord_inc[0]
            end = int(row[self.pos_indices[2]]) + self.coord_inc[1]
            self.location = gi(chromosome, start, end)
            self.stats['n_rows', chromosome] += 1
            yield self.location, tuple(row)

class BedGraphIterator(TabixIterator):
    """
        Iterates a bedgraph file and yields float values
        If a strand is passed, all yielded intervals will have this strand assigned.
    """
    def __init__(self, bedgraph_file, chromosome=None, start=None, end=None,
                 region=None, chr_prefix_operator=None, per_position=False,
                 strand=None):
        super().__init__(tabix_file=bedgraph_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=per_position, coord_inc=[1, 0], pos_indices=[0, 1, 2])
        self.strand=strand
    def __iter__(self) -> (gi, str):
        for loc, t in super().__iter__():
            loc.strand=self.strand
            yield loc, float(t[3])

class BedIterator(TabixIterator):
    """
        Iterates a BED file and yields 1-based coordinates and pysam BedProxy objects
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asBed
    """
    def __init__(self, bed_file, chromosome=None, start=None, end=None,
                 region=None, chr_prefix_operator=None, per_position=False):
        assert guess_file_format(bed_file)=='bed', f"expected BED file but guessed file format is {guess_file_format(bed_file)}"
        super().__init__(tabix_file=bed_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=per_position, coord_inc=[1, 0], pos_indices=[0, 1, 2])
    def __iter__(self) -> (gi, str):
        for bed in self.file.fetch(reference=self.escape_chr(self.chromosome),
                                   start=self.start - 1,  # 0-based coordinates in pysam!
                                   end=self.end,
                                   parser=pysam.asBed()):  # @UndefinedVariable
            chromosome = self.unescape_chr(bed.contig)
            start = bed.start + self.coord_inc[0]
            end = bed.end + self.coord_inc[1]
            strand = bed.strand if 'strand' in bed else None
            self.location = gi(chromosome, start, end, strand)
            self.stats['n_rows', chromosome] += 1
            yield self.location, bed

class GFF3Iterator(TabixIterator):
    """
        Iterates a GTF/GFF3 file and yields 1-based coordinates and dicts containing key/value pairs parsed from
        the respective info sections. The feature_type, source, score and phase fields from the GFF/GTF entries are
        copied to this dict (NOTE: attribute fields with the same name will be overloaded).
        @see https://pysam.readthedocs.io/en/latest/api.html#pysam.asGTF
    """
    def __init__(self, gtf_file, chromosome=None, start=None, end=None,
                 region=None, chr_prefix_operator=None, per_position=False):
        super().__init__(tabix_file=gtf_file, chromosome=chromosome, start=start, end=end, region=region,
                         per_position=per_position, coord_inc=[0, 0], pos_indices=[0, 1, 2])
        self.file_format=guess_file_format(gtf_file)
        assert self.file_format in ['gtf', 'gff'], f"expected GFF3/GTF file but guessed file format is {self.file_format}"
    def __iter__(self) -> (gi, str):
        for row in self.file.fetch(reference=self.escape_chr(self.chromosome),
                                   start=self.start - 1,  # 0-based coordinates in pysam!
                                   end=self.end,
                                   parser=pysam.asTuple()):  # @UndefinedVariable
            chromosome, source, feature_type, start, end, score, strand, phase, info = row
            self.location = gi(self.unescape_chr(chromosome), int(start) + self.coord_inc[0], int(end) + self.coord_inc[1], strand)
            info = parse_gff_info(info, self.file_format)
            info['feature_type'] = None if feature_type=='.' else feature_type
            info['source'] = None if source=='.' else source
            info['score'] = None if score=='.' else float(score)
            info['phase'] = None if phase=='.' else int(phase)
            self.stats['n_rows', chromosome] += 1
            yield self.location, info

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

    def __init__(self, df, feature, coord_columns=['Chromosome', 'Start', 'End'], is_sorted=False, per_position=False):
        self.stats = Counter()
        self.file = None
        self.location = None
        self.df = df if is_sorted else df.sort_values(['Chromosome', 'Start'])
        self.feature = feature
        self.coord_columns = coord_columns
        self.per_position=per_position

    def __iter__(self) -> (gi, str):
        for _, row in self.df.iterrows():
            chromosome = row[self.coord_columns[0]]
            start = row[self.coord_columns[1]]
            end = row[self.coord_columns[2]]
            self.location = gi(chromosome, start, end)
            self.stats['n_rows', chromosome] += 1
            yield self.location, row[self.feature]

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
                 report_mismatches=False, min_base_quality=0):
        super().__init__(bam_file, chromosome, start, end, location, file_format, per_position=False)
        self.min_mapping_quality = min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters = tag_filters
        self.report_mismatches = report_mismatches
        self.min_base_quality = min_base_quality

    def __iter__(self) -> (gi, str):
        md_check = False
        for r in self.file.fetch(contig=self.chromosome,
                                 start=self.start,
                                 end=self.end,
                                 until_eof=True):
            self.location = gi(r.reference_name, r.reference_start + 1, r.reference_end, '-' if r.is_reverse else '+')
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
                yield self.location, (r, mm) # yield read/mismatch tuple
            else:
                yield self.location, r


class FastPileupIterator(LocationIterator):
    """
        Fast pileup iterator that yields a complete pileup (w/o insertions) over a set of genomic positions. This is
        more lightweight and considerably faster than pysams pileup() but klacks some features (such as 'ignore_overlaps'
        or 'ignore_orphans'). By default it basically reports what is seen in the default IGV view.


        :parameter reported_positions  either a range (start/end) or a set of genomic positions for which counts will be reported.
        :parameter min_base_quality filters pileup based on minimum per-base quality
        :parameter max_depth restricts maximum pileup depth.

        :return A base:count Counter object. Base is 'None' for deletions.
        Note that this implementation does not support 'ignore_overlaps' or 'ignore_orphans' options like pysam.pileup()
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
        super().__init__(bam_file, chromosome, self.start, self.end, file_format, per_position=True)
        self.min_mapping_quality = min_mapping_quality
        self.flag_filter = flag_filter
        self.max_span = max_span
        self.tag_filters = tag_filters
        self.min_base_quality = min_base_quality
        self.max_depth = max_depth
        self.count_dict = Counter()

    def __iter__(self) -> (gi, str):
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
            yield self.location, self.count_dict[gpos] if gpos in self.count_dict else Counter()


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
        self.per_position=self.orgit.per_position

    def __iter__(self) -> (gi, (tuple, tuple)):
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
            yield mloc, (locations, values)

    def close(self):
        try:
            self.orgit.close()
        except AttributeError:
            pass


class SyncPerPositionIterator(LocationIterator):
    """ Synchronizes the passed location iterators by genomic location and yields
        genomic positions, interval, values and indices of the respective originating iterators.
        Expects coordinate-sorted location iterators!
        Example:
            for pos,(intervals, values, indices) in SyncPerPositionIterator([it1, it2, it3]):
                ...
    """
    def __init__(self, iterables, refdict=None):
        """
        :parameter iterables a list of location iterators
        """
        self.iterables = iterables
        for it in iterables:
            assert isinstance(it, LocationIterator), "Only implemented for LocationIterators"
        self.iterators = [peekable(it) for it in iterables]  # make peekable iterators
        if refdict is None:
            self.refdict = ReferenceDict.merge_and_validate(*[it.refdict for it in iterables])
            if self.refdict is None:
                print("WARNING: could not determine refdict from iterators: using alphanumerical chrom order.")
            else:
                print("Iterating merged refdict:", self.refdict)
        else:
            self.refdict = refdict
            print("Iterating refdict:", self.refdict)
        self.chr2gi = {} # a str->list dict storing locations per chromosome
        # defined chrom sort order or create on the fly
        self.chroms=SortedSet() if self.refdict is None else self.refdict.keys() # chrom must be in same order as in iterator!
        # load first pos
        self.update()
    def update(self):
        """ Update until position """
        for idx, it in enumerate(self.iterators):
            while True:
                nxt = it.peek(default=None)  # loc, value
                if nxt is None: break # iterator exhausted
                loc, data = next(it)
                loc.data = data # piggyback data and iterator index
                loc.idx = idx
                if loc.chromosome not in self.chr2gi:
                    self.chr2gi[loc.chromosome] = SortedList()
                    if not self.refdict:
                        self.chroms.add(loc.chromosome) # add chrom on the fly as we don't know them for iterators w/o refdict
                self.chr2gi[loc.chromosome].add(loc)
                if not loc.left_match(self.chr2gi[loc.chromosome][0]): break # done
    def pop_leftmost(self, chrom):
        """
            return a list with all intervals in chr2gi[chrom] that share the same start coordinate.
            The items are removed from chr2gi[chrom]
        """
        if chrom not in self.chr2gi or len(self.chr2gi[chrom])==0:
            return None
        s0 = self.chr2gi[chrom][0].start
        ret=[self.chr2gi[chrom].pop(0)]
        for _ in range(len(self.chr2gi[chrom])):
            if self.chr2gi[chrom][0].start==s0:
                ret.append(self.chr2gi[chrom].pop(0))
            else: break
        return ret
    def iterate_blocked(self) -> (gi, tuple):
        for chrom in self.chroms:
            while True:
                current = self.pop_leftmost(chrom) # pop leftmost intervals sharing same start position
                if current is None:
                    break
                self.update()
                yield current # current is a list of intervals with same chrom + left endpoint
    def __iter__(self) -> (gi, tuple):
        current=[]
        b=None
        for a, b in pairwise(self.iterate_blocked()): # a+b: non-empty lists with shared chromosome and left-pos
            current.extend(a)
            a0,b0=a[0],b[0]
            end=b0.start-1 if a0.chromosome==b0.chromosome else max(x.end for x in current)
            for pos in gi(a0.chromosome, a0.start, end): # iterate per position until next startpoint
                current=[x for x in current if x.end>=pos.start] # drop passed intervals
                assert len(current)==len([x for x in current])
                yield pos, ([x for x in current], [p.idx for p in current], [p.data for p in current])
            if a0.chromosome!=b0.chromosome:
                current=[] # chrom change
        if b is not None: # remainig intervals
            current.extend(b)
            for pos in gi(b0.chromosome, b0.start, max(x.end for x in current)):
                current = [x for x in current if x.end >= pos.start]  # drop passed intervals
                assert len(current) == len([x for x in current])
                yield pos, ([x for x in current], [p.idx for p in current], [p.data for p in current])

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
    Iterates a fastq file and yields FastqRead onjects (named tuples containing sequence name, sequence and quality
    string (omitting line 3 in the FASTQ file)).

    Note that this is not a location iterator.
    """

    def __init__(self, fastq_file):
        self.file = fastq_file
        self.per_position=False
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
