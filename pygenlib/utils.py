"""
General (low-level) utility methods



@author: niko.popitsch@univie.ac.at
"""
import gzip
import math
import numbers
import os
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, abc
from dataclasses import dataclass, field
from enum import IntEnum
from functools import reduce
from itertools import zip_longest
from pathlib import Path

import h5py
import pysam
from Bio import pairwise2
from tqdm import tqdm


# ------------------------------------------------------------------------
# genomic interval implementation
# ------------------------------------------------------------------------

@dataclass(frozen=True)
class gi:
    """
        Genomic intervals (gi) in pygenlib are inclusive and 1-based.
        Points are represented by intervals with same start+stop coordinate, empty intervals by passing start>end
        coordinates (e.g., gi('chr1', 1,0).is_empty() -> True).

        GIs are implemented as frozen(immutable) dataclasses and can be used, e.g., as keys in a dict.
        They can be instantiated by passing chrom/start/stop coordinates or can be parsed form a string.

        Intervals can be stranded.
        Using None for each component of the coordinates is allowed to represent unbounded intervals

        sorting
        -------
        Chromosomes group intervals and the order of intervals from different groups (chromosomes) is left undefined.
        To sort also by chromosome, one can use a @ReferenceDict which defined the chromosome order:
        sorted(gis, key=lambda x: (refdict.index(x.chromosome), x)) # note that the index of chromosome 'None' is always 0
        TODO: test overlap/etc for empty intervals (start>end)
    """
    chromosome: str = None
    start: int = -math.inf
    end: int = math.inf
    strand: str = None

    def __post_init__(self):
        """ Some sanity checks and default values """
        object.__setattr__(self, 'start', -math.inf if self.start is None else self.start)
        object.__setattr__(self, 'end', math.inf if self.end is None else self.end)
        object.__setattr__(self, 'strand', self.strand if self.strand != '.' else None)
        assert isinstance(self.start, numbers.Number)
        assert isinstance(self.end, numbers.Number)
        assert self.strand in [None, '+', '-']

    def __len__(self):
        if self.is_empty():
            return 0
        return self.end - self.start + 1
    @classmethod
    def from_str(cls, loc_string):
        """ Parse from <chr>:<start>-<end> (<strand>). Strand is optional"""
        pattern = re.compile("(\w+):(\d+)-(\d+)(?:[\s]*\(([+-])\))?$")
        match=pattern.findall(loc_string.strip().replace(',', '')) # convenience
        if len(match)==0:
            return None
        chromosome, start, end, strand = match[0]
        strand = None if strand=='' else strand
        return cls(chromosome, int(start), int(end), strand)
    def __repr__(self):
        return f"{self.chromosome}:{self.start}-{self.end}{'' if self.strand is None else f' ({self.strand})'}"

    def get_stranded(self, strand):
        """Get a new object with same coordinates; the strand will be set according to the passed variable."""
        return gi(self.chromosome, self.start, self.end, strand)
    def to_file_str(self):
        """ returns a sluggified string representation "<chrom>_<start>_<end>_<strand>"        """
        return f"{self.chromosome}_{self.start}_{self.end}_{'u' if self.strand is None else self.strand}"

    def is_empty(self):
        return self.start>self.end
    def is_stranded(self):
        return self.strand is not None

    def cs_match(self, other, strand_specific=False):
        """ True if this location is on the same chrom/strand as the passed one.
            will not compare chromosomes if they are unrestricted in one of the intervals.
            Empty intervals always return False hee
        """
        if strand_specific and self.strand != other.strand:
            return False
        if (self.chromosome) and (other.chromosome) and (self.chromosome!=other.chromosome):
            return False
        if self.is_empty():
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

    def overlaps(self, other, strand_specific=False) -> bool:
        """ Tests whether this interval overlaps the passed one.
            Supports unrestricted start/end coordinates and optional strand check
        """
        if not self.cs_match(other, strand_specific):
            return False
        return self.start <= other.end and other.start <= self.end

    def envelops(self, other, strand_specific=False) -> bool:
        """ Tests whether this interval envelops the passed one.
        """
        if not self.cs_match(other, strand_specific):
            return False
        return self.start <= other.start and self.end >= other.end

    def overlap(self, other, strand_specific=False) -> int:
        """Calculates the overlap with the passed one"""
        if not self.cs_match(other, strand_specific):
            return 0
        return min(self.end, other.end) - max(self.start, other.start) + 1.

    def split_coordinates(self) -> (str, int, int):
        return self.chromosome, self.start, self.end

    @classmethod
    def merge(cls, l):
        """ Merges a list of intervals.
            If intervals are not on the same chromosome or if strand is not matching, None is returned
            The resulting interval will inherit the chromosome and strand of the first passed one.
        """
        if l is None:
            return None
        if len(l) == 1:
            return l[0]
        merged = None
        for x in l:
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

    def distance(self, other,  strand_specific=False):
        """
            Distance to other interval.
            - None if chromosomes do not match
            - 0 if intervals overlap
            - negative if other < self
        """
        if self.cs_match(other, strand_specific=strand_specific):
            if self.overlaps(other):
                return 0
            return other.start-self.end if other>self else other.end-self.start
        return None
    def __iter__(self):
        for pos in range(self.start, self.end+1):
            yield gi(self.chromosome, pos, pos, self.strand)

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
        self.name=name
        self.fun_alias=fun_alias
        if fun_alias is not None:
            self.orig=d.copy()
            self.d={fun_alias(k):v for k,v in d.items()} # apply fun to keys
        else:
            self.orig=self
    def __getitem__(self, key):
        return self.d[key]
    def __len__(self):
        return len(self.d)
    def __iter__(self):
        return iter(self.d)
    def __repr__(self):
        #return f"Refset{'' if self.name is None else self.name} (len: {len(self.d)})"
        return f"Refset (size: {len(self.d.keys())}): {self.d.keys()}{f' (aliased from {self.orig.keys()})' if self.fun_alias else ''}, {self.d.values()} name: {self.name} "
    def alias(self, chr):
        if self.fun_alias:
            return self.fun_alias(chr)
        return chr
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

    @classmethod
    def merge_and_validate(cls, *refsets, check_order=False, included_chrom=[]):
        """
            Checks whether the passed reference sets are compatible and returns the
            merged reference set containing the intersection of common references
        """
        refsets = [r for r in refsets if r is not None]
        if len(refsets) == 0:
            return None
        # intersect all contig lists while preserving order (set.intersection() or np.intersect1d() do not work!)
        shared_ref = {k: None for k in intersect_lists(*[list(r.keys()) for r in refsets], check_order=check_order) if
                      (len(included_chrom) == 0) or (k in included_chrom)}
        # check whether contig lengths match
        for r in refsets:
            for contig, oldlen in shared_ref.items():
                newlen = r.get(contig)
                if newlen is None:
                    continue
                if oldlen is None:
                    shared_ref[contig] = newlen
                else:
                    assert oldlen == newlen, f"Incompatible lengths for contig ({oldlen}!={newlen}) when comparing refsets {refsets}"
        return ReferenceDict(shared_ref, name=','.join([r.name for r in refsets]), fun_alias=None )

# --------------------------------------------------------------
# Commandline and config handling
# --------------------------------------------------------------

def parse_args(args, parser_dict, usage):
    """
        Parses commandline arguments.
        expects an args list that contains
        1) the calling script name
        2) the respective mode
        3) commandline arguments
    """
    MODES = parser_dict.keys()
    if len(args) <= 1 or args[1] in ['-h', '--help']:
        print(usage.replace("MODE", "[" + ",".join(MODES) + "]"))
        sys.exit(1)
    mode = args[1]
    if mode not in MODES:
        print("No/invalid mode %s provided. Please use one of %s" % (mode, ",".join(MODES)))
        print(usage)
        sys.exit(1)
    usage = usage.replace("MODE", mode)
    return mode, parser_dict[mode].parse_args(args[2:])


def get_config(config, keys, default_value=None, required=False):
    """
        Gets a configuration value from a config dict (e.g., loaded from JSON).
        Keys can be a list of keys (that will be traversed) or a single value.
        If the key is missing and required is True, an error will be thrown. Otherwise, the configured default value will be returned.
        Example:
            cmd_bcftools = get_config(config, ['params','cmd','cmd_bcftools], required=True)
            cmd_bcftools = get_config(config, ['params/cmd/cmd_bcftools], required=True)
            threads = get_config(config, 'threads', default_value=1)
        TODO: possibly replace with omegaconf
    """
    if isinstance(keys, str):  # handle single strings
        keys = keys.split('/')
    d = config
    for k in keys:
        if k not in d:
            assert (not required), 'Mandatory config path "%s" missing' % ' > '.join(keys)
            return default_value
        d = d[k]
    return d


def ensure_outdir(outdir=None) -> os.PathLike:
    """ Ensures that the configured output dir exists (will use current dir if none provided) """
    outdir = os.path.abspath(outdir if outdir else os.getcwd())
    if not outdir.endswith("/"):
        outdir += "/"
    if not os.path.exists(outdir):
        print("Creating dir " + outdir)
        os.makedirs(outdir)
    return outdir


# --------------------------------------------------------------
# Collection helpers
# --------------------------------------------------------------


def check_list(lst, mode='inc1') -> bool:
    """
        Tests whether the numeric (comparable) items in a list are
        mode=='inc': increasing
        mode=='dec': decreasing
        mode=='inc1': increasing by one
        mode=='dec1': decreasing by one
    """
    if mode == 'inc':
        return all(x < y for x, y in zip(lst, lst[1:]))
    elif mode == 'inc1':
        return all(x + 1 == y for x, y in zip(lst, lst[1:]))
    elif mode == 'dec':
        return all(x > y for x, y in zip(lst, lst[1:]))
    elif mode == 'dec1':
        return all(x - 1 == y for x, y in zip(lst, lst[1:]))
    return None


def split_list(l, n, is_chunksize=False) -> list:
    """
        Splits a list into sublists.
        if is_chunksize is True: splits into chunks of length n
        if is_chunksize is False: splits it into n (approx) equal parts
        If l is not a list it is tried to convert it.
        returns a generator.
        example:
            list(split_list(range(0,10), 3))
            list(split_list(range(0,10), 3, is_chunksize=True))

        TODO: replace by more_iterools chunked and chunked_even
    """
    if not isinstance(l, list):
        l = list(l)
    if is_chunksize:
        return [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n)]
    else:
        k, m = divmod(len(l), n)
        return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def intersect_lists(*lists, check_order=False) -> list:
    """ Intersects lists (iterables) while preserving order.
        Order is determined by the last provided list.
        If check_order is True, the order of all input lists wrt. shared elements is asserted.
        usage:
            intersect_lists([1,2,3,4],[3,2,1]) # [3,2,1]
            intersect_lists(*list_of_lists)
            intersect_lists([1,2,3,4],[1,4],[3,1], check_order=True)
            intersect_lists((1,2,3,5),(1,3,4,5)) # [1,3,5]
    """
    if len(lists) == 0:
        return []

    def intersect_lists_(list1, list2):
        return list(filter(lambda x: x in list1, list2))

    isec = reduce(intersect_lists_, lists)
    for l in lists:
        if check_order:
            assert [x for x in l if x in isec] == isec, f"Input list have differing order of shared elements {isec}"
        elif [x for x in l if x in isec] != isec:
            print(f"WARN: Input list have differing order of shared elements {isec}")
    return isec


def cmp_sets(a, b) -> (bool, bool, bool):
    """ Set comparison. Returns shared, unique(a) and unique(b) items """
    return a & b, a.difference(b), b.difference(a)


# --------------------------------------------------------------
# I/O handling
# --------------------------------------------------------------

def grouper(iterable, n, fillvalue=None):
    """ Groups n lines into a list """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def to_str(*args, sep=',', na='NA') -> str:
    """
        Converts an object to a string representation. Iterables with be joined by the configured separator.
        Objects with zero length or None will be converted to the configured 'NA' representation.
        NOTE: different from dm-tree.flatten() or treevalue.flatten() but much slower?
        examples:
            to_str([12,[None,[1,'',[]]]]) # '12,NA,1,NA,NA'
    """
    if (args is None) or (len(args) == 0):
        return na
    if len(args) == 1:
        args = args[0]
    if (args is None):
        return na
    if hasattr(args, '__len__') and callable(getattr(args, '__len__')) and len(args) == 0:
        return na
    if isinstance(args, str):
        return args
    if hasattr(args, '__len__') and hasattr(args, '__iter__') and callable(getattr(args, '__iter__')):
        return sep.join([to_str(x) for x in args])
    return str(args)


def write_data(dat, out=None, sep='\t', na='NA'):
    """
        Write data to TSV file. Values will be written via thee to_str() method (i.e., None will become 'NA').
        Collections will be written as comma-separated strings, nested structures will be flattened.
        If out is None, the respective string will be written, Otherwise it will be written to the respective output handle.
    """
    s = sep.join([to_str(x, na=na) for x in dat])
    if out is None:
        return s
    print(s, file=out)


def format_fasta(string, ncol=80) -> str:
    """ Format a string for FASTA files """
    return '\n'.join(string[i:i + ncol] for i in range(0, len(string), ncol))


def dir_tree(root: Path, prefix: str = '', space='    ', branch='│   ', tee='├── ', last='└── ', max_lines=10,
             glob=None):
    """ A recursive generator yielding a visual tree structure line by line.
        max_lines: maximum yielded lines per directory.
        glob: optional glob param to filter. Example: glob='**/*.py" to list all .py files
        @see https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    contents = list(root.iterdir()) if (glob is None) else list(root.glob(glob))
    if len(contents) > max_lines:
        contents = contents[:max_lines] + [Path('...')]
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from dir_tree(path, prefix=prefix + extension, max_lines=max_lines, glob=glob)


def print_dir_tree(root: Path, max_lines=10, glob=None):
    if isinstance(root, str):
        root = Path(root)
    for line in dir_tree(root, max_lines=max_lines, glob=glob):
        print(line)


def count_lines(file) -> int:
    """Counts lines in (gzipped) files. Slow."""
    if (file.endswith(".gz")):
        with gzip.open(file, 'rb') as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else:
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        return i + 1


def gunzip(in_file, out_file=None) -> str:
    """ Gunzips a file and returns the filename of the resulting file """
    assert in_file.endswith(".gz"), "in_file must be a .gz file"
    if out_file is None:
        out_file = in_file[:-3]
    with gzip.open(in_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_file


def slugify(value, allow_unicode=False) -> str:
    """
    Slugify a string (e.g., to get valid filenames w/o extensions).
    @see https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '_', value).strip('_')


# --------------------------------------------------------------
# Sequence handling
# --------------------------------------------------------------

class ParseMap(dict):
    """
        Extends default dict to return 'missing char' for missing keys (similar to defaultdict, but doies not enter new values).
        Should be used with translate()Should be used with translate()
    """

    def __init__(self, *args, **kwargs):
        self.missing_char = kwargs.get('missing_char', 'N')
        kwargs.pop('missing_char', None)
        self.update(*args, **kwargs)

    def __missing__(self, key):
        return self.missing_char


TMAP = {'dna': ParseMap({x: y for x, y in zip(b'ATCGatcgNn', b'TAGCTAGCNN')}, missing_char='*'),
        'rna': ParseMap({x: y for x, y in zip(b'AUCGaucgNn', b'UAGCUAGCNN')}, missing_char='*')
        }


def reverse_complement(seq, tmap='dna') -> str:
    """
        Calculate reverse complement DNA/RNA sequence.
        Returned sequence is uppercase, N's are kept.
        seq_type can be 'dna' (default) or 'rna'
    """
    if isinstance(tmap, str):
        tmap = TMAP[tmap]
    return seq[::-1].translate(tmap)


def complement(seq, tmap='dna') -> str:
    """
        Calculate complement DNA/RNA sequence.
        Returned sequence is uppercase, N's are kept.
        seq_type can be 'dna' (default) or 'rna'
    """
    if isinstance(tmap, str):
        tmap = TMAP[tmap]
    return seq.translate(tmap)


def pad_n(seq, minlen, padding_char='N') -> str:
    """
        Adds up-/downstream padding with the configured padding character to ensure a given minimum length of the passed sequence
    """
    ret = seq
    if (len(ret) < minlen):
        pad0 = padding_char * int((minlen - len(ret)) / 2)
        pad1 = padding_char * int(minlen - (len(ret) + len(pad0)))
        ret = ''.join([pad0, ret, pad1])
    return (ret)


def rnd_seq(n, alpha='ACTG', m=1):
    """
        Creates m random sequence of length n using the provided alphabet (default: DNA bases).
        To use different character frequencies, pass each character as often as expected in the frequency distribution.
        Example:    rnd_seq(100, 'GC'* 60 + 'AT' * 40, 5) # 5 sequences of length 100 with 60% GC
    """
    if m <= 0:
        return None
    res = [''.join(random.choice(alpha) for _ in range(n)) for _ in range(m)]
    return res if m > 1 else res[0]


def count_gc(s) -> (int, float):
    """
        Counts number of G+C bases in string.
         Returns the number of GC bases and the length-normalized fraction.
    """
    ngc = s.count('G') + s.count('C')
    return ngc, ngc / len(s)


def count_rest(s, rest=['GGTACC', 'GAATTC', 'CTCGAG', 'CATATG', 'ACTAGT']) -> int:
    """ Counts number of restriction sites, see https://portals.broadinstitute.org/gpp/public/resources/rules """
    return sum([r in s for r in rest])


def longest_hp_gc_len(seq) -> (int, int):
    """
        Counts HP length (any allele) from start.
        This method counts any character including N's
    """
    c = Counter()
    last_char = None
    for base in seq:
        if last_char == base:  # hp continue
            c['hp'] += 1
        else:
            if c['hp'] > c['longest_hp']:
                c['longest_hp'] = c['hp']
            c['hp'] = 1
        if base in ['G', 'C']:
            c['gc'] += 1
        else:
            if c['gc'] > c['longest_gc']:
                c['longest_gc'] = c['gc']
            c['gc'] = 0
        last_char = base
    if c['hp'] > c['longest_hp']:
        c['longest_hp'] = c['hp']
    if c['gc'] > c['longest_gc']:
        c['longest_gc'] = c['gc']
    # get max base:return max(c, key=c.get)
    return c['longest_hp'], c['longest_gc']


def longest_GC_len(seq) -> int:
    """ Counts HP length (any allele) from start """
    c = Counter()
    last_char = None
    for base in seq:
        if last_char in ['G', 'C']:
            c['GC'] += 1
        else:
            c['GC'] = 1
        last_char = base
    # get max base:return max(c, key=c.get)
    return max(c.values())


def align_sequence(query, target, print_alignment=False) -> (float, int, int):
    """
        Global alignment of query to target sequence with default scoring.
        Returns a length-normalized alignment score and the start and end positions of the alignment.
        TODO replace with Bio.Align.PairwiseAligner and expose parameters
    """
    aln = pairwise2.align.globalxs(  # globalxs(sequenceA, sequenceB, open, extend) -> alignments
        query,
        target,
        -2,
        -1,
        penalize_end_gaps=(False, False),  # do not penalize starting/ending gaps
        score_only=False,
        one_alignment_only=True)[0]
    _, _, score, _, _ = aln
    score = score / len(query)
    startpos = len(aln[0]) - len(aln[0].lstrip('-'))
    endpos = len([x for x in aln[1][:len(aln[0].rstrip('-'))] if x != '-'])
    if print_alignment:
        print(pairwise2.format_alignment(*aln))
    return score, startpos, endpos


def find_all(a, sub) -> int:
    """Finds all indices of the passed substring"""
    start = 0
    while True:
        start = a.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def kmer_search(seq, kmer_set, include_revcomp=False) -> dict:
    """ Exact kmer search. Can include revcomp if configured """
    ret = dict()
    for kmer in kmer_set:
        ret[kmer] = list(find_all(seq, kmer))
        if include_revcomp:
            ret[kmer].extend(list(find_all(seq, reverse_complement(kmer))))
    return ret


def find_gpos(genome_fa, kmers, included_chrom=None) -> Counter:
    """ Returns a dict that maps the passed kmers to their (exact match) genomic positions (1-based).
        Positions are returned as (chr, pos1) tuples.
        included_chromosomes (list): if configured, then only the respective chromosomes will be considered.
    """
    fasta = pysam.Fastafile(genome_fa)
    ret = Counter()
    chroms = fasta.references if included_chrom is None else set(fasta.references) & set(included_chrom)
    for c in tqdm(chroms, total=len(chroms), desc='Searching chromosome'):
        pos = kmer_search(fasta.fetch(c), kmers)
        for kmer in kmers:
            if kmer not in ret:
                ret[kmer] = []
            for pos0 in pos[kmer]:
                ret[kmer].append((c, pos0 + 1, '+'))
    return ret


# --------------------------------------------------------------
# genomics helpers
# --------------------------------------------------------------


def parse_gff_attributes(info, fmt='gff3'):
    """ parses GFF3/GTF info sections """
    if '#' in info: # remove optional comment section (e.g., in flybase gtf)
        info=info.split('#')[0].strip()
    if fmt.lower() == 'gtf':
        return {k: v.translate({ord(c): None for c in '"'}) for k, v in
                [a.strip().split(' ', 1) for a in info.split(';') if ' ' in a]}
    return {k: v for k, v in [a.split('=') for a in info.split(';') if '=' in a]}


def bgzip_and_tabix(in_file, out_file=None, create_index=True, del_uncompressed=True,
                    preset='auto', seq_col=0, start_col=1, end_col=1, line_skip=0, zerobased=False, csi=False):
    """
        Will BGZIP the passed file and creates a tabix index with the given params if create_index is True
        presets:
            'gff' : (TBX_GENERIC, 1, 4, 5, ord('#'), 0),
            'bed' : (TBX_UCSC, 1, 2, 3, ord('#'), 0),
            'psltbl' : (TBX_UCSC, 15, 17, 18, ord('#'), 0),
            'sam' : (TBX_SAM, 3, 4, 0, ord('@'), 0),
            'vcf' : (TBX_VCF, 1, 2, 0, ord('#'), 0),
            'auto': guess from file extension
    """
    if out_file is None:
        out_file = in_file + '.gz'
    assert out_file.endswith(".gz"), "out_file must be a .gz file"
    pysam.tabix_compress(in_file, out_file, force=True)  # @UndefinedVariable
    if create_index:
        if preset == 'auto':
            preset = guess_file_format(in_file)
            if preset == 'gtf':
                preset = 'gff'  # pysam default
            if preset not in ['gff','bed','psltbl','sam','vcf']: # currerntly supported by tabix
                preset = None
            print(f"Detected file format for index creation: {preset}")
        pysam.tabix_index(out_file, preset=preset, force=True, seq_col=seq_col, start_col=start_col, end_col=end_col,
                          meta_char='#', line_skip=line_skip, zerobased=zerobased)  # @UndefinedVariable
    if del_uncompressed:
        os.remove(in_file)


def count_reads(in_file):
    """ Counts reads in different file types """
    ftype = guess_file_format(in_file)
    if ftype == 'fastq':
        return count_lines(in_file) / 4.0
    elif ftype == 'sam':
        raise NotImplementedError("SAM/BAM file read counting not implemeted yet!")
    else:
        raise NotImplementedError(f"Cannot count reads in file of type {ftype}.")


# --------------------------------------------------------------
# genomics helpers :: SAM/BAM specific
# --------------------------------------------------------------
# BAM flags, @see https://broadinstitute.github.io/picard/explain-flags.html
class BAM_FLAG(IntEnum):
    BAM_FPAIRED = 0x1  # the read is paired in sequencing, no matter whether it is mapped in a pair
    BAM_FPROPER_PAIR = 0x2  # the read is mapped in a proper pair
    BAM_FUNMAP = 0x4  # the read itself is unmapped; conflictive with BAM_FPROPER_PAIR
    BAM_FMUNMAP = 0x8  # the mate is unmapped
    BAM_FREVERSE = 0x10  # the read is mapped to the reverse strand
    BAM_FMREVERSE = 0x20  # the mate is mapped to the reverse strand
    BAM_FREAD1 = 0x40  # this is read1
    BAM_FREAD2 = 0x80  # this is read2
    BAM_FSECONDARY = 0x100  # not primary alignment
    BAM_FQCFAIL = 0x200  # QC failure
    BAM_FDUP = 0x400  # optical or PCR duplicate
    BAM_SUPPLEMENTARY = 0x800  # optical or PCR duplicate


# Default BAM flag filter (3844) as used, e.g., in IGV
DEFAULT_FLAG_FILTER = BAM_FLAG.BAM_FUNMAP | BAM_FLAG.BAM_FSECONDARY | BAM_FLAG.BAM_FQCFAIL | BAM_FLAG.BAM_FDUP | BAM_FLAG.BAM_SUPPLEMENTARY


@dataclass
class TagFilter:
    """
        Filter reads if the specified tag has one of the provided filter_values.
        Can be inverted for filtering if specified values is found.
    """
    tag: str
    filter_values: field(default_factory=list)
    filter_if_no_tag: bool = False
    inverse: bool = False

    def filter(self, r):
        if r.has_tag(self.tag):
            value_exists = r.get_tag(self.tag) in self.filter_values
            return value_exists != self.inverse
        else:
            return self.filter_if_no_tag
        return False


def get_softclip_seq(read: pysam.AlignedSegment):
    """Extracts soft-clipped sequences from the passed read"""
    left, right = None, None
    pos = 0
    for i, (op, l) in enumerate(read.cigartuples):
        if (i == 0) & (op == 4):
            left = l
        if (i == len(read.cigartuples) - 1) & (op == 4):
            right = l
        pos += l
    return left, right

def toggle_chr(s):
    """
        Simple function that toggle 'chr' prefixes. If the passed reference name start with 'chr',
        then it is removed, otherwise it is added. If None is passed, None is returned.
        Can be used as fun_alias function.
    """
    if s is None:
        return None
    if isinstance(s,str) and s.startswith('chr'):
        return s[3:]
    else:
        return f'chr{s}'


def get_reference_dict(fh, fun_alias=None) -> dict:
    """ Extracts chromosome names, order and (where possible) length from pysam objects.

    Parameters
    ----------
    fh : pysam object
    fun_alias : aliasing functions (see RefDict)

    Returns
    -------
    dict: chromosome name to length

    Raises
    ------
    NotImplementedError
        if input type is not supported yet
    """
    if isinstance(fh, str):
        fh=open_file_obj(fh)
    if isinstance(fh, pysam.Fastafile):  # @UndefinedVariable
        return ReferenceDict({c: fh.get_reference_length(c) for c in fh.references},
                             name=f'References from FASTA file {fh.filename}', fun_alias=fun_alias)
    elif isinstance(fh, pysam.AlignmentFile):  # @UndefinedVariable
        return ReferenceDict({c: fh.header.get_reference_length(c) for c in fh.references},
                             name=f'References from SAM/BAM file {fh.filename}',fun_alias=fun_alias)
    elif isinstance(fh, pysam.TabixFile):  # @UndefinedVariable
        return ReferenceDict({c: None for c in fh.contigs}, name=f'References from TABIX file {fh.filename}',
                             fun_alias=fun_alias)  # no ref length info in tabix
    elif isinstance(fh, pysam.VariantFile):  # @UndefinedVariable
        return ReferenceDict({c: fh.header.contigs.get(c).length for c in fh.header.contigs},
                             name=f'References from VCF file {fh.filename}',fun_alias=fun_alias)
    else:
        raise NotImplementedError(f"Unknown input object type {type(fh)}")

default_file_extensions={
    'fasta': ('.fa', '.fasta', '.fna', '.fa.gz', '.fasta.gz'),
    'sam': ('.sam'),
    'bam': ('.bam'),
    'tsv': ('.tsv', '.tsv.gz'),
    'bed': ('.bed', '.bed.gz', '.bedgraph', '.bedgraph.gz'),
    'vcf': ('.vcf', '.vcf.gz'),
    'bcf': ('.bcf'),
    'gff': ('.gff3', '.gff3.gz'),
    'gtf': ('.gtf', '.gtf.gz'),
    'fastq': ('.fq', '.fastq', '.fq.gz', '.fastq.gz'),
}

def guess_file_format(file_name, file_extensions=default_file_extensions):
    """
    Guesses the file format from the file extension
    :param file_name:
    :param file_extensions:
    :return:
    """
    # fn, ext = os.path.splitext(file_name.lower())
    # if ext == '.gz':
    #     fn, ext = os.path.splitext(fn)
    #     ext += '.gz'
    # return file_extensions.get(ext, )
    for ff, ext in file_extensions.items(): # TODO: make faster
        if file_name.endswith(ext):
            return ff
    return None


def open_file_obj(fh, file_format=None, file_extensions=default_file_extensions) -> (object):
    """ Opens a file object.

    If a pysam compatible file format was detected, the respcetive pysam object is instantiated.

    :parameter fh : str or file path object
    :parameter file_format : str
        Can be any <supported_formats> or None for auto-detection from filename (valid file extensions can be configured).

    :returns file_handle : instance (file/pysam object)
    """
    fh = str(fh)  # convert path to str
    if file_format is None:  # auto detect via file extension
        file_format = guess_file_format(fh, file_extensions)
    # instantiate pysam object
    if file_format == 'fasta':
        fh = pysam.Fastafile(fh)  # @UndefinedVariable
        was_opened = True
    elif file_format == 'sam':
        fh = pysam.AlignmentFile(fh, "r")  # @UndefinedVariable
        was_opened = True
    elif file_format == 'bam':
        fh = pysam.AlignmentFile(fh, "rb")  # @UndefinedVariable
        was_opened = True
    elif file_format == 'bed':
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == 'gtf':
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == 'gff':
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == 'tsv':
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == 'vcf':
        fh = pysam.VariantFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == 'bcf':
        fh = pysam.VariantFile(fh, mode="rb")  # @UndefinedVariable
    elif file_format == 'fastq':
        fh = gzip.open(fh, 'rb') if fh.endswith('.gz') else open(fh, mode="r")
    else:
        raise NotImplementedError(f"Unsupported input format for file {fh}")
    return fh



def get_covered_contigs(bam_files):
    """ Returns all contigs that have some coverage across a set of BAMs.

    Parameters
    ----------
    bam_files : str
        File paths of input BAM files

    Returns
    -------
    set
        A set of strings representing all contigs that have data in at least one of the BAM files.
    """
    covered_contigs = set()
    for b in bam_files:
        s = pysam.AlignmentFile(b, "rb")  # @UndefinedVariable
        covered_contigs.update([c for c, _, _, t in s.get_index_statistics() if t > 0])
    return covered_contigs


# --------------------------------------------------------------
# genomics helpers :: VCF specific
# --------------------------------------------------------------

def move_id_to_info_field(vcf_in, info_field_name, vcf_out, desc=None):
    """
        move all ID entries from a VCF to a new info field
    """
    if desc is None:
        desc = info_field_name
    vcf = pysam.VariantFile(vcf_in, 'r')
    header = vcf.header
    header.add_line("##cmd=move_id_to_info_field()")
    header.add_line("##INFO=<ID=%s,Number=1,Type=String,Description=\"%s\">" % (info_field_name, desc))
    out = pysam.VariantFile(vcf_out, mode="w", header=header)
    for record in vcf.fetch():
        record.info[info_field_name] = ','.join(record.id.split(';'))
        record.id = '.'
        written = out.write(record)
    out.close()


def add_contig_headers(vcf_in, ref_fasta, vcf_out):
    """
        Add missing contig headers to a VCF file
    """
    vcf = pysam.VariantFile(vcf_in, 'r')
    header = vcf.header
    fa = pysam.Fastafile(ref_fasta)
    for c in [c for c in fa.references if c not in header.contigs]:
        header.add_line("##contig=<ID=%s,length=%i" % (c, fa.get_reference_length(c)))
    with pysam.VariantFile(vcf_out, mode="w", header=header) as out:
        for record in vcf.fetch():
            written = out.write(record)


# --------------------------------------------------------------
# genomics helpers :: nanopore specific
# --------------------------------------------------------------

def _fast5_tree(h5node, prefix: str = '', space='    ', branch='│   ', tee='├── ', last='└── ', max_lines=10,
                show_attrs=True, show_values=True):
    """ Recursively yielding strings describing the structure of an h5 file """
    if hasattr(h5node, 'keys'):
        contents = list(h5node.keys())
        if len(contents) > max_lines:
            contents = contents[:max_lines] + [Path('...')]
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, key in zip(pointers, contents):
            attrs_str = ''
            if show_attrs and hasattr(h5node[key], 'attrs'):
                attrs_str = [f'{k}={v}' for k, v in zip(h5node[key].attrs.keys(), h5node[key].attrs.values())]
                if len(attrs_str) > 0:
                    attrs_str = ' {' + ','.join(attrs_str) + '}'
                else:
                    attrs_str = ''
            yield prefix + pointer + key + attrs_str
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from _fast5_tree(h5node[key], prefix=prefix + extension, max_lines=max_lines)


def print_fast5_tree(fast5_file, max_lines=10, n_reads=1, show_attrs=True):
    """
        Prints the structure of a fast5 file.
        example: /Volumes/groups/ameres/Niko/projects/Ameres/nanopore/data/nanocall_gfp/rawdata/singlesampleruns/wtgfpivt/1bc1/20230201_1500_MN32894_FAQ55498_125c10c7/fast5/FAQ55498_b5ea166b_0.fast5
    """
    with h5py.File(fast5_file, 'r') as f:
        for cnt, rn in enumerate(f.keys()):
            for line in _fast5_tree(f[rn], prefix=rn + ' ', max_lines=max_lines, show_attrs=show_attrs):
                print(line)
            print('---')
            if cnt + 1 >= n_reads:
                return


def get_read_attr_info(fast5_file, rn, path):
    """
        Returns a dict with attribute values. For debugging only.
        example: get_read_info(fast5_file, 'read_00262802-9463-45c8-b22d-f68d1047c6fc','Analyses/Basecall_1D_000/BaseCalled_template/Trace')
    """
    with h5py.File(fast5_file, 'r') as f:
        x = f[rn][path]
        return {k: v for k, v in zip(x.attrs.keys(), x.attrs.values())}


def get_bcgs(fast5_file):
    """
        Returns a list of basecall groups from the 1st read
    """
    with h5py.File(fast5_file, 'r') as f:
        first_rn = next(iter(f.keys()))
        return [a for a in list(f[first_rn]['Analyses']) if a.startswith("Basecall_")]

