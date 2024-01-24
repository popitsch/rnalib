"""
This module implements various general (low-level) utility methods

"""
import gzip
import os
import random
import re
import shutil
import ssl
import sys
import timeit
import unicodedata
import urllib.request
from collections import Counter, namedtuple, defaultdict
from dataclasses import dataclass, field
from functools import reduce
from itertools import groupby, zip_longest, islice
from pathlib import Path
from typing import Optional, List

import h5py
import mygene
import pandas as pd
import pysam
from IPython.core.display import HTML
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import rnalib


# --------------------------------------------------------------
# Commandline and config handling
# --------------------------------------------------------------


# --------------------------------------------------------------
# Collection helpers
# --------------------------------------------------------------


# --------------------------------------------------------------
# I/O handling
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sequence handling
# --------------------------------------------------------------


# --------------------------------------------------------------
# genomics helpers
# --------------------------------------------------------------


# --------------------------------------------------------------
# genomics helpers :: SAM/BAM specific
# --------------------------------------------------------------


# --------------------------------------------------------------
# genomics helpers :: VCF specific
# --------------------------------------------------------------


# --------------------------------------------------------------
# genomics helpers :: nanopore specific
# --------------------------------------------------------------


# --------------------------------------------------------------
# Utility functions for ipython notebooks
# --------------------------------------------------------------


def parse_args(args, parser_dict, usage):
    """
        Parses commandline arguments.
        expects an args list that contains
        1) the calling script name
        2) the respective mode
        3) commandline arguments
    """
    modes = parser_dict.keys()
    if len(args) <= 1 or args[1] in ['-h', '--help']:
        print(usage.replace("MODE", "[" + ",".join(modes) + "]"))
        sys.exit(1)
    mode = args[1]
    usage = usage.replace("MODE", mode)
    if mode not in modes:
        print("No/invalid mode %s provided. Please use one of %s" % (mode, ",".join(modes)))
        print(usage)
        sys.exit(1)
    return mode, parser_dict[mode].parse_args(args[2:])


def ensure_outdir(outdir=None) -> os.PathLike:
    """ Ensures that the configured output dir exists (will use current dir if none provided) """
    outdir = os.path.abspath(outdir if outdir else os.getcwd())
    if not outdir.endswith("/"):
        outdir += "/"
    if not os.path.exists(outdir):
        print("Creating dir " + outdir)
        os.makedirs(outdir)
    return outdir


def check_list(lst, mode='inc1') -> bool:
    """
        Tests whether the (numeric, comparable) items in a list are
        mode=='inc': increasing
        mode=='dec': decreasing
        mode=='inc1': increasing by one
        mode=='dec1': decreasing by one
        mode=='eq': all equal
    """
    if mode == 'inc':
        return all(x < y for x, y in zip(lst, lst[1:]))
    elif mode == 'inc1':
        return all(x + 1 == y for x, y in zip(lst, lst[1:]))
    elif mode == 'dec':
        return all(x > y for x, y in zip(lst, lst[1:]))
    elif mode == 'dec1':
        return all(x - 1 == y for x, y in zip(lst, lst[1:]))
    elif mode == 'eq':
        g = groupby(lst)  # see itertools
        return next(g, True) and not next(g, False)
    return None


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def split_list(lst, n, is_chunksize=False) -> list:
    """
        Splits a list into sublists.

        Parameters
        ----------
        lst: list
            input list to be split. If no list is passed, it is tried to convert via the list(...) method.
        n: int
            number of sublists
        is_chunksize: bool (default: False)
            if is_chunksize is True: splits into chunks of length n
            if is_chunksize is False: splits it into n (approx) equal parts

        Examples
        --------
        >>> list(split_list(range(0,10), 3))
        >>> list(split_list(range(0,10), 3, is_chunksize=True))

        Returns
        -------
        generator of lists

        Notes
        -----
        Will probably be replaced by more_iterools chunked and chunked_even
    """
    if not isinstance(lst, list):
        lst = list(lst)
    if is_chunksize:
        return [lst[i * n:(i + 1) * n] for i in range((len(lst) + n - 1) // n)]
    else:
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def intersect_lists(*lists, check_order=False) -> list:
    """ Intersects lists (iterables) while preserving order. Order is determined by the last provided list.
        If check_order is True, the order of all input lists wrt. shared elements is asserted.

        Examples
        --------
        >>> intersect_lists([1,2,3,4],[3,2,1]) # [3,2,1]
        >>> list_of_lists = ...
        >>> intersect_lists(*list_of_lists)
        >>> intersect_lists([1,2,3,4],[1,4],[3,1], check_order=True)
        >>> intersect_lists((1,2,3,5),(1,3,4,5)) # [1,3,5]
    """
    if len(lists) == 0:
        return []

    def intersect_lists_(list1, list2):
        return list(filter(lambda x: x in list1, list2))

    isec = reduce(intersect_lists_, lists)
    for lst in lists:
        if check_order:
            assert [x for x in lst if x in isec] == isec, f"Input list have differing order of shared elements {isec}"
        elif [x for x in lst if x in isec] != isec:
            print(f"WARN: Input list have differing order of shared elements {isec}")
    return isec


def cmp_sets(a, b) -> (bool, bool, bool):
    """ Set comparison. Returns shared, unique(a) and unique(b) items """
    return a & b, a.difference(b), b.difference(a)


def get_unique_keys(dict_of_dicts):
    """ Returns all unique key names from a dict of dicts.
        Example: get_unique_keys({'a':{'1':12,'2':13}, 'b': {'1':14,'3':43}})
    """
    keys = set()
    for d in dict_of_dicts.values():
        keys |= d.keys()
    return keys


def calc_set_overlap(a, b) -> float:
    """ Calculates the overlap between two sets """
    return len(a & b) / len(a | b)


def grouper(iterable, n, fill_value=None):
    """ Groups n lines into a list """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def to_str(*args, sep=',', na='NA') -> str:
    """
        Converts an object to a string representation. Iterables will be joined by the configured separator.
        Objects with zero length or None will be converted to the configured 'NA' representation.

        Examples
        --------
        >>> to_str([12,[None,[1,'',[]]]]) # 12,NA,1,NA,NA

        Notes
        -----
        This implementation is different from dm-tree.flatten() or treevalue.flatten() but slower?
    """
    if (args is None) or (len(args) == 0):
        return na
    if len(args) == 1:
        args = args[0]
    if args is None:
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
        If out is None, the respective string will be written, Otherwise it will be written to the respective output
        handle.
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

        Parameters
        ----------
        root:
            Path instance pointing to a directory
        prefix:
            optional prefix string to prepend to each output line
        space:
            optional prefix string to prepend to lines indicating a regular file
        branch:
            optional prefix string to prepend to lines indicating the last item in a directory
        tee:
            optional prefix string to prepend to lines indicating an item in a directory
        last:
            optional prefix string to prepend to lines indicating the last item in the entire tree
        max_lines:
            maximum yielded lines per directory.
        glob:
            optional glob param to filter. Use, e.g., `'**/*.py'` to list all .py files. default: None (all files)

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
    if file.endswith(".gz"):
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


def remove_extension(p, remove_gzip=True):
    """ Returns a resolved PosixPath of the passed path and removes the extension.
        Will also remove '.gz' extensions if remove_gzip is True.
        example remove_extension('b/c.txt.gz') -> <pwd>/b/c
    """
    p = Path(p).resolve()
    if remove_gzip and '.gz' in p.suffixes:
        p = p.with_suffix('')  # drop '.gz'
    return p.with_suffix('')  # drop ext


def download_file(url, filename, show_progress=True):
    """ Downloads a file from the passed (https) url into a  file with the given path
        Examples
        --------
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tempdirname:
        >>>     fn=download_file(..., f"{tempdirname}/{filename}")
        >>>     # do something with this file
        >>>     # Note that the temporary created dir will be removed once the context manager is closed.
    """

    def print_progress(block_num, block_size, total_size):
        print(f"progress: {round(block_num * block_size / total_size * 100, 2)}%", end="\r")

    ssl._create_default_https_context = ssl._create_unverified_context  # noqa
    urllib.request.urlretrieve(url, filename, print_progress if show_progress else None)
    return filename


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
        Adds up-/downstream padding with the configured padding character to ensure a given minimum length of the
        passed sequence.
    """
    ret = seq
    if len(ret) < minlen:
        pad0 = padding_char * int((minlen - len(ret)) / 2)
        pad1 = padding_char * int(minlen - (len(ret) + len(pad0)))
        ret = ''.join([pad0, ret, pad1])
    return ret


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


def count_rest(s, rest=('GGTACC', 'GAATTC', 'CTCGAG', 'CATATG', 'ACTAGT')) -> int:
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


def find_gpos(genome_fa, kmers, included_chrom=None) -> defaultdict[list]:
    """ Returns a dict that maps the passed kmers to their (exact match) genomic positions (1-based).
        Positions are returned as (chr, pos1) tuples.
        included_chromosomes (list): if configured, then only the respective chromosomes will be considered.
    """
    fasta = pysam.Fastafile(genome_fa)
    ret = defaultdict(list)
    chroms = fasta.references if included_chrom is None else set(fasta.references) & set(included_chrom)
    for c in tqdm(chroms, total=len(chroms), desc='Searching chromosome'):
        pos = kmer_search(fasta.fetch(c), kmers)
        for kmer in kmers:
            if kmer not in ret:
                ret[kmer] = []
            for pos0 in pos[kmer]:
                ret[kmer].append((c, pos0 + 1, '+'))
    return ret


def parse_gff_attributes(info, fmt='gff3'):
    """ parses GFF3/GTF info sections """
    if '#' in info:  # remove optional comment section (e.g., in flybase gtf)
        info = info.split('#')[0].strip()
    if fmt.lower() == 'gtf':
        return {k: v.translate({ord(c): None for c in '"'}) for k, v in
                [a.strip().split(' ', 1) for a in info.split(';') if ' ' in a]}
    return {k.strip(): v for k, v in [a.split('=') for a in info.split(';') if '=' in a]}


def bgzip_and_tabix(in_file, out_file=None, create_index=True, del_uncompressed=True,
                    preset='auto', seq_col=0, start_col=1, end_col=1, line_skip=0, zerobased=False):
    """
    BGZIP the input file and create a tabix index with the given parameters if create_index is True.

    Parameters
    ----------
    in_file : str
        The input file to be compressed.
    out_file : str, optional
        The output file name. Default is in_file + '.gz'.
    create_index : bool, optional
        Whether to create a tabix index. Default is True.
    del_uncompressed : bool, optional
        Whether to delete the uncompressed input file. Default is True.
    preset : str, optional
        The preset format for the tabix index. Default is 'auto'.
        presets:
        * 'gff' : (TBX_GENERIC, 1, 4, 5, ord('#'), 0),
        * 'bed' : (TBX_UCSC, 1, 2, 3, ord('#'), 0),
        * 'psltbl' : (TBX_UCSC, 15, 17, 18, ord('#'), 0),
        * 'sam' : (TBX_SAM, 3, 4, 0, ord('@'), 0),
        * 'vcf' : (TBX_VCF, 1, 2, 0, ord('#'), 0),
        * 'auto': guess from file extension
    seq_col : int, optional
        The column number for the sequence name. Default is 0.
    start_col : int, optional
        The column number for the start position. Default is 1.
    end_col : int, optional
        The column number for the end position. Default is 1.
    line_skip : int, optional
        The number of lines to skip at the beginning of the file. Default is 0.
    zerobased : bool, optional
        Whether the start position is zero-based. Default is False.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If out_file does not end with '.gz'.

    Notes
    -----
    This function requires the pysam package to be installed.
    TODO add csi support

    Examples
    --------
    >>> bgzip_and_tabix('input.vcf', create_index=True)

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
            if preset not in ['gff', 'bed', 'psltbl', 'sam', 'vcf']:  # currently supported by tabix
                preset = None
            # print(f"Detected file format for index creation: {preset}")
        pysam.tabix_index(out_file, preset=preset, force=True, seq_col=seq_col, start_col=start_col, end_col=end_col,
                          meta_char='#', line_skip=line_skip, zerobased=zerobased)  # noqa @UndefinedVariable
    if del_uncompressed:
        os.remove(in_file)


def _fast5_tree(h5node, prefix: str = '', space='    ', branch='│   ', tee='├── ', last='└── ', max_lines=10,
                show_attrs=True):
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
    """
    with h5py.File(fast5_file, 'r') as f:
        for cnt, rn in enumerate(f.keys()):
            for line in _fast5_tree(f[rn], prefix=rn + ' ', max_lines=max_lines, show_attrs=show_attrs):
                print(line)
            print('---')
            if cnt + 1 >= n_reads:
                return


def get_bcgs(fast5_file):
    """
        Returns a list of basecall groups from the 1st read
    """
    with h5py.File(fast5_file, 'r') as f:
        first_rn = next(iter(f.keys()))
        return [a for a in list(f[first_rn]['Analyses']) if a.startswith("Basecall_")]


def display_textarea(txt):
    """ Display a (long) text in a scrollable HTML text area """
    display(HTML(f"<textarea rows='4' cols='120'>{txt}</textarea>"))


def display_list(lst):
    """ Display a list as an HTML list"""
    display(HTML("<ul>"))
    for i in lst:
        display(HTML(f"<li>{i}</li>"))
    display(HTML("</ul>"))


def head_counter(cnt, non_empty=True):
    """Displays n items from the passed counter. If non_empty is true then only items with len>0 are shown"""
    if non_empty:
        cnt = Counter({k: cnt.get(k, 0) for k in cnt.keys() if len(cnt.get(k)) > 0})
    display(Counter({k: v for k, v in islice(cnt.items(), 1, 10)}), HTML("[...]"))


def plot_times(title, times, n=None,
               reference_method=None,
               show_speed=True,
               ax=None,
               orientation='h'
               ):
    """
        Helper method to plot a dict with timings (seconds).
        If n is passed and show_speed is true, the method will display iterations per second.
        If reference_method is set then it will also display the speed increase of the fastest method compared to
        the reference method/
    """
    ax = ax or plt.gca()
    labels, values = zip(*sorted(times.items(), key=lambda item: item[1])) # sort by value
    if show_speed and n is not None:
        values = [n / v for v in values]
        if reference_method is not None and reference_method in times:
            times_other = {k: v for k, v in times.items() if k != reference_method}
            fastest_other = min(times_other, key=times_other.get)
            a = (reference_method, n / times[reference_method])
            b = (fastest_other, n / times_other[fastest_other])
            a, b = (a, b) if a[1] > b[1] else (b, a)  # a: fastest, b: 2nd/reference
            ax.set_title(
                f"{title}\n{a[0]} is the fastest method and {(a[1] / b[1] - 1) * 100}%\nfaster than {b[0]}",
                fontsize=10)
        else:
            ax.set_title(f"{title}")
        data_lab = "it/s"
    else:
        ax.set_title(f"{title}")
        data_lab = "seconds"

    if orientation.startswith('h'):
        ax.barh(range(len(labels)), values, 0.8, )
        ax.set_yticks(range(len(labels)), labels, rotation=0)
        ax.set_xlabel(data_lab)
    else:
        ax.bar(range(len(labels)), values, 0.8, )
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_ylabel(data_lab)


def guess_file_format(file_name, file_extensions=None):
    """
    Guesses the file format from the file extension
    :param file_name:
    :param file_extensions:
    :return:
    """
    if file_extensions is None:
        file_extensions = default_file_extensions
    if file_name is not None:
        for ff, ext in file_extensions.items():  # TODO: make faster
            if file_name.endswith(ext):
                return ff
    return None


class ParseMap(dict):
    """
        Extends default dict to return 'missing char' for missing keys (similar to defaultdict, but does not
        enter new values). Should be used with translate()
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.missing_char = kwargs.get('missing_char', 'N')
        kwargs.pop('missing_char', None)
        self.update(*args, **kwargs)

    def __missing__(self, key):
        return self.missing_char


TMAP = {'dna': ParseMap({x: y for x, y in zip(b'ATCGatcgNn', b'TAGCTAGCNN')}, missing_char='*'),
        'rna': ParseMap({x: y for x, y in zip(b'AUCGaucgNn', b'UAGCUAGCNN')}, missing_char='*')
        }
default_file_extensions = {
    'fasta': ('.fa', '.fasta', '.fna', '.fas', '.fa.gz', '.fasta.gz', '.fna.gz', '.fas.gz'),
    'sam': ('.sam',),
    'bam': ('.bam',),
    'tsv': ('.tsv', '.tsv.gz'),
    'bed': ('.bed', '.bed.gz', '.bedgraph', '.bedgraph.gz'),
    'vcf': ('.vcf', '.vcf.gz'),
    'bcf': ('.bcf',),
    'gff': ('.gff3', '.gff3.gz'),
    'gtf': ('.gtf', '.gtf.gz'),
    'fastq': ('.fq', '.fastq', '.fq.gz', '.fastq.gz'),
}


class AutoDict(dict):
    """Implementation of perl's autovivification feature.
    https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python/651879#651879
    """

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_config(config, keys, default_value=None, required=False):
    """
        Gets a configuration value from a config dict (e.g., loaded from JSON).
        Keys can be a list of keys (that will be traversed) or a single value.
        If the key is missing and required is True, an error will be thrown. Otherwise, the configured default value
        will be returned.

        Examples
        --------
        >>> cmd_bcftools = get_config(config, ['params','cmd','cmd_bcftools'], required=True)
        >>> threads = get_config(config, 'config/process/threads', default_value=1, required=False)

        Notes
        -----
        Consider replacing this with omegaconf
    """
    if isinstance(keys, str):  # handle single strings
        keys = keys.split('/')
    d = config
    for k in keys:
        if k is None:
            continue  # ignore None keys
        if k not in d:
            assert (not required), 'Mandatory config path "%s" missing' % ' > '.join(keys)
            return default_value
        d = d[k]
    return d


def open_file_obj(fh, file_format=None, file_extensions=None) -> object:
    """
    Opens a file object. If a pysam compatible file format was detected, the respective pysam object is instantiated.

    Parameters
    ----------
    fh : str or file path object
    file_format : str
        Can be any <supported_formats> or None for auto-detection from filename (valid file extensions can be
        configured).
    file_extensions: dict
        a dict of file extension mappings

    Returns
    -------
        file_handle : instance (file/pysam object)
    """
    if file_extensions is None:
        file_extensions = default_file_extensions
    fh = str(fh)  # convert path to str
    if file_format is None:  # auto detect via file extension
        file_format = guess_file_format(fh, file_extensions)
    # instantiate pysam object
    if file_format == 'fasta':
        fh = pysam.Fastafile(fh)  # @UndefinedVariable
    elif file_format == 'sam':
        fh = pysam.AlignmentFile(fh, "r")  # @UndefinedVariable
    elif file_format == 'bam':
        fh = pysam.AlignmentFile(fh, "rb")  # @UndefinedVariable
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


def toggle_chr(s):
    """
        Simple function that toggle 'chr' prefixes. If the passed reference name start with 'chr',
        then it is removed, otherwise it is added. If None is passed, None is returned.
        Can be used as fun_alias function.
    """
    if s is None:
        return None
    if isinstance(s, str) and s.startswith('chr'):
        return s[3:]
    else:
        return f'chr{s}'


@dataclass(frozen=True)
class GeneSymbol:
    """
        Class for representing a gene symbol, name and taxonomy id.
    """
    symbol: str  #
    name: str  #
    taxid: int  #

    def __repr__(self):
        return f"{self.symbol} ({self.name}, tax: {self.taxid})"


def count_reads(in_file):
    """ Counts reads in different file types """
    ftype = guess_file_format(in_file)
    if ftype == 'fastq':
        return count_lines(in_file) / 4.0
    elif ftype == 'sam':
        raise NotImplementedError("SAM/BAM file read counting not implemeted yet!")
    else:
        raise NotImplementedError(f"Cannot count reads in file of type {ftype}.")


def get_softclip_seq(read: pysam.AlignedSegment) -> tuple[Optional[int], Optional[int]]:
    """
    Extracts soft-clipped sequences from the passed read.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to extract soft-clipped sequences from.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        A tuple containing the left and right soft-clipped sequences, respectively. If no soft-clipped sequence is
        found, the corresponding value in the tuple is None.

    Examples
    --------
    >>> r = pysam.AlignedSegment()
    >>> r.cigartuples = [(4, 5), (0, 10)]
    >>> get_softclip_seq(r)
    (5, None)
    """
    left, right = None, None
    pos = 0
    for i, (op, l) in enumerate(read.cigartuples):
        if (i == 0) & (op == 4):
            left = l
        if (i == len(read.cigartuples) - 1) & (op == 4):
            right = l
        pos += l
    return left, right


def get_covered_contigs(bam_files):
    """ Returns all contigs that have some coverage across a set of BAMs.

    Parameters
    ----------
    bam_files : str
        File paths of input BAM files. If a single path is passed, it will be converted to a list.

    Returns
    -------
    set
        A set of strings representing all contigs that have data in at least one of the BAM files.
    """
    if isinstance(bam_files, str):
        bam_files = [bam_files]
    covered_contigs = set()
    for b in bam_files:
        s = pysam.AlignmentFile(b, "rb")  # @UndefinedVariable
        covered_contigs.update([c for c, _, _, t in s.get_index_statistics() if t > 0])
    return covered_contigs


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
        out.write(record)
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
            out.write(record)


def get_read_attr_info(fast5_file, rn, path):
    """
        Returns a dict with attribute values. For debugging only.
        Example
        -------
        >>> get_read_attr_info(fast5_file, 'read_00262802-9463-45c8-b22d-f68d1047c6fc',
        >>>                    'Analyses/Basecall_1D_000/BaseCalled_template/Trace')
    """
    with h5py.File(fast5_file, 'r') as f:
        x = f[rn][path]
        return {k: v for k, v in zip(x.attrs.keys(), x.attrs.values())}


class Timer:
    """ Simple class for collecting timings of code blocks.

    Examples
    --------
    >>> import time
    >>> times=Counter()
    >>> with Timer(times, "sleep1") as t:
    >>>     time.sleep(.1)
    >>> print('executed ', t.name)
    >>> with Timer(times, "sleep2") as t:
    >>>     time.sleep(.2)
    >>> print(times)
    """

    def __init__(self, tdict, name):
        self.tdict = tdict
        self.name = name

    def __enter__(self):
        self.tdict[self.name] = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tdict[self.name] = timeit.default_timer() - self.tdict[self.name]


def read_alias_file(gene_name_alias_file, disable_progressbar=False) -> (dict, set):
    """ Reads a gene name aliases from the passed file.
        Supports the download format from genenames.org and vertebrate.genenames.org.
        Returns an alias dict and a set of currently known (active) gene symbols
    """
    aliases = {}
    current_symbols = set()
    if gene_name_alias_file:
        tab = pd.read_csv(gene_name_alias_file, sep='\t',
                          dtype={'alias_symbol': str, 'prev_symbol': str, 'symbol': str}, low_memory=False,
                          keep_default_na=False)
        for r in tqdm(tab.itertuples(), desc='load gene aliases', total=tab.shape[0], disable=disable_progressbar):
            sym = r.symbol.strip()  # noqa
            current_symbols.add(sym)
            for a in r.alias_symbol.split("|"):  # noqa
                if len(a.strip()) > 0:
                    aliases[a.strip()] = sym
            for a in r.prev_symbol.split("|"):  # noqa
                if len(a.strip()) > 0:
                    aliases[a.strip()] = sym
    return aliases, current_symbols


def norm_gn(g, current_symbols=None, aliases=None) -> str:
    """
    Normalizes gene names. Will return a stripped version of the passed gene name.
    The gene symbol will be updated to the latest version if an alias table is passed (@see read_alias_file()).
    If a set of current (up-to date) symbols is passed that contains the passed symbol, no aliasing will be done.
    """
    if g is None:
        return None
    g = g.strip()  # remove whitespace
    if (current_symbols is not None) and (g in current_symbols):
        return g  # there is a current symbol so don't look for aliases
    if aliases is None:
        return g
    return g if aliases is None else aliases.get(g, g)


def geneid2symbol(gene_ids):
    """
        Queries gene names for the passed gene ids from MyGeneInfo via https://pypi.org/project/mygene/
        Gene ids can be, e.g., EntrezIds (e.g., 60) or ensembl gene ids (e.g., 'ENSMUSG00000029580') or a mixed list.
        Returns a dict { entrezid : GeneSymbol }
        Example: geneid2symbol(['ENSMUSG00000029580', 60]) - will return mouse and human actin beta
    """
    mg = mygene.MyGeneInfo()
    galias = mg.getgenes(set(gene_ids), filter='symbol,name,taxid')
    id2sym = {x['query']: GeneSymbol(x.get('symbol', x['query']), x.get('name', None), x.get('taxid', None)) for x in
              galias}
    return id2sym


def calc_3end(tx, width=200):
    """
        Utility function that returns a list of genomic intervals containing the last <width> bases
        of the passed transcript or None if not possible
    """
    ret = []
    for ex in tx.exon[::-1]:
        if len(ex) < width:
            ret.append(ex.get_location())
            width -= len(ex)
        else:
            s, e = (ex.start, ex.start + width - 1) if (ex.strand == '-') else (ex.end - width + 1, ex.end)
            ret.append(rnalib.gi(ex.chromosome, s, e, ex.strand))
            width = 0
            break
    return ret if width == 0 else None


def gt2zyg(gt) -> (int, int):
    """
    Extracts zygosity information from a genotype string.

    Parameters
    ----------
    gt genotype

    Returns
    -------
    a tuple (zygosity, call):
        * zygosity is 2 if all called alleles are the same, 1 if there are mixed called alleles or 0 if no call
        * call: 0 if no-call or homref, 1 otherwise
    """
    dat = gt.split('/') if '/' in gt else gt.split('|')
    if set(dat) == {'.'}:  # no call
        return 0, 0
    dat_clean = [x for x in dat if x != '.']  # drop no-calls
    if set(dat_clean) == {'0'}:  # homref in all called samples
        return 2, 0
    return 2 if len(set(dat_clean)) == 1 else 1, 1


#: A read in a FASTQ file
FastqRead = namedtuple('FastqRead', 'name seq qual')


@dataclass
class TagFilter:
    """
        Filter reads if the specified tag has one of the provided filter_values.
        Can be inverted for filtering if specified values is found.
        If filter_if_no_tag is True (default: False), then the read is filtered if the tag is not present.
    """
    tag: str
    filter_values: List = field(default_factory=list)
    filter_if_no_tag: bool = False
    inverse: bool = False

    def filter(self, r):
        if r.has_tag(self.tag):
            value_exists = r.get_tag(self.tag) in self.filter_values
            return value_exists != self.inverse
        else:
            return self.filter_if_no_tag
