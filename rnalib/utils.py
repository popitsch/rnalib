"""
This module implements various general (low-level) utility methods

"""
import gzip
import itertools
import logging
import math
import numbers
import os
import random
import re
import shutil
import ssl
import time
import timeit
import unicodedata
import urllib.request
import warnings
from collections import Counter, namedtuple, defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from functools import reduce
from io import StringIO
from itertools import groupby, zip_longest, islice, pairwise
from pathlib import Path
from typing import Optional, List
import html

import h5py
import mygene
import numpy as np
import pandas as pd
import pyBigWig
import pybedtools
import pysam
import seaborn as sns
from IPython.core.display_functions import display
from IPython.display import HTML, Javascript, clear_output
from matplotlib import pyplot as plt
from termcolor import colored
from tqdm.auto import tqdm

import rnalib as rna

# --------------------------------------------------------------
# datastructures
# --------------------------------------------------------------

GeneSymbol = namedtuple("GeneSymbol", "symbol name taxid")
GeneSymbol.__doc__ = (
    "A named tuple representing a gene symbol with symbol, name, and taxid fields."
)


# --------------------------------------------------------------
# Commandline and config handling
# --------------------------------------------------------------


def ensure_outdir(outdir=None) -> os.PathLike:
    """Ensures that the configured output dir exists (will use current dir if none provided)"""
    outdir = os.path.abspath(outdir if outdir else os.getcwd())
    if not outdir.endswith("/"):
        outdir += "/"
    if not os.path.exists(outdir):
        logging.debug("Creating dir " + outdir)
        os.makedirs(outdir)
    return outdir


# --------------------------------------------------------------
# Collection helpers
# --------------------------------------------------------------


def check_list(lst, mode="inc1") -> bool:
    """
    Tests whether the (numeric, comparable) items in a list are
    * mode=='inc': increasing (strictly monotonic)
    * mode=='dec': decreasing (strictly monotonic)
    * mode=='inceq': increasing (monotonic)
    * mode=='deceq': decreasing (monotonic)
    * mode=='inc1': increasing by one
    * mode=='dec1': decreasing by one
    * mode=='eq': all equal
    """
    if mode == "inc":
        return all(x < y for x, y in zip(lst, lst[1:]))
    elif mode == "inc1":
        return all(x + 1 == y for x, y in zip(lst, lst[1:]))
    if mode == "inceq":
        return all(x <= y for x, y in zip(lst, lst[1:]))
    elif mode == "dec":
        return all(x > y for x, y in zip(lst, lst[1:]))
    elif mode == "dec1":
        return all(x - 1 == y for x, y in zip(lst, lst[1:]))
    elif mode == "deceq":
        return all(x >= y for x, y in zip(lst, lst[1:]))
    elif mode == "eq":
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
        return [lst[i * n : (i + 1) * n] for i in range((len(lst) + n - 1) // n)]
    else:
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def intersect_lists(*lists, check_order=False) -> list:
    """Intersects lists (iterables) while preserving order. Order is determined by the last provided list.
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
            assert [
                x for x in lst if x in isec
            ] == isec, f"Input lists have differing order of shared elements {isec}"
        elif [x for x in lst if x in isec] != isec:
            warnings.warn(f"Input lists have differing order of shared elements {isec}")
    return isec


def cmp_sets(a, b) -> (bool, bool, bool):
    """Set comparison. Returns shared, unique(a) and unique(b) items"""
    return a & b, a.difference(b), b.difference(a)


def get_unique_keys(dict_of_dicts):
    """Returns all unique key names from a dict of dicts.
    Example: get_unique_keys({'a':{'1':12,'2':13}, 'b': {'1':14,'3':43}})
    """
    keys = set()
    for d in dict_of_dicts.values():
        keys |= d.keys()
    return keys


def calc_set_overlap(a, b) -> float:
    """Calculates the overlap between two sets"""
    return len(a & b) / len(a | b)


def powerset(it):
    """
    Returns all possible subsets of the passed iterable.
    Parameters
    ----------
    it: iterable

    Returns
    -------
    The powerset of the passed iterable

    Examples
    --------
    >>> list(powerset([1,2,3])) # [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    """
    s = list(it)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def grouper(iterable, n, fill_value=None):
    """Groups n lines into a list"""
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


# --------------------------------------------------------------
# I/O handling
# --------------------------------------------------------------


def to_set(x, sep=",") -> set:
    """Converts the passed object to a set if not already.
    If None is passed, an empty set is returned.
    if a string is passed, it will be split by the specified separator.
    if a set, list or tuple is passed, it will be converted to a set.
    If a number or iterable is passed, it will be converted to a set with one element.
    Returns
    -------
    set
    """
    if x is None:
        return set()
    elif isinstance(x, str):
        return set(x.split(sep))
    elif isinstance(x, (set, list, tuple)):
        return set(x)
    elif isinstance(x, numbers.Number):
        return {x}
    elif hasattr(x, "__iter__") and hasattr(x, "__len__"):
        return {x}
    return set(x)


def to_list(x, sep=",") -> list:
    """Converts the passed object to a list if not already.
    If None is passed, an empty list is returned.
    if a string is passed, it will be split by the specified separator.
    if a set, list or tuple is passed, it will be converted to a list.
    If a number or iterable is passed, it will be converted to a list with one element.

    Returns
    -------
    list
    """
    if x is None:
        return list()
    elif isinstance(x, str):
        return list(x.split(sep))
    elif isinstance(x, (set, list, tuple)):
        return list(x)
    elif isinstance(x, numbers.Number):
        return [x]
    elif hasattr(x, "__iter__") and hasattr(x, "__len__"):
        return [x]
    return list(x)


def to_str(*args, sep=",", na="NA") -> str:
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
    if (
        hasattr(args, "__len__")
        and callable(getattr(args, "__len__"))
        and len(args) == 0
    ):
        return na
    if isinstance(args, str):
        return args
    if (
        hasattr(args, "__len__")
        and hasattr(args, "__iter__")
        and callable(getattr(args, "__iter__"))
    ):
        return sep.join([to_str(x, sep=sep, na=na) for x in args])
    return str(args)


def write_data(dat, out=None, sep="\t", na="NA"):
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
    """Format a string for FASTA files"""
    return "\n".join(string[i : i + ncol] for i in range(0, len(string), ncol))


def convert_size(size_bytes):
    """Convert bytes to human-readable format,
    see https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def dir_tree(
    root: Path,
    prefix: str = "",
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
    max_lines=10,
    glob=None,
    show_size=True,
):
    """A recursive generator yielding a visual tree structure line by line.

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
    show_size:
        optional; if True, the size of the file will be printed in brackets after the filename
    @see https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    if isinstance(root, str):
        root = Path(root)
    contents = list(root.iterdir()) if (glob is None) else list(root.glob(glob))
    if len(contents) > max_lines:
        contents = contents[:max_lines] + [Path("...")]
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        size = (
            f" ({convert_size(path.stat().st_size)})"
            if show_size and path.is_file()
            else ""
        )
        yield prefix + pointer + path.name + size  # print the item
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from dir_tree(
                path, prefix=prefix + extension, max_lines=max_lines, glob=glob
            )


def print_dir_tree(root: Path, max_lines=10, glob=None):
    if isinstance(root, str):
        root = Path(root)
    for line in dir_tree(root, max_lines=max_lines, glob=glob):
        print(line)


def count_lines(file) -> int:
    """Counts lines in (gzipped) files. Slow."""
    if file.endswith(".gz"):
        with gzip.open(file, "rb") as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    else:
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        return i + 1


def gunzip(in_file, out_file=None) -> str:
    """Gunzips a file and returns the filename of the resulting file"""
    assert in_file.endswith(".gz"), "in_file must be a .gz file"
    if out_file is None:
        out_file = in_file[:-3]
    with gzip.open(in_file, "rb") as f_in:
        with open(out_file, "wb") as f_out:
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
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "_", value).strip("_")


def remove_extension(p, remove_gzip=True) -> str:
    """Returns a string representation of a resolved PosixPath of the passed path with removed file extension.
    Will also remove '.gz' extensions if remove_gzip is True.
    example remove_extension('b/c.txt.gz') -> '<pwd>/b/c'
    """
    p = Path(p).resolve()
    if remove_gzip and ".gz" in p.suffixes:
        p = p.with_suffix("")  # drop '.gz'
    return str(p.with_suffix(""))  # drop ext


class UrlretrieveTqdm:
    def __init__(self, filename):
        self.pbar = None
        self.filename = os.path.basename(filename)

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(
                total=total_size,
                desc=f"Downloading {self.filename}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                position=0,
                leave=True,
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.set_description(f"Download complete.")

    def __enter__(self):
        pass

    def __exit__(self, extype, value, traceback):
        if self.pbar:
            self.pbar.close()


def download_file(url, filename, show_progress=True):
    """Downloads a file from the passed (https) url into a  file with the given path

    Parameters
    ----------
    url: str
        URL to download from
    filename: str
        Path to the file to be created
    show_progress: bool
        If True, a progress bar will be shown

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tempdirname:
    >>>     fn=download_file(..., f"{tempdirname}/{filename}")
    >>>     # do something with this file
    >>>     # Note that the temporary created dir will be removed once the context manager is closed.
    """

    # def print_progress(block_num, block_size, total_size):
    #    print(
    #        f"progress: {min(100, round(block_num * block_size / total_size * 100, 2))}%",
    #        end="\r",
    #    )
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa
    if show_progress:
        with UrlretrieveTqdm(filename) as pbar:
            # urllib.request.urlretrieve(url, filename, print_progress if show_progress else None)
            urllib.request.urlretrieve(url, filename, pbar)
    else:
        urllib.request.urlretrieve(url, filename)
    return filename


# --------------------------------------------------------------
# Sequence handling
# --------------------------------------------------------------


def reverse_complement(seq, tmap="dna") -> str:
    """
    Calculate reverse complement DNA/RNA sequence.
    Returned sequence is uppercase, N's are kept.
    seq_type can be 'dna' (default) or 'rna'
    """
    if isinstance(tmap, str):
        tmap = TMAP[tmap]
    return seq[::-1].translate(tmap)


def complement(seq, tmap="dna") -> str:
    """
    Calculate complement DNA/RNA sequence.
    Returned sequence is uppercase, N's are kept.
    seq_type can be 'dna' (default) or 'rna'
    """
    if isinstance(tmap, str):
        tmap = TMAP[tmap]
    return seq.translate(tmap)


def pad_n(seq, minlen, padding_char="N") -> str:
    """
    Adds up-/downstream padding with the configured padding character to ensure a given minimum length of the
    passed sequence.
    """
    ret = seq
    if len(ret) < minlen:
        pad0 = padding_char * int((minlen - len(ret)) / 2)
        pad1 = padding_char * int(minlen - (len(ret) + len(pad0)))
        ret = "".join([pad0, ret, pad1])
    return ret


def rnd_seq(n, alpha="ACTG", m=1):
    """
    Creates m random sequence of length n using the provided alphabet (default: DNA bases).
    To use different character frequencies, pass each character as often as expected in the frequency distribution.
    Example:    rnd_seq(100, 'GC'* 60 + 'AT' * 40, 5) # 5 sequences of length 100 with 60% GC
    """
    if m <= 0:
        return None
    res = ["".join(random.choice(alpha) for _ in range(n)) for _ in range(m)]
    return res if m > 1 else res[0]


def count_gc(s) -> (int, float):
    """
    Counts number of G+C bases in string.
     Returns the number of GC bases and the length-normalized fraction.
    """
    ngc = s.count("G") + s.count("C")
    return ngc, ngc / len(s)


def count_rest(s, rest=("GGTACC", "GAATTC", "CTCGAG", "CATATG", "ACTAGT")) -> int:
    """Counts number of restriction sites, see https://portals.broadinstitute.org/gpp/public/resources/rules"""
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
            c["hp"] += 1
        else:
            if c["hp"] > c["longest_hp"]:
                c["longest_hp"] = c["hp"]
            c["hp"] = 1
        if base in ["G", "C"]:
            c["gc"] += 1
        else:
            if c["gc"] > c["longest_gc"]:
                c["longest_gc"] = c["gc"]
            c["gc"] = 0
        last_char = base
    if c["hp"] > c["longest_hp"]:
        c["longest_hp"] = c["hp"]
    if c["gc"] > c["longest_gc"]:
        c["longest_gc"] = c["gc"]
    # get max base:return max(c, key=c.get)
    return c["longest_hp"], c["longest_gc"]


def longest_GC_len(seq) -> int:
    """Counts HP length (any allele) from start"""
    c = Counter()
    last_char = None
    for base in seq:
        if last_char in ["G", "C"]:
            c["GC"] += 1
        else:
            c["GC"] = 1
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
    """Exact kmer search. Can include revcomp if configured"""
    ret = dict()
    for kmer in kmer_set:
        ret[kmer] = list(find_all(seq, kmer))
        if include_revcomp:
            ret[kmer].extend(list(find_all(seq, reverse_complement(kmer))))
    return ret


def find_gpos(genome_fa, kmers, included_chrom=None) -> defaultdict[list]:
    """Returns a dict that maps the passed kmers to their (exact match) genomic positions (1-based).
    Positions are returned as (chr, pos1) tuples.
    included_chromosomes (list): if configured, then only the respective chromosomes will be considered.
    """
    fasta = pysam.Fastafile(genome_fa)
    ret = defaultdict(list)
    chroms = (
        fasta.references
        if included_chrom is None
        else set(fasta.references) & set(included_chrom)
    )
    for c in tqdm(chroms, total=len(chroms), desc="Searching chromosome"):
        pos = kmer_search(fasta.fetch(c), kmers)
        for kmer in kmers:
            if kmer not in ret:
                ret[kmer] = []
            for pos0 in pos[kmer]:
                ret[kmer].append((c, pos0 + 1, "+"))
    return ret


def parse_gff_attributes(info, fmt="gff3"):
    """parses GFF3/GTF info sections"""
    try:
        if "#" in info:  # remove optional comment section (e.g., in flybase gtf)
            info = info.split("#")[0].strip()
        if fmt.lower() == "gtf":
            return {
                k: v.translate({ord(c): None for c in '"'})
                for k, v in [
                    a.strip().split(" ", 1) for a in info.split(";") if " " in a.strip()
                ]
            }
        return {
            k.strip(): v for k, v in [a.split("=") for a in info.split(";") if "=" in a]
        }
    except ValueError as e:
        logging.error(f"Error parsing GFF3/GTF info section: {info}", e)
        raise e


def compact_bedgraph_file(bedgraph_file, out_file=None):
    """
    Compacts a bedgraph file by merging adjacent entries with the same value.
    """
    out_file = bedgraph_file + ".compact.bedgraph" if out_file is None else out_file
    written = False
    with open(out_file, "wt") as f:
        last_loc, last_dat = None, None
        for loc, dat in rna.it(bedgraph_file, style="bedgraph"):
            if last_loc is None:
                last_loc, last_dat = loc, dat
                written = False
                continue
            if (last_loc.end == loc.start - 1) and (last_dat == dat):
                last_loc = rna.gi(
                    last_loc.chromosome, last_loc.start, loc.end, last_loc.strand
                )
                # print("Merged", last_loc, last_dat)
                written = False
                continue
            print(
                to_str(
                    last_loc.chromosome,
                    last_loc.start - 1,
                    last_loc.end,
                    last_dat,
                    sep="\t",
                ),
                file=f,
            )
            written = True
            last_loc, last_dat = loc, dat
        print(
            to_str(
                last_loc.chromosome,
                last_loc.start - 1,
                last_loc.end,
                last_dat,
                sep="\t",
            ),
            file=f,
        )
    return bgzip_and_tabix(out_file)


def bgzip_and_tabix(
    in_file,
    out_file=None,
    sort=False,
    create_index=True,
    del_uncompressed=True,
    preset="auto",
    seq_col=0,
    start_col=1,
    end_col=1,
    meta_char=ord("#"),
    line_skip=0,
    zerobased=False,
):
    """
    BGZIP the input file and create a tabix index with the given parameters if create_index is True.
    File is sorted with pybedtools if sort is True.

    Parameters
    ----------
    in_file : str
        The input file to be compressed.
    out_file : str, optional
        The output file name. Default is in_file + '.gz'.
    sort : bool, optional
        Whether to sort the input file. Default is False.
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
    meta_char : int, optional
        The character that indicates a comment line. Default is ord('#').
    line_skip : int, optional
        The number of lines to skip at the beginning of the file. Default is 0.
    zerobased : bool, optional
        Whether the start position is zero-based. Default is False.

    Returns
    -------
    The filename of the compressed file.

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
        out_file = in_file + ".gz"
    assert out_file.endswith(".gz"), "out_file must be a .gz file"
    if sort:
        pre, post = os.path.splitext(in_file)
        sorted_in_file = pre + ".sorted" + post
        pybedtools.BedTool(in_file).sort().saveas(sorted_in_file)
        if del_uncompressed:
            os.remove(in_file)
        in_file = sorted_in_file
    pysam.tabix_compress(in_file, out_file, force=True)  # @UndefinedVariable
    if create_index:
        if preset == "auto":
            preset = guess_file_format(in_file)
            if preset == "gtf":
                preset = "gff"  # pysam default
        if preset == "bedgraph":
            preset = "bed"  # pysam default
        if preset not in [
            "gff",
            "bed",
            "psltbl",
            "sam",
            "vcf",
        ]:  # currently supported by tabix
            preset = None
        if preset is not None and line_skip > 0:
            # NOTE that there seems to be a bug in pysam that causes the line_skip parameter to be overwritten if a
            # preset code is used. Here we catch this case and use explicit seq_col, start_col, end_col instead.
            # @see https://github.com/samtools/htslib/blob/develop/htslib/tbx.h#L41 and
            # https://github.com/pysam-developers/pysam/blob/master/pysam/libctabix.pyx
            # preset_code : (seq_col, start_col, end_col, meta_char, zero_based)
            _PYSAM_PRESET_CONF = {
                "gff": (0, 3, 4, "#", False),
                "bed": (0, 1, 2, "#", True),
                "psltbl": (14, 16, 17, "#", False),
                "sam": (2, 3, -1, "@", False),
                "vcf": (0, 1, -1, "#", False),
            }
            seq_col, start_col, end_col, meta_char, zerobased = _PYSAM_PRESET_CONF[
                preset
            ]
            preset = None  # now call w/o preset code
        pysam.tabix_index(
            out_file,
            preset=preset,
            force=True,
            seq_col=seq_col,
            start_col=start_col,
            end_col=end_col,
            meta_char=meta_char,
            line_skip=line_skip,
            zerobased=zerobased,
        )  # noqa @UndefinedVariable
    if del_uncompressed:
        os.remove(in_file)
    return out_file


# --------------------------------------------------------------
# genomics helpers :: nanopore specific
# --------------------------------------------------------------


def _fast5_tree(
    h5node,
    prefix: str = "",
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
    max_lines=10,
    show_attrs=True,
):
    """Recursively yielding strings describing the structure of an h5 file"""
    if hasattr(h5node, "keys"):
        contents = list(h5node.keys())
        if len(contents) > max_lines:
            contents = contents[:max_lines] + [Path("...")]
        # contents each get pointers that are ├── with a final └── :
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, key in zip(pointers, contents):
            attrs_str = ""
            if show_attrs and hasattr(h5node[key], "attrs"):
                attrs_str = [
                    f"{k}={v}"
                    for k, v in zip(
                        h5node[key].attrs.keys(), h5node[key].attrs.values()
                    )
                ]
                if len(attrs_str) > 0:
                    attrs_str = " {" + ",".join(attrs_str) + "}"
                else:
                    attrs_str = ""
            yield prefix + pointer + key + attrs_str
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from _fast5_tree(
                h5node[key], prefix=prefix + extension, max_lines=max_lines
            )


def print_fast5_tree(fast5_file, max_lines=10, n_reads=1, show_attrs=True):
    """
    Prints the structure of a fast5 file.
    """
    with h5py.File(fast5_file, "r") as f:
        for cnt, rn in enumerate(f.keys()):
            for line in _fast5_tree(
                f[rn], prefix=rn + " ", max_lines=max_lines, show_attrs=show_attrs
            ):
                print(line)
            print("---")
            if cnt + 1 >= n_reads:
                return


def print_small_file(filename, show_linenumber=False, max_lines=10):
    """Prints the first 10 lines of a file"""
    infile = (
        gzip.open(filename, "rt")
        if filename.endswith(".gz")
        else open(filename, mode="rt")
    )
    for i, line in enumerate(infile):
        if i >= max_lines:
            print("...")
            break
        if show_linenumber:
            print(f"line {i + 1}: {line}", end="")
        else:
            print(line, end="")
    infile.close()


def get_bcgs(fast5_file):
    """
    Returns a list of basecall groups from the 1st read
    """
    with h5py.File(fast5_file, "r") as f:
        first_rn = next(iter(f.keys()))
        return [a for a in list(f[first_rn]["Analyses"]) if a.startswith("Basecall_")]


# --------------------------------------------------------------
# Utility functions for ipython notebooks
# --------------------------------------------------------------


def display_animated_gif(gif_url):
    display(
        HTML(
            """
    <!-- gh-pages -->
    <link rel='stylesheet' href='https://unpkg.com/freezeframe@3.0.10/build/css/freezeframe_styles.min.css'>
    <script type='text/javascript' src='https://unpkg.com/freezeframe@3.0.10/build/js/freezeframe.pkgd.min.js'></script>

    <script type='text/javascript'> 
      third = new freezeframe('.my_class_3').capture().setup();
      $(function() {
        $('.start').click(function(e) {
          e.preventDefault();
          third.trigger();
        });
      })
    </script>
    """
            + f"""
    <img class="my_class_3 freezeframe-responsive" src="{gif_url}" />
    <button class="start">restart</button>
    """
            + """
    <script>setTimeout( function() { third.trigger(); }, 1000);</script>
    """
        )
    )


def display_textarea(txt, rows=4, cols=120):
    """Display a (long) text in a scrollable HTML text area"""
    display(HTML(f"<textarea rows='{rows}' cols='{cols}'>{txt}</textarea>"))


def display_list(lst):
    """Display a list as an HTML list"""
    display(HTML("<ul>"))
    for i in lst:
        display(HTML(f"<li>{i}</li>"))
    display(HTML("</ul>"))


# def display_popup(msg, clear=True):
#     if clear:
#         clear_output()
#     display(HTML(f'<p style="color:red">{msg}</p>'))
#     display(HTML(f"<script>alert('{msg}');</script>"))


def display_popover(msg, bg_color="red", clear=True):
    if clear:
        clear_output()
    rndid = f"popover_{random.random()}"
    display(
        HTML(
            f"""
        <div id="{rndid}" style="color: {bg_color};position: fixed; bottom: 50%!important;" popover>{msg}</div>
    """
        )
    )
    display(
        Javascript(
            f"""
        const popover = document.getElementById("{rndid}");
        popover.showPopover();
    """
        )
    )


# def display_help(obj, rows=10):
#    """ Display the docstring of the passed object """
#    display(HTML(f"<strong>{obj.__name__} docstring:</strong>"))
#    display_textarea(obj.__doc__, rows=rows)


def display_help(obj, icon="help_icon.png", bg_color="#dcf6fa"):
    """Display the docstring of the passed object"""
    msg = html.escape(obj.__doc__).replace("\n", "<br/>").replace("\t", "&nbsp;" * 4)
    title = f"{obj.__name__} docstring"
    rndid = f"help_icon{random.random()}"
    display(
        HTML(
            f"""<a id="{rndid}"><img id="help_icon" style="cursor: help; float:left; width: 20px" src="{icon}"/>&nbsp;&nbsp;&nbsp;<b>{title}</b></a>"""
        )
    )
    msg = (
        f"<h1>{title}</h1>"
        + "<p style='font-family:Courier New; font-size: 14px;'>"
        + msg
        + "</p>"
    )
    display(
        Javascript(
            '''
    document.getElementById("'''
            + rndid
            + '''").addEventListener("click", function() {
        var win = window.open("", "", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=1000, height=600");
        win.document.body.innerHTML = "'''
            + msg
            + '''";
        win.document.title = "'''
            + title
            + '''";
        win.document.body.style.background = "'''
            + bg_color
            + """";
        });
        """
        )
    )


def head_counter(cnt, non_empty=True):
    """Displays n items from the passed counter. If non_empty is true then only items with len>0 are shown"""
    if non_empty:
        cnt = Counter(
            {
                k: cnt.get(k, 0)
                for k in cnt.keys()
                if cnt.get(k) is not None and len(cnt.get(k)) > 0
            }
        )
    display(Counter({k: v for k, v in islice(cnt.items(), 1, 10)}), HTML("[...]"))


def plot_times(
    title,
    times,
    n=None,
    reference_method=None,
    show_speed=True,
    show_fastest=False,
    ax=None,
    orientation="h",
    highlight_bar=None,
):
    """
    Helper method to plot a dict with timings (seconds).
    If n is passed and show_speed is true, the method will display iterations per second.
    If reference_method is set then it will also display the speed increase of the fastest method compared to
    the reference method/
    """
    ax = ax or plt.gca()
    labels, values = zip(
        *sorted(times.items(), key=lambda item: item[1])
    )  # sort by value
    if show_speed and n is not None:
        values = [n / v for v in values]
        if reference_method is not None and reference_method in times:
            times_other = {k: v for k, v in times.items() if k != reference_method}
            fastest_other = min(times_other, key=times_other.get)
            a = (reference_method, n / times[reference_method])
            b = (fastest_other, n / times_other[fastest_other])
            a, b = (a, b) if a[1] > b[1] else (b, a)  # a: fastest, b: 2nd/reference
            if show_fastest:
                ax.set_title(
                    f"{title}\n{a[0]} is the fastest method and {(a[1] / b[1] - 1) * 100}%\nfaster than {b[0]}",
                    fontsize=10,
                )
            else:
                ax.set_title(
                    f"{title}",
                    fontsize=10,
                )
        else:
            ax.set_title(f"{title}")
        data_lab = "it/s"
    else:
        ax.set_title(f"{title}")
        data_lab = "seconds"
    if orientation.startswith("h"):
        barlst = ax.barh(
            range(len(labels)),
            values,
            0.8,
        )
        if highlight_bar is not None:
            barlst[labels.index(highlight_bar)].set_color('r')
        ax.set_yticks(range(len(labels)), labels, rotation=0)
        ax.set_xlabel(data_lab)
    else:
        barlst = ax.bar(
            range(len(labels)),
            values,
            0.8,
        )
        if highlight_bar is not None:
            barlst[labels.index(highlight_bar)].set_color('r')
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
        self.missing_char = kwargs.get("missing_char", "N")
        kwargs.pop("missing_char", None)
        self.update(*args, **kwargs)

    def __missing__(self, key):
        return self.missing_char


TMAP = {
    "dna": ParseMap(
        {x: y for x, y in zip(b"ATCGatcgNn", b"TAGCTAGCNN")}, missing_char="*"
    ),
    "rna": ParseMap(
        {x: y for x, y in zip(b"AUCGaucgNn", b"UAGCUAGCNN")}, missing_char="*"
    ),
}
default_file_extensions = {
    "fasta": (
        ".fa",
        ".fasta",
        ".fna",
        ".fas",
        ".fa.gz",
        ".fasta.gz",
        ".fna.gz",
        ".fas.gz",
    ),
    "sam": (".sam",),
    "bam": (".bam",),
    "tsv": (".tsv", ".tsv.gz"),
    "bed": (".bed", ".bed.gz"),
    "bedgraph": (".bedgraph", ".bedgraph.gz"),
    "vcf": (".vcf", ".vcf.gz", ".gvcf", ".gvcf.gz"),
    "bcf": (".bcf",),
    "gff": (".gff3", ".gff3.gz"),
    "gtf": (".gtf", ".gtf.gz"),
    "fastq": (".fq", ".fastq", ".fq.gz", ".fastq.gz"),
    "bigwig": (".bw", ".bigWig"),
    "bigbed": (".bigBed",),
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
        keys = keys.split("/")
    d = config
    for k in keys:
        if k is None:
            continue  # ignore None keys
        if k not in d:
            assert not required, 'Mandatory config path "%s" missing' % " > ".join(keys)
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
        file_handle : instance (file/pysam/pyBigWig object)
    """
    if file_extensions is None:
        file_extensions = default_file_extensions
    fh = str(fh)  # convert path to str
    if file_format is None:  # auto detect via file extension
        file_format = guess_file_format(fh, file_extensions)
    # instantiate pysam object
    if file_format == "fasta":
        fh = pysam.Fastafile(fh)  # @UndefinedVariable
    elif file_format == "sam":
        fh = pysam.AlignmentFile(fh, "r")  # @UndefinedVariable
    elif file_format == "bam":
        fh = pysam.AlignmentFile(fh, "rb")  # @UndefinedVariable
    elif file_format == "bed":
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "bedgraph":
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "gtf":
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "gff":
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "tsv":
        fh = pysam.TabixFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "vcf":
        fh = pysam.VariantFile(fh, mode="r")  # @UndefinedVariable
    elif file_format == "bcf":
        fh = pysam.VariantFile(fh, mode="rb")  # @UndefinedVariable
    elif file_format == "fastq":
        fh = gzip.open(fh, "rb") if fh.endswith(".gz") else open(fh, mode="r")
    elif file_format == "bigwig" or file_format == "bigbed":
        fh = pyBigWig.open(fh)
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
    if isinstance(s, str) and s.startswith("chr"):
        return s[3:]
    else:
        return f"chr{s}"


# --------------------------------------------------------------
# genomics helpers :: SAM/BAM specific
# --------------------------------------------------------------


def yield_unaligned_reads(dat_file: str):
    """
    Convenience method that yields FastqRead items representing unaligned reads from either a FASTQ or
    an unaligned BAM file.

    Examples
    --------
    >>> for r in yield_unaligned_reads('unaligned_reads.bam'):
    >>>     print(r)
    >>> for r in yield_unaligned_reads('unaligned_reads.fastq.gz'):
    >>>     print(r)
    """
    if rna.guess_file_format(dat_file) == 'fastq':  # FASTQ file input
        for read in tqdm(rna.it(dat_file)):
            yield read
    elif rna.guess_file_format(dat_file) == 'bam':  # unaligned BAM file input
        for _, read in tqdm(rna.it(dat_file, include_unmapped=True)):
            if not read.is_mapped:
                yield rna.FastqRead(
                    read.query_name,
                    read.query_sequence,
                    ''.join(map(lambda x: chr(x + 33), read.query_qualities)),
                )
    else:
        raise NotImplementedError(f"unsupported format for {dat_file}")


def aligns_to(anno_blocks, read_blocks, min_overlap=0.95):
    """Calculates the fraction of aligned read bases that overlap with the passed annotation and returns True if >= min_frac
    Parameters
    ----------
    anno_blocks : list
        List of annotation blocks (e.g., exons)
    read_blocks : list
        List of read blocks (e.g., [rna.gi(read.reference_name,a+1,b) for a,b in read.get_blocks()])
    min_overlap : float
        The minimum overlap fraction required for a positive result

    Returns
    -------
    bool
        True if the fraction of aligned read bases that overlap with the passed annotation is >= min_frac
    """
    rblen, overlap = 0, 0
    for sb in read_blocks:
        rblen += len(sb)
        for txb in anno_blocks:
            overlap += max(0, sb.overlap(txb))
    return overlap / rblen >= min_overlap


def count_reads(in_file):
    """Counts reads in different file types"""
    ftype = guess_file_format(in_file)
    if ftype == "fastq":
        return count_lines(in_file) / 4.0
    elif ftype == "sam":
        raise NotImplementedError("SAM/BAM file read counting not implemeted yet!")
    else:
        raise NotImplementedError(f"Cannot count reads in file of type {ftype}.")


def get_softclip_seq(
    read: pysam.AlignedSegment, report_seq=False
) -> tuple[Optional[int], Optional[int]]:
    """
    Extracts soft-clipped sequences from the passed read.

    Parameters
    ----------
    read : pysam.AlignedSegment
        The read to extract soft-clipped sequences from.
    report_seq : bool
        If True, the soft-clipped sequences will be returned, otherwise their length is returned.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        A tuple containing the left and right soft-clipped sequences/lengths, respectively. If no soft-clipped
        sequence is found, the corresponding value in the tuple is None.

    Examples
    --------
    >>> r = pysam.AlignedSegment()
    >>> r.cigartuples = [(4, 5), (0, 10)]
    >>> get_softclip_seq(r)
    (5, None)
    """
    left, right = None, None
    for i, (op, l) in enumerate(read.cigartuples):
        if (i == 0) & (op == 4):
            left = read.query_sequence[:l] if report_seq else l
        if (i == len(read.cigartuples) - 1) & (op == 4):
            right = read.query_sequence[l:] if report_seq else l
    return left, right


def get_softclipped_seq_and_qual(read):
    """
    Returns the sequence+qualities of this read w/o softclipped bases
    """
    seq, qual = read.query_sequence, read.query_qualities
    if read.cigartuples is not None:
        op, ln = read.cigartuples[0]
        if op == 4:
            seq = seq[ln:]
            qual = qual[ln:]
        op, ln = read.cigartuples[-1]
        if op == 4:
            seq = seq[:-ln]
            qual = qual[:-ln]
    return seq, qual


def get_covered_contigs(bam_files):
    """Returns all contigs that have some coverage across a set of BAMs.

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


def get_covered_regions(bam_file):
    """Returns all covered regions in a BAM file.
    This is slow as it iterates over all reads in the BAM file and should be used for small files only.
    """
    for c in get_covered_contigs(bam_file):
        all_reg = [loc for loc, _ in rna.it(bam_file, region=rna.gi(c))]
        if len(all_reg) > 0:
            yield all_reg[0] + all_reg[-1]


def downsample_per_chrom(bam_file, max_reads, out_file_bam=None):
    """Simple convenience method that randomly subsamples reads to ensure max_reads per chromosome.
    The resulting BAM file will be sorted and indexed.

    Parameters
    ----------
    bam_file : str
        The input BAM file.
    max_reads : int
        The maximum number of reads to keep per chromosome.
    out_file_bam : str
        The output file name. If None, it will be set to bam_file + '.subsampled_max<max_reads>.bam'.
    """
    samfile = pysam.AlignmentFile(bam_file, "rb", check_sq=False)  # @UndefinedVariable
    if out_file_bam is None:
        out_file_bam = bam_file + '.subsampled_max%i.bam' % max_reads
    samout = pysam.AlignmentFile(out_file_bam, "wb", template=samfile)
    read_indices = {
        i.contig: random.sample(range(i.mapped), min(max_reads, i.mapped))
        for i in samfile.get_index_statistics()
    }
    for c in samfile.references:
        idx = 0
        ri = set(read_indices[c])
        for read in samfile.fetch(c):
            if idx in ri:
                samout.write(read)
            idx += 1
    samout.close()
    try:
        pysam.sort("-o", out_file_bam + '.tmp.bam', out_file_bam)  # @UndefinedVariable
        os.replace(out_file_bam + '.tmp.bam', out_file_bam)
        pysam.index(out_file_bam)  # @UndefinedVariable
    except Exception as e:
        print("error sorting+indexing bam: %s" % e)


def is_paired(bam_file, n=10000):
    """Returns True if the BAM contains PE reads.
    The method samples the first n reads and checks if any of them are paired."""
    assert guess_file_format(bam_file) == "bam", "Input file must be a BAM file."
    for i, (loc, r) in enumerate(rna.it(bam_file)):
        if r.is_paired:
            return True
        if i > n:
            break
    return False


def extract_aligned_reads_from_fastq(
    bam_file,
    fastq1_file,
    fastq2_file=None,
    region=None,
    out_file_prefix=None,
    max_reads=None,
):
    """Extracts reads from one (SE) or two (PE) FASTQ files that are mapped in a BAM file at the provided region.
    If region is None, all reads are considered, if max_reads is set, only the first max_reads reads are extracted.
    Returns the filename(s) of the FASTQ file(s).

    Parameters
    ----------
    bam_file : str
        The input BAM file.
    fastq1_file : str
        The input FASTQ file for the first read.
    fastq2_file : str or None
        The input FASTQ file for the second read (for PE data)
    region : str or None
        The region to extract reads from. If None, all reads are considered.
    out_file_prefix : str or None
        The prefix for the output FASTQ file(s). If None, it will be set to bam_file + '.<region>.<fastq1/2>.fq'.
    max_reads : int or None
        The maximum number of reads to extract. If None, all reads are considered.

    Returns
    -------
    tuple
        The number of written reads (sum if PE reads) and the filename of the FASTQ file if single-end data,
        or both filenames if paired-end data.
    """
    is_paired = fastq2_file is not None
    if out_file_prefix is None:
        out_file_prefix = f"{os.path.splitext(bam_file)[0]}.{rna.slugify(region)}"
    print("prefix", out_file_prefix)
    read_names = {r.query_name for _, r in rna.it(bam_file, region=region)}
    if max_reads is not None:
        read_names = {x for i, x in enumerate(read_names) if i < max_reads}
    ret = []
    wr1, wr2 = 0, 0
    f1_out = f"{out_file_prefix}.R1.fq" if is_paired else f"{out_file_prefix}.fq"
    with open(f1_out, 'wt') as out:
        for r in tqdm(rna.it(fastq1_file)):
            name = r.name.replace('/', ' ').split(" ")[0][1:]
            if name in read_names:
                print(f"{r.name}\n{r.seq}\n+\n{r.qual}", file=out)
                wr1 += 1
    ret.append(rna.bgzip_and_tabix(f1_out, create_index=False))
    if is_paired:
        f2_out = f"{out_file_prefix}.R2.fq"
        with open(f2_out, 'wt') as out:
            for r in tqdm(rna.it(fastq2_file)):
                name = r.name.replace('/', ' ').split(" ")[0][1:]
                if name in read_names:
                    print(f"{r.name}\n{r.seq}\n+\n{r.qual}", file=out)
                    wr2 += 1
        ret.append(rna.bgzip_and_tabix(f2_out, create_index=False))
        print(f"Written {wr1} reads to {f1_out} and {wr2} reads to {f2_out}")
    else:
        print(f"Written {wr1} reads to {f1_out}")
    return (wr1, ret[0]) if len(ret) == 1 else (wr1, ret[0], ret[1])


def get_tx_indices(tx, sequence_type='spliced_sequence'):
    """
    Returns the splice sequence of the transcript, a numpy array with genomic indices of the respective nucleotides in the
    genome and a numpy array with the distances to the next consecutive nucleotide in the genome for each nucleotide in
    the transcript.

    Parameters
    ----------
    tx : rnalib.Transcript
        The transcript object
    sequence_type : str
        The type of sequence to return. Can be 'spliced_sequence' or 'rna_sequence'. Default is 'spliced_sequence'.

    Returns
    -------
    splice_seq : str
        The spliced transcript sequence
    idx : np.array
        The genomic indices of the respective nucleotides in the genome
    idx0 : np.array
        The distances to the next consecutive nucleotide in the genome for each nucleotide in the transcript

    """
    chrom, strand = tx.chromosome, tx.strand
    if sequence_type == 'spliced_sequence':
        splice_seq = tx.spliced_sequence
        if strand == "-":
            idx = np.concatenate(
                [
                    np.array(list(range(ex.start, ex.start + len(ex))))
                    for ex in reversed(tx.exon)
                ]
            )[::-1]
        else:
            idx = np.concatenate(
                [np.array(list(range(ex.start, ex.start + len(ex)))) for ex in tx.exon]
            )
        idx0 = (
            np.array([(a - b - 1) for a, b in pairwise(idx)])
            if strand == "-"
            else np.array([b - a - 1 for a, b in pairwise(idx)])
        )
    elif sequence_type == 'rna_sequence':
        splice_seq = tx.rna_sequence
        idx = np.array(list(range(tx.start, tx.start + len(tx))))
        if strand == "-":
            idx = idx[::-1]
        idx0 = np.array([0] * len(tx))
    else:
        raise ValueError(f"Invalid sequence type: {sequence_type}")
    return splice_seq, idx, idx0


def get_aligned_blocks(
    tx,
    spliced_seq_start: int,
    spliced_seq_end: int,
    splice_seq=None,
    idx=None,
    idx0=None,
    sequence_type='spliced_sequence',
):
    """
    Return the aligned blocks of a read sequence to the genomic sequence.
    The read sequence is defined by the passed transcript and the spliced_seq_start and spliced_seq_end indices.

    Examples
    --------
    >>> t = rna.Transcriptome(...)
    >>> rs, bl = get_aligned_blocks(t['ENST00000674681.1'], 0, 100)

    Parameters
    ----------
    tx : rnalib.Transcript
        The transcript object
    spliced_seq_start : int
        The start index of the read in the spliced transcript sequence
    spliced_seq_end : int
        The end index of the read in the spliced transcript sequence
    splice_seq : str
        The spliced transcript sequence. If None, tx.spliced_sequence is used.
    idx : np.array
        The genomic indices of the respective nucleotides in the genome. If None, it is calculated from the transcript.
    idx0 : np.array
        The distances to the next consecutive nucleotide in the genome for each nucleotide in the transcript.
        If None, it is calculated from the transcript.

    Returns
    -------
    read_seq : str
        The read sequence
    blocks : list
        The (sorted) aligned blocks of the read sequence to the genomic sequence
    """

    chrom, strand = tx.chromosome, tx.strand
    if splice_seq is None or idx is None or idx0 is None:
        splice_seq, idx, idx0 = get_tx_indices(tx, sequence_type=sequence_type)
    read_seq = splice_seq[spliced_seq_start:spliced_seq_end]
    if len(read_seq) < spliced_seq_end - spliced_seq_start:
        return None, None
    read_idx = idx[spliced_seq_start:spliced_seq_end]
    read_idx0 = idx0[spliced_seq_start:spliced_seq_end]
    if sequence_type == 'spliced_sequence':
        blocks = []
        start = read_idx[0]
        if len(np.where(read_idx0 != 0)) > 0:
            for splice_off in np.where(read_idx0 != 0)[0]:
                if splice_off + 1 >= len(read_idx):
                    break  # special case: SJ at the end of the read
                end = read_idx[splice_off]
                if strand == "-":
                    start, end = end, start
                blocks.append(rna.gi(chrom, start, end, strand))
                start = read_idx[splice_off + 1]
        end = read_idx[-1]
        if strand == "-":
            start, end = end, start
        blocks.append(rna.gi(chrom, start, end, strand))
        if strand == "-":
            blocks = blocks[::-1]
    else:
        start, end = read_idx[0], read_idx[-1]
        if strand == "-":
            start, end = end, start
        blocks = [rna.gi(chrom, start, end, strand)]
    return read_seq, blocks


class MismatchProfile(Counter):
    """
    Mismatch profile for sequence error handling.
    """

    def __init__(self, d: dict = None, alpha=None, *args, **kwargs):
        super().__init__(d, *args, **kwargs)
        if alpha is None:
            alpha = {'A', 'C', 'G', 'T'}  # supported alp
        self.alpha = alpha

    @classmethod
    def get_flat_profile(cls, seq_err=0.1 / 100, alpha=None):
        """Returns a flat sequencing error profile with the given mismatch error probability."""
        scale = 1 / (seq_err / 12)
        se = cls(d=None, alpha=alpha)
        for strand in ['+', '-']:
            for ref in se.alpha:
                for alt in se.alpha:
                    if ref != alt:
                        se[(ref, alt, strand)] = 1  # 12 cases -> seq_err / 12 = 1
                    else:
                        se[(ref, alt, strand)] = int(
                            (1 - seq_err) / 4 * scale
                        )  # 4 cases
        return se

    def add(self, ref: str, alt: str, strand: str):
        """Adds a reference/alternative pair to the profile.
        Converts the passed reference/alternative pair to upper case and checks if they are valid.
        """
        ref = ref.upper()
        alt = alt.upper()
        assert (
            ref in self.alpha and alt in self.alpha
        ), f"Invalid ref/alt pair: {ref}/{alt}"
        self[(ref, alt, strand)] += 1

    def __str__(self):
        return "\n".join(
            [f"{ref}>{alt}: {v} ({s})" for (ref, alt, s), v in self.items()]
        )

    def __repr__(self):
        return self.__str__()

    def get_prob(self, ref: str, alt: set = None, strand: str = "+", revcomp=True):
        """Returns the probability for observing the passed reference/alternative pair.
        if alt is None, any of the other bases is assumed.
        """
        if strand == '-' and revcomp:
            ref = rna.reverse_complement(ref)
        if alt is None:
            alt = self.alpha - set(ref)
        else:
            if strand == '-' and revcomp:
                alt = {rna.reverse_complement(a) for a in alt}
        tot = sum([c for (r, a, s), c in self.items() if r == ref and s == strand])
        if tot == 0:
            return 0
        return (
            sum(
                [
                    c
                    for (r, a, s), c in self.items()
                    if r == ref and a in alt and s == strand
                ]
            )
            / tot
        )

    def get_mismatch_prob(self, strand: str):
        """Returns the probability of observing a mismatch."""
        tot = sum([c for (r, a, s), c in self.items() if s == strand])
        if tot == 0:
            return 0
        return sum([c for (r, a, s), c in self.items() if r != a and s == strand]) / tot

    def get_n_mismatches(self):
        """Returns the number of mismatches."""
        return sum([c for (r, a, s), c in self.items() if r != a])

    def to_dataframe(self):
        """Converts the mismatch profile to a DataFrame."""
        return pd.DataFrame.from_records(
            [(a, b, s, c) for (a, b, s), c in self.items()],
            columns=['ref', 'alt', 'strand', 'count'],
        )

    def plot_mm_profile(self, ax=None, prob=True):
        if prob:
            dat = pd.DataFrame.from_records(
                [
                    (f"{ref}>{alt}", strand, self.get_prob(ref, alt, strand))
                    for (ref, alt, strand), c in self.items()
                    if ref != alt
                ],
                columns=['mm', 'strand', 'mm_prob'],
            )
            return sns.barplot(
                dat.sort_values(['strand', 'mm']),
                x='mm',
                y='mm_prob',
                hue='strand',
                ax=ax,
            )
        else:
            dat = pd.DataFrame.from_records(
                [
                    (f"{ref}>{alt}", strand, c)
                    for (ref, alt, strand), c in self.items()
                    if ref != alt
                ],
                columns=['mm', 'strand', 'mm_count'],
            )
            return sns.barplot(
                dat.sort_values(['strand', 'mm']),
                x='mm',
                y='mm_count',
                hue='strand',
                ax=ax,
            )

    def save(self, file):
        """Saves the mismatch profile to a file."""
        self.to_dataframe().to_csv(file, sep="\t", index=False)

    def add_seq_err(self, seq, strand: str = '+'):
        """Adds a sequencing error according to the profile to the passed sequence.
        By convention, the passed sequence should always be in 5'->3' direction (as written in a BAM file).
        If strand is '-', the probabilities are calculated for the reverse complement of the sequence.
        """
        alpha = list(self.alpha)
        ret, n_seqerr = [], 0
        for r in seq:
            a = np.random.choice(
                alpha, 1, p=[self.get_prob(r, a, strand=strand) for a in alpha]
            )[0]
            if a != r:
                n_seqerr += 1
            ret.append(a)
        return ''.join(ret), n_seqerr

    @classmethod
    def load(cls, file):
        """Loads the mismatch profile from a file."""
        se = cls()
        df = pd.read_csv(file, sep="\t")
        for r in df.itertuples():
            se[r.ref, r.alt, r.strand] = r.count
        return se

    @classmethod
    def from_bam(
        cls,
        bam_file,
        features,
        min_cov=10,
        max_mm_frac=0.1,
        max_sample=1e6,
        strand_specific=True,
        alpha=None,
        fasta_file=None,
        disable_progressbar=False,
    ):
        """Creates a mismatch profile from a BAM file.
        The mismatch profile is created by analyzing the passed regions in the BAM file.
        Only positions with a minimum coverage of min_cov are considered.
        Positions with a mismatch fraction > max_mm_frac are skipped.
        The analysis stops after max_mm_cnt mismatches have been counted.

        Parameters
        ----------
        bam_file : str
            The input BAM file.
        features : list
            List of genomic features to analyze (e.g., genes). The features will be randomly permutated and
            analyzed in this order. They must have an associated sequence attribute in order to calculate the
            reference base. If None, all covered regions in the BAM file will be analyzed (slow) and "N" will be
            used as reference base.
        min_cov : int
            The minimum coverage required for a position to be considered.
        max_mm_frac : float
            The maximum mismatch fraction allowed for a position to be considered.
        max_sample : int
            The maximum number of mismatches to count.
        strand_specific : bool
            If True, a strand-specific profile is created.
        alpha : set
            The set of valid nucleotides.
        disable_progressbar : bool
            If True, the progress bar is disabled.
        """
        se = cls(alpha=alpha)
        if features is None:
            reg = list()
            for c in rna.get_covered_regions(bam_file):
                reg += c.split_by_maxwidth(1000)
        else:
            reg = features.copy()
        assert len(reg) > 0, "No features to analyze"
        random.shuffle(reg)
        bam_refdict = rna.RefDict.load(bam_file)
        fit = None if fasta_file is None else rna.it(fasta_file).file
        if fit is None and not hasattr(reg[0], 'sequence'):
            logging.warning(
                "No reference genome file provided. Using 'N' as reference base."
            )
        stats = Counter()
        for g in tqdm(
            reg, desc="analyzing regions", total=len(reg), disable=disable_progressbar
        ):
            if g.chromosome not in bam_refdict:
                logging.warning(
                    f"Skipping region {g}: chromosome not found in BAM file"
                )
                continue
            stats["analysed_regions"] += 1
            # TODO: progressbar should show the number of analyzed mismatches until max_sample is reached
            seq = (
                g.sequence
                if hasattr(g, 'sequence')
                else fit.fetch(g.chromosome, g.start - 1, g.end)
                if fit is not None
                else 'N' * len(g)
            )  # get sequence
            for loc, ac_ss in rna.it(
                bam_file, style='pileup', strand_specific=True, region=g
            ):
                stats["iterated_columns"] += 1
                if ac_ss.total() > min_cov:
                    ref = seq[loc.start - g.start]
                    mm_frac = (
                        sum([c for (b, strand), c in ac_ss.items() if b != ref])
                        / ac_ss.total()
                    )
                    if mm_frac > max_mm_frac:
                        stats["potential _SNP_columns"] += 1
                        continue
                    stats["analysed_columns"] += 1
                    strand = '+' if loc.strand == '+' else '-'
                    for (b, s), c in ac_ss.items():
                        strand = "+" if not strand_specific else s
                        if b in se.alpha:
                            se[(ref, b, strand)] += c
                else:
                    stats["lowcov_columns"] += 1
            if se.get_n_mismatches() >= max_sample:
                break
        if fit is not None:
            fit.close()
        print(stats)
        return se


class BamWriter:
    """A simple helper class for creating custom BAM files.
    The BAM header is initialized from the passed genome FASTA file.
    """

    def __init__(self, genome_fa: str, out_file_bam: str, sort_and_index=True):
        self.fasta = pysam.Fastafile(genome_fa)  # @UndefinedVariable
        self.rd = rna.RefDict.load(self.fasta)
        header = {
            'HD': {'VN': '1.0'},
            'SQ': [{'SN': chrom, 'LN': chrlen} for chrom, chrlen in self.rd.items()],
        }
        if not os.path.isdir(Path(out_file_bam).parent.absolute()):
            os.makedirs(Path(out_file_bam).parent.absolute())
        self.samout = pysam.AlignmentFile(out_file_bam, "wb", header=header)
        self.out_file_bam = out_file_bam
        self.sort_and_index = sort_and_index
        self._stats = Counter()
        print("Writing to ", out_file_bam)

    def write(
        self,
        aligned_blocks: list,
        query_sequence: str = None,
        query_qualities: str = None,
        query_name: str = None,
        mm: list[tuple] = None,
        mapping_quality=255,
        tags=None,
    ):
        """Writes a read to the BAM file."""
        if not isinstance(aligned_blocks, list):
            aligned_blocks = [aligned_blocks]
        read_span = rna.GI.merge(aligned_blocks)  # merge blocks
        assert (
            read_span is not None
        ), f"Could not merge passed aligned blocks {aligned_blocks}"  # checks strand/chrom
        assert (
            read_span.chromosome in self.rd
        ), f"Chromosome {read_span.chrom} not found in reference genome"
        assert read_span.strand is not None, f"Strand not set for read {read_span}"
        aligned_blocks_len = sum([len(b) for b in aligned_blocks])
        if query_sequence is None:
            query_sequence = ''.join(
                [
                    self.fasta.fetch(read_span.chromosome, b.start - 1, b.end)
                    for b in aligned_blocks
                ]
            )
        assert len(query_sequence) == aligned_blocks_len, (
            f"Sequence length {len(query_sequence)} does not match aligned "
            f"blocks length {aligned_blocks_len}"
        )
        if query_qualities is None:
            query_qualities = "~" * len(query_sequence)  # max quality
        assert len(query_qualities) == aligned_blocks_len, (
            f"Quality length {len(query_qualities)} does not match aligned "
            f"blocks length {aligned_blocks_len}"
        )
        NM = 0
        if mm is not None:
            for off, alt in mm:
                query_sequence = query_sequence[:off] + alt + query_sequence[off + 1 :]
                NM += 1
        r = pysam.AlignedSegment()
        r.query_name = (
            f"read{self._stats['reads']}_{read_span.to_file_str()}"
            if query_name is None
            else query_name
        )
        r.query_sequence = query_sequence
        r.query_qualities = pysam.qualitystring_to_array(query_qualities)
        r.flag = 0 if read_span.strand == '+' else 16
        r.reference_id = self.rd.index(read_span.chromosome)
        r.reference_start = read_span.start - 1
        r.mapping_quality = mapping_quality
        if len(aligned_blocks) == 1:
            cigar = [(0, len(aligned_blocks[0]))]
        else:  # more than one block
            cigar = list()
            b = None
            for a, b in pairwise(aligned_blocks):
                cigar.append((0, len(a)))  # M-block
                cigar.append((3, a.distance(b) - 1))  # N-block
            cigar.append((0, len(b)))  # final M-block
        r.cigar = tuple(cigar)
        if tags is None:
            tags = []
        # if mm is not None:
        #     tags += [('NM', NM)]
        r.tags = tags
        self.write_read(r)

    def write_read(self, read: pysam.AlignedSegment):
        self._stats['reads'] += 1
        self.samout.write(read)

    def __enter__(self):
        return self

    def __exit__(self, extype, value, traceback):
        self.close()

    def close(self):
        self.samout.close()
        if self.sort_and_index:
            sort_and_index_bam(self.out_file_bam)


def sort_and_index_bam(bam_file):
    """Sort and index a BAM file"""
    try:
        pysam.sort("-o", bam_file + '.tmp.bam', bam_file)  # @UndefinedVariable
        os.replace(bam_file + '.tmp.bam', bam_file)
        pysam.index(bam_file)  # @UndefinedVariable
    except Exception as e:
        print(f"error sorting+indexing bam: {e}")


def merge_bam_files(
    out_file: str,
    bam_files: list,
    sort_output: bool = False,
    del_in_files: bool = False,
):
    """Merge multiple BAM files, sort and index results.

    Parameters
    ----------
    out_file : str
        The output file name.
    bam_files : list
        A list of input BAM files.
    sort_output : bool
        Whether to sort the output file. Default is False.
    del_in_files : bool
        Whether to delete the input files after merging. Default is False.

    Returns
    -------
    The filename of the merged BAM file.
    """
    if bam_files is None or len(bam_files) == 0:
        logging.error("no input BAM file provided")
        return None
    samfile = pysam.AlignmentFile(bam_files[0], "rb")  # @UndefinedVariable
    with pysam.AlignmentFile(
        out_file + ".unsorted.bam", "wb", template=samfile
    ) as out:  # @UndefinedVariable
        for f in bam_files:
            samfile = None
            try:
                samfile = pysam.AlignmentFile(f, "rb")  # @UndefinedVariable
                for read in samfile.fetch(until_eof=True):
                    out.write(read)
            except Exception as e:
                logging.error(f"error opening bam {f}: {e}")
                raise e
            finally:
                if samfile:
                    samfile.close()
    if sort_output:
        try:
            pysam.sort("-o", out_file, out_file + ".unsorted.bam")  # @UndefinedVariable
            os.remove(out_file + ".unsorted.bam")
            if del_in_files:
                for f in bam_files + [b + ".bai" for b in bam_files]:
                    if os.path.exists(f):
                        os.remove(f)
        except Exception as e:
            logging.error(f"error sorting bam {out_file}: {e}")
            raise e
    else:
        os.rename(out_file + ".unsorted.bam", out_file)
        if del_in_files:
            for f in bam_files + [b + ".bai" for b in bam_files]:
                os.remove(f)
    # index
    try:
        pysam.index(out_file)  # @UndefinedVariable
    except Exception as e:
        logging.error(f"error indexing bam {out_file}: {e}")
    return out_file


def move_id_to_info_field(vcf_in, info_field_name, vcf_out, desc=None):
    """
    move all ID entries from a VCF to a new info field
    """
    if desc is None:
        desc = info_field_name
    vcf = pysam.VariantFile(vcf_in, "r")
    header = vcf.header
    header.add_line("##cmd=move_id_to_info_field()")
    header.add_line(
        '##INFO=<ID=%s,Number=1,Type=String,Description="%s">' % (info_field_name, desc)
    )
    out = pysam.VariantFile(vcf_out, mode="w", header=header)
    for record in vcf.fetch():
        record.info[info_field_name] = ",".join(record.id.split(";"))
        record.id = "."
        out.write(record)
    out.close()


def add_contig_headers(vcf_in, ref_fasta, vcf_out):
    """
    Add missing contig headers to a VCF file
    """
    vcf = pysam.VariantFile(vcf_in, "r")
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
    with h5py.File(fast5_file, "r") as f:
        x = f[rn][path]
        return {k: v for k, v in zip(x.attrs.keys(), x.attrs.values())}


class Timer:
    """Simple class for collecting timings of code blocks.

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
    """Reads a gene name aliases from the passed file.
    Supports the download format from genenames.org and vertebrate.genenames.org.
    Returns an alias dict and a set of currently known (active) gene symbols
    """
    aliases = {}
    current_symbols = set()
    if gene_name_alias_file:
        tab = pd.read_csv(
            gene_name_alias_file,
            sep="\t",
            dtype={"alias_symbol": str, "prev_symbol": str, "symbol": str},
            low_memory=False,
            keep_default_na=False,
        )
        for r in tqdm(
            tab.itertuples(),
            desc="load gene aliases",
            total=tab.shape[0],
            disable=disable_progressbar,
        ):
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
    galias = mg.getgenes(set(gene_ids), filter="symbol,name,taxid")
    id2sym = {
        x["query"]: GeneSymbol(
            x.get("symbol", x["query"]), x.get("name", None), x.get("taxid", None)
        )
        for x in galias
    }
    return id2sym


def calc_3end(tx, width=200, report_partial=True, na_value=None):
    """
    Utility function that returns a (sorted) list of genomic intervals containing the last <width> bases
    of the passed transcript or None if not possible (e.g., if transcript is too short).
    TODO: should report a "Feature" with parent tx so that, e.g., sequence slicing works

    Parameters
    ----------
    tx : rna.Feature
        The transcript to calculate the 3' end for.
    width : int
        The number of bases to return. Default is 200.
    report_partial:
        If True, 3'ends shorter than width will be reported, otherwise the na_value will be returned in this case.
    na_value : any
        The value to return if the 3# end could not be calculated (e.g., if the transcript is too short. Default is
        None.)
    """
    ret = []
    for ex in tx.exon[::-1]:
        if len(ex) < width:
            ret.append(ex.get_location())
            width -= len(ex)
        else:
            s, e = (
                (ex.start, ex.start + width - 1)
                if (ex.strand == "-")
                else (ex.end - width + 1, ex.end)
            )
            ret.append(rna.gi(ex.chromosome, s, e, ex.strand))
            width = 0
            break
    return sorted(ret) if report_partial or (width == 0) else na_value


# --------------------------------------------------------------
# genomics helpers :: VCF specific
# --------------------------------------------------------------


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
    dat = gt.split("/") if "/" in gt else gt.split("|")
    if set(dat) == {"."}:  # no call
        return 0, 0
    dat_clean = [x for x in dat if x != "."]  # drop no-calls
    if set(dat_clean) == {"0"}:  # homref in all called samples
        return 2, 0
    return 2 if len(set(dat_clean)) == 1 else 1, 1


#: A read in a FASTQ file. Print in FASTQ format, e.g., like this: print('\n'.join([r.name, r.seq, '+', r.qual]))
FastqRead = namedtuple("FastqRead", "name seq qual")


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


# --------------------------------------------------------------
# ARCHS4 helpers
# download data from https://maayanlab.cloud/archs4/download.html
# e.g., https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.2.h5
# data
# │ expression            uint32 | (67186, 722425)
# meta
# │ genes
# │   biotype               str    | (67186,)
# │   ensembl_gene_id       str    | (67186,)
# │   symbol                str    | (67186,)
# │ info
# │   author                str    | ()
# │   contact               str    | ()
# │   creation-date         str    | ()
# │   laboratory            str    | ()
# │   version               str    | ()
# │ samples
# │   channel_count         str    | (722425,)
# │   characteristics_ch1   str    | (722425,)
# │   contact_address       str    | (722425,)
# │   contact_city          str    | (722425,)
# │   contact_country       str    | (722425,)
# │   contact_institute     str    | (722425,)
# │   contact_name          str    | (722425,)
# │   contact_zip           str    | (722425,)
# │   data_processing       str    | (722425,)
# │   extract_protocol_ch1  str    | (722425,)
# │   geo_accession         str    | (722425,)
# │   instrument_model      str    | (722425,)
# │   last_update_date      str    | (722425,)
# │   library_selection     str    | (722425,)
# │   library_source        str    | (722425,)
# │   library_strategy      str    | (722425,)
# │   molecule_ch1          str    | (722425,)
# │   organism_ch1          str    | (722425,)
# │   platform_id           str    | (722425,)
# │   readsaligned          uint32 | (722425,)
# │   relation              str    | (722425,)
# │   sample                str    | (722425,)
# │   series_id             str    | (722425,)
# │   singlecellprobability  float64 | (722425,)
# │   source_name_ch1       str    | (722425,)
# │   status                str    | (722425,)
# │   submission_date       str    | (722425,)
# │   taxid_ch1             str    | (722425,)
# │   title                 str    | (722425,)
# │   type                  str    | (722425,)
# --------------------------------------------------------------
def get_sample_meta_keys(file="data/human_gene_v2.2.h5"):
    with h5py.File(file, "r") as f:
        return list(f["meta/samples"].keys())


def get_archs4_sample_dict(file="data/human_gene_v2.2.h5", remove_sc=True):
    """Returns a dict of GSM ids and sample indices.
    If remove_sc is True (default), then single cell samples are removed.
    Example
    -------
    >>> nosc_samples = get_archs4_sample_dict()
    >>> ten_random_ids = random.sample(list(nosc_samples), 10)
    """
    with h5py.File(file, "r") as f:
        gsm_ids = [x.decode("UTF-8") for x in np.array(f["meta/samples/geo_accession"])]
        if remove_sc:
            singleprob = np.array(f["meta/samples/singlecellprobability"])
            idx = sorted(list(np.where(singleprob < 0.5)[0]))
        else:
            idx = sorted(range(len(gsm_ids)))
    return {gsm_ids[i]: i for i in idx}


def get_sample_metadata(
    samples,
    sample_dict=None,
    keys=None,
    file="data/human_gene_v2.2.h5",
    disable_progressbar=False,
):
    if sample_dict is None:
        sample_dict = get_archs4_sample_dict(file)  # all samples (non sc)
    if keys is None:
        keys = get_sample_meta_keys(file)  # all keys
    sample_idx = sorted([sample_dict[s] for s in samples])
    with h5py.File(file, "r") as f:
        res = []
        for k in (pbar := tqdm(keys, disable=disable_progressbar)):
            pbar.set_description(f"Accessing column {k}")
            if k in [
                "submission_date",
                "last_update_date",
            ]:  # date conversion. filter with df[df[
                # 'last_update_date']>'2022']
                from datetime import datetime

                res.append(
                    np.array(
                        [
                            datetime.strptime(d.decode("utf-8"), "%b %d %Y")
                            for d in f["meta/samples/%s" % k][sample_idx]
                        ]
                    )
                )
            else:
                res.append(np.array(f["meta/samples/%s" % k][sample_idx]))
    res = pd.DataFrame(res, index=keys, columns=samples).T
    return res


# --------------------------------------------------------------
# div
# --------------------------------------------------------------


def random_sample(conf_str, rng=np.random.default_rng(seed=None)):
    """
    Draws random samples from a distribution that is configured by the passed (configuration) string.
    If the conf_str is numeric (i.e., a constant value), it is cast to a float and returned.
    Supported distributions are all numpy.random functions (e.g., normal, uniform, etc.).
    Examples
    --------
    >>> random_sample(12), random_sample(12.0), random_sample('12') # constant value
    >>> random_sample('uniform(1,2,100)') # 100 random numbers, uniformly sampled from [1; 2]
    >>> random_sample('normal(3, 0.8, size=(2, 4))') # 2 x 4 random numbers from a normal distribution around 3 with sd 0.8
    >>> for x in random_sample("normal(3000,10,size=(5,1000))"):
    >>>     plt.hist(x, 30, density=True, histtype=u'step') # plot 5 histograms # noqa
    """
    if isinstance(conf_str, numbers.Number) or conf_str.isnumeric():
        return float(conf_str)
    tok = conf_str.split("(", 1)
    supported_dist = [x for x in dir(rng) if not x.startswith("_")]
    if len(tok) != 2 or tok[0] not in supported_dist:
        raise NotImplementedError(f"Unsupported distribution config: {conf_str}")
    return eval(f"rng.{conf_str}")


def random_intervals(
    chromosomes=('1',), start_range=range(0, 1000), len_range=range(1, 100), n=10
):
    """Generates random genomic intervals.
    Parameters
    ----------
    chromosomes : list
        The chromosomes to sample from.
    start_range : range
        The range of start positions.
    len_range : range
        The range of lengths.
    n : int
        The number of intervals to generate.
    Returns
    -------
    A sorted list of genomic intervals.
    """
    refdict = rna.RefDict({c: None for c in chromosomes})
    chroms = random.choices(chromosomes, k=n)
    starts = random.choices(start_range, k=n)
    lens = random.choices(len_range, k=n)
    return rna.GI.sort(
        [rna.gi(c, s, s + l) for c, s, l in zip(chroms, starts, lens)], refdict
    )


def execute_screencast(
    command_file,
    col=True,
    default_delay=1,
    console_prompt=">>> ",
    min_typing_delay=0.001,
    max_typing_delay=0.05,
):
    """
    Executes the passed command file in a python shell.
    Example
    -------
    >>> execute_screencast('screencast.txt')

    To record a screencast, create a file with the commands to execute and add comments with the delay in
    seconds. Then run the following command to record the screencast (make sure that rnalib is installed):
    >>> terminalizer record -c myconfig.yml --skip-sharing --command "python3 -c \"import rnalib as rna; rna.execute_screencast('myscript.py')\"" ${name} # noqa
    where myconfig.yml is a terminalizer configuration file and myscript.py is the python script to execute.

    Todo: Use ast to parse and highlight the commands

    """

    def type_str(string, min_delay=0.01, max_delay=0.1):
        """emulate typing a string with random delays"""
        for char in string:
            print(char, end="", flush=True)
            if not char.isspace():
                time.sleep(random.uniform(min_delay, max_delay))
        print()

    def colored_cmd(
        cmd,
        comment,
        col=True,
        def_col="green",
        eq_col="yellow",
        comment_col="red",
        banner_col="magenta",
    ):
        """colorize a command string.
        Todo: use ast to parse and highlight the commands
        """
        if len(cmd) == 0:  # a 'banner' comment
            return colored(f"#{comment}", color=banner_col) if col else comment
        cmd = (
            colored("=", eq_col).join([colored(t, def_col) for t in cmd.split("=")])
            if col
            else cmd
        )
        if len(comment) > 0:
            cmd += (
                colored(f" # {comment}", color=comment_col) if col else f" # {comment}"
            )
        # ast.dump(ast.parse(cmd)) # note that AST does not retain comments or whitespace
        return cmd

    lines = []
    blank = len(console_prompt) * " "
    current_delay = default_delay
    with open(command_file, mode="rt") as cin:
        for line in cin:
            if line.endswith("\n"):
                line = line[:-1]  # remove newline
            if line == "":
                continue  # skip empty lines
            # get comment
            line, comment = line.split("#") if "#" in line else (line, "")
            # parse time from comment, e.g., "sometext (t=0.5)" -> [0.5]
            if len(comment) > 0:
                tok = re.findall(r"[-+]?t=(\d*\.*\d+)\)", comment)
                if len(tok) == 1:
                    comment = comment.replace(f"(t={tok[0]})", "")
                    current_delay = float(tok[0])
            lines.append((line, comment, current_delay))
    commands = []  # tuple of (cmd, prompt, delay)
    for line, comment, delay in lines:
        cmd = line.rstrip()
        prompt = colored_cmd(cmd, comment, col=col)
        if len(line) > len(line.lstrip()):  # attach to previous command
            c, p, d = commands[-1]
            c += f"\n{cmd}"
            p += f"\n{blank}{prompt}"
            commands[-1] = (c, p, d)
            continue
        commands.append((cmd, prompt, delay))
    # run
    print(console_prompt, end="", flush=True)
    time.sleep(1)
    for cmd, prompt, delay in commands:
        type_str(prompt, min_delay=min_typing_delay, max_delay=max_typing_delay)
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(cmd)
        output = stdout.getvalue()
        if len(output) > 0:
            print(output)
        print(console_prompt, end="", flush=True)
        time.sleep(delay)
