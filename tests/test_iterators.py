"""
Tests for iterators
"""
import itertools
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pybedtools
import pyranges as pr
import pysam
import pytest
from sortedcontainers import SortedList, SortedSet

import rnalib
from rnalib import open_file_obj, TagFilter, FastaIterator, TabixIterator, \
    BedGraphIterator, BedIterator, gt2zyg, VcfIterator, GFF3Iterator, PandasIterator, BioframeIterator, MemoryIterator, \
    PybedtoolsIterator, DEFAULT_FLAG_FILTER, ReadIterator, FastPileupIterator, BlockStrategy, BlockLocationIterator, \
    SyncPerPositionIterator, AnnotationIterator, FastqIterator, gi, ReferenceDict, PyrangesIterator
from rnalib.testdata import get_resource
from rnalib.utils import toggle_chr

assert rnalib.__RNALIB_TESTDATA__ is not None, ("Please set rnalib.__RNALIB_TESTDATA__ variable to the testdata "
                                                "directory path")
def merge_yields(lst) -> (gi, tuple):
    """ Takes an enumeration of (loc,payload) tuples and returns a tuple (merged location, payloads) """
    l1, l2 = zip(*lst)
    mloc = gi.merge(list(l1))
    return mloc, l2


def from_str(s):
    return [gi.from_str(x) for x in s.split(',')]


@pytest.fixture(autouse=True)
def testdata() -> (dict, pd.DataFrame):
    # Some overlap tests
    # .........1........  ....2......
    # |-a-|
    #      |-b-|
    #   |-c-|
    #           |-d-|
    #                         |-e-|
    #                     |-feature--------|
    #                         |-g-|
    #                         |--h---|
    d = {
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'feature': gi('2', 1, 50),
        'g': gi('2', 21, 30),
        'h': gi('2', 21, 50),
    }
    df = pd.DataFrame([(loc.chromosome, loc.start, loc.end, name) for name, loc in d.items()],
                      columns=['Chromosome', 'Start', 'End', 'Name']).reset_index(
        drop=True)  # note: this test df is *not* sorted!
    return d, df


def loc_list(s):
    return [gi.from_str(x) for x in s.split(',')]


def test_MemoryIterator(testdata):
    d, df = testdata
    assert len(MemoryIterator(d).to_list()) == len(d)
    assert MemoryIterator(d, '2', 10, 20).to_list() == [(gi.from_str('2:1-50'), 'feature')]
    # with aliasing
    assert MemoryIterator(d, 'chr2', 10, 20, fun_alias=toggle_chr).to_list() == [(gi.from_str('chr2:1-50'), 'feature')]


def test_to_dataframe(testdata):
    d, df = testdata
    df = df.sort_values(['Chromosome', 'Start', 'End']).reset_index(drop=True)
    df2 = MemoryIterator(d).to_dataframe(fun_col=('Name',)).drop(columns=['Strand'])
    assert df.equals(df2)
    # build a dataframe with read names and mismatch counts from BAM
    df2 = ReadIterator(get_resource('small_example_bam'), report_mismatches=True, min_base_quality=10). \
        to_dataframe(fun=lambda loc, item, fun_col, default_value: [item[0].query_name, len(item[1])],
                     fun_col=('ReadName', 'MM'),
                     dtypes={'MM': float})
    # assert proper dtypes
    assert df2['MM'].dtype == np.dtype('float64')
    # assert number of reads with at least 2 MM:
    assert len(df2.query("MM>1.0").index) == 435


def test_FastaIterator():
    fasta_file = get_resource('ACTB+SOX2_genome')
    # read seq via pysam
    with pysam.Fastafile(fasta_file) as fh:
        ref = {c: fh.fetch(c) for c in fh.references}
    # consume all
    all_chrom = ''.join([s for _, s in FastaIterator(fasta_file, 'chr3', width=1, step=1).to_list()])
    assert (all_chrom == ref['chr3'])
    # with aliasing
    all_chrom = ''.join([s for _, s in FastaIterator(fasta_file, '3', width=1, step=1, fun_alias=toggle_chr).to_list()])
    assert (all_chrom == ref['chr3'])
    # some edge cases where the provided sequence is shorter than the requested window size
    assert FastaIterator(fasta_file, 'chr7', 3, 6, width=5, step=3, padding=False).to_list() == [
        (gi('chr7', 3, 7), 'GTGCN')]  # 5-mer from region of size 4, wo padding
    assert FastaIterator(fasta_file, 'chr7', 3, 6, width=5, step=3, padding=True).to_list() == [
        (gi.from_str('chr7:1-5'), 'NNGTG'),
        (gi.from_str('chr7:4-8'), 'TGCNN')]  # 5-mer from region of size 4, wiwth padding
    # consume in tiling windows
    tiled = ''.join([s for _, s in FastaIterator(fasta_file, 'chr7', None, None, width=3, step=3).to_list()])
    assert (tiled[:-1] == ref[
        'chr7'])  # NOTE cut last char in tiled as it is padded by a single N (as len(ref['chr7']) % 3 = 2)
    # get the first 10 5-mers with and w/o padding
    fivemers = [s for _, s in FastaIterator(fasta_file, 'chr7', None, None, width=5, step=2, padding=False).to_list()][
               :10]
    assert fivemers == ['TTGTG', 'GTGCC', 'GCCAT', 'CATTA', 'TTACA', 'ACACT', 'ACTCC', 'TCCAG', 'CAGCC', 'GCCTG']
    fivemers = [s for _, s in FastaIterator(fasta_file, 'chr7', None, None, width=5, step=2, padding=True).to_list()][
               :10]
    assert fivemers == ['NNTTG', 'TTGTG', 'GTGCC', 'GCCAT', 'CATTA', 'TTACA', 'ACACT', 'ACTCC', 'TCCAG', 'CAGCC']
    # get 11-mers with padding
    ctx = [s for _, s in FastaIterator(fasta_file, 'chr7', 1, 10, width=11, step=1, padding=True)]
    assert ctx[:5] == ['NNNNNTTGTGC', 'NNNNTTGTGCC', 'NNNTTGTGCCA', 'NNTTGTGCCAT', 'NTTGTGCCATT']
    assert ''.join([x[5] for x in ctx]) == ref['chr7'][:10]


def test_TabixIterator():
    vcf_file = get_resource('test_snps_vcf')
    bed_file = get_resource('test_bed')
    bedg_file = get_resource('test_bedgraph')  # includes track header
    # read VCF file as TSV
    ti = TabixIterator(vcf_file, region='1:1-10', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert (merge_yields(ti.to_list())[0] == gi('1', 1, 10))
    ti = TabixIterator(vcf_file, chromosome='1', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert (merge_yields(ti.to_list())[0] == gi('1', 1, 20))
    ti = TabixIterator(vcf_file, chromosome='2', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert len([(loc, t) for loc, t in ti.to_list()]) == 1
    with pytest.raises(AssertionError) as e_info:
        TabixIterator(vcf_file, 'unknown_contig', 5, 10)
    print(f'Expected assertion: {e_info}')
    # BED file with added 'chr' prefix
    ti = TabixIterator(bed_file, 'chr1', 1, 10, coord_inc=[1, 0], fun_alias=toggle_chr)
    assert (merge_yields(ti.to_list())[0] == gi('chr1', 6, 15))  # start is 0-based, end is 1-based
    # bedgraph file but parsed as Tabixfile
    # 0.042+0.083+0.125+0.167+0.208+4*0.3+0.7*2+0.8*2+0.1*20 == 6.824999999999999
    assert sum(
        [float(r[3]) * len(loc) for loc, r in TabixIterator(bedg_file, coord_inc=[1, 0]).to_list()]) == pytest.approx(
        6.825)
    # test open intervals
    assert len(TabixIterator(bed_file, region=gi('1'), coord_inc=[0, 0], pos_indices=[0, 1, 1]).to_list()) == 2


def test_GFF3Iterator():
    stats = Counter()
    for loc, info in GFF3Iterator(get_resource('gencode_gff'), 'chr7'):
        stats[info['feature_type']] += 1
    assert stats == {'exon': 107,
                     'CDS': 59,
                     'five_prime_UTR': 32,
                     'transcript': 25,
                     'three_prime_UTR': 19,
                     'start_codon': 16,
                     'stop_codon': 12,
                     'gene': 3}
    # GTF with aliasing
    stats = Counter()
    for loc, info in GFF3Iterator(get_resource('gencode_gtf'), '7', fun_alias=toggle_chr):
        stats[info['feature_type']] += 1
    assert stats == Counter({'exon': 107,
                             'CDS': 59,
                             'UTR': 51,
                             'transcript': 25,
                             'start_codon': 16,
                             'stop_codon': 12,
                             'gene': 3})


def test_PandasIterator(testdata):
    d, df = testdata
    it = PandasIterator(df, 'Name', coord_columns=['Chromosome', 'Start', 'End', 'Strand'], coord_off=(0, 0))
    assert {k: v for v, k in it} == d
    # with aliasing
    it = PandasIterator(df, 'Name', fun_alias=toggle_chr, coord_columns=['Chromosome', 'Start', 'End', 'Strand'],
                        coord_off=[0, 0])
    d1 = {n: gi('chr' + l.chromosome, l.start, l.end, l.strand) for n, l in d.items()}
    assert {k: v for v, k in it} == d1



def test_BlockLocationIterator(testdata):
    with BlockLocationIterator(TabixIterator(get_resource('test_bed'), coord_inc=[1, 0], fun_alias=toggle_chr),
                               strategy=BlockStrategy.OVERLAP) as it:
        locs = [loc for loc, _ in it]
        assert locs == loc_list('chr1:6-15,chr2:10-150')
    d, df = testdata
    with PandasIterator(df, 'Name', coord_columns=['Chromosome', 'Start', 'End', 'Strand'],
                        coord_off=(0, 0)) as it:
        assert [loc for loc, _ in BlockLocationIterator(it, strategy=BlockStrategy.OVERLAP)] == loc_list(
            '1:1-20,1:30-40 ,2:1-50')
        assert BlockLocationIterator(it).to_list()[-1][1][1] == ['e', 'g', 'h']  # same start coord
        assert BlockLocationIterator(it, strategy=BlockStrategy.RIGHT).to_list()[-2][1][1] == ['e',
                                                                                               'g']  # same end coord
    # with chr toggle
    with PandasIterator(df, 'Name', coord_columns=['Chromosome', 'Start', 'End', 'Strand'], coord_off=(0, 0),
                        fun_alias=toggle_chr) as it:
        assert BlockLocationIterator(it, strategy=BlockStrategy.RIGHT).to_list()[-2][1][1] == ['e',
                                                                                               'g']  # with aliasing
    right_sorted = BlockLocationIterator(PandasIterator(
        df.sort_values(['Chromosome', 'End']), 'Name', is_sorted=True,
        coord_columns=['Chromosome', 'Start', 'End', 'Strand'], coord_off=(0, 0)),
        strategy=BlockStrategy.RIGHT)
    assert [x[1] for _, x in right_sorted.to_list()[-2:]] == [['e', 'g'], ['feature', 'h']]


def test_AnnotationIterator(testdata):
    # simple test
    a = {
        'A': gi('chr1', 1, 4),
        'B': gi('chr1', 1, 5),
        'C': gi('chr3', 5, 6),
    }
    b = {
        'D1': gi('chr1', 1, 4),
        'D2': gi('chr1', 1, 4),
        'E': gi('chr2', 1, 5),
        'F': gi('chr3', 1, 5),
        'G': gi('chr4', 5, 6),
    }

    def format_results(lst):
        return [(anno, [x[1] for x in i2]) for loc, (anno, i2) in lst]

    assert format_results(AnnotationIterator(MemoryIterator(a), MemoryIterator(b)).to_list()) == \
           [('A', ['D1', 'D2']),
            ('B', ['D1', 'D2']),
            ('C', ['F'])]
    # iterate only chr1
    assert format_results(AnnotationIterator(MemoryIterator(a, chromosome='chr1'), MemoryIterator(b)).to_list()) == \
           [('A', ['D1', 'D2']),
            ('B', ['D1', 'D2'])]

    # multiple iterators and labels
    with AnnotationIterator(MemoryIterator(a), [MemoryIterator(b), MemoryIterator(b)], ['A', 'B']) as it:
        assert ([[i.data.anno, [x.data for x in i.data.A], [x.data for x in i.data.B]] for i in it.to_list()]) == \
               [['A', ['D1', 'D2'], ['D1', 'D2']], ['B', ['D1', 'D2'], ['D1', 'D2']], ['C', ['F'], ['F']]]

    # Annotate intervals from a bed file with values from a bedgraph file
    # overlap with bedgraph file, calculate overlap and sum scores
    # NOTE bedgraph file contains interval (1:7-10, 0.3)
    with AnnotationIterator(BedIterator(get_resource('test_bed')),
                            BedGraphIterator(get_resource('test_bedgraph')),
                            labels=['scores']) as it:
        assert (
               [(i.anno.name, sum([x.data * loc.overlap(x.location) for x in i.scores])) for loc, i in it.to_list()]) == \
               [('int1', 1.408), ('int2', 0.3), ('int3', 0)]
        print('stats:', it.stats)

    # envelop scenario:
    # it:    |----A-----------------|
    #         |----X--|   |---Y-|
    # an:          |-1-||---2-----|
    # test whether this gives same results for X/Y with and w/o A
    a1 = {
        'A': gi.from_str('1:1-20'),
        'X': gi.from_str('1:2-5'),
        'Y': gi.from_str('1:7-10')
    }
    a2 = a1.copy()
    del a2['A']
    b = {
        '1': gi.from_str('1:4-6'),
        '2': gi.from_str('1:6-15'),
    }
    res1 = {i.location: i.data for i in (AnnotationIterator(MemoryIterator(a1), MemoryIterator(b)).to_list())}
    res2 = {i.location: i.data for i in (AnnotationIterator(MemoryIterator(a2), MemoryIterator(b)).to_list())}
    # test whether omitting 'A' leads to different results!
    for k in res1.keys() & res2.keys():
        assert res1[k] == res2[k]
    # real world example for such a situation:
    # roi = gi('chr20', 653200, 654306)
    # bedit=BedGraphIterator('/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/mappability/GRCh38.k24.umap.bedgraph.gz',
    #                  chromosome=roi.chromosome, start=roi.start, end=roi.end)
    # gffit=GFF3Iterator('/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/annotation/gencode.v39.annotation.sorted.gff3.gz',
    #                    chromosome=roi.chromosome, start=roi.start, end=roi.end)
    # with AnnotationIterator(gffit, [bedit]) as it:
    #     for loc,dat in it:
    #         print(loc, dat.anno['ID'], dat.it0)


def test_SyncPerPositionIterator( testdata):
    # simple test
    a = from_str("1:1-4,1:1-5,1:2-3,1:2-4,1:6-7,2:5-6")
    for pos, (i1, i2) in SyncPerPositionIterator([MemoryIterator(a), MemoryIterator(a.copy())]):
        assert i1 == i2

    # simple test 2
    a = {
        'A': gi('chr1', 1, 4),
        'B': gi('chr1', 1, 5),
        'C': gi('chr1', 2, 3)
    }
    b = {
        'D': gi('chr1', 2, 4),
        'E': gi('chr1', 6, 7),
        'F': gi('chr2', 5, 6),
    }
    # check whether all positions are iterated
    assert [(p, [x[1] for x in i1], [x[1] for x in i2]) for p, (i1, i2) in
            SyncPerPositionIterator([MemoryIterator(a), MemoryIterator(b)]).to_list()] == \
           [(gi.from_str('chr1:1-1'), ['A', 'B'], []),
            (gi.from_str('chr1:2-2'), ['A', 'B', 'C'], ['D']),
            (gi.from_str('chr1:3-3'), ['A', 'B', 'C'], ['D']),
            (gi.from_str('chr1:4-4'), ['A', 'B'], ['D']),
            (gi.from_str('chr1:5-5'), ['B'], []),
            (gi.from_str('chr1:6-6'), [], ['E']),
            (gi.from_str('chr1:7-7'), [], ['E'])]

    assert [(p, [x[1] for x in i1], [x[1] for x in i2]) for p, (i1, i2) in
            SyncPerPositionIterator([MemoryIterator(a), MemoryIterator(b)],
                                    refdict=MemoryIterator(b).refdict).to_list()] == \
           [(gi.from_str('chr1:1-1'), ['A', 'B'], []),
            (gi.from_str('chr1:2-2'), ['A', 'B', 'C'], ['D']),
            (gi.from_str('chr1:3-3'), ['A', 'B', 'C'], ['D']),
            (gi.from_str('chr1:4-4'), ['A', 'B'], ['D']),
            (gi.from_str('chr1:5-5'), ['B'], []),
            (gi.from_str('chr1:6-6'), [], ['E']),
            (gi.from_str('chr1:7-7'), [], ['E']),
            (gi.from_str('chr2:5-5'), [], ['F']),
            (gi.from_str('chr2:6-6'), [], ['F'])]

    # complex test with random dataset
    @dataclass(frozen=True)
    class test_feature(gi):
        feature_id: str = None  # a unique id
        feature_type: str = None  # a fetaur etype (e.g., exon, intron, etc.)
        name: str = None  # an optional name
        parent: object = None  # an optional parent

        @classmethod
        def from_gi(cls, loc, feature_id=None, feature_type=None, name=None, parent=None):
            """  """
            return cls(loc.chromosome, loc.start, loc.end, loc.strand, feature_id, feature_type, name, parent)

        def location(self):
            """Returns a genomic interval representing the genomic location of this feature."""
            return gi(self.chromosome, self.start, self.end, self.strand)

    class SyncPerPositionIteratorTestDataset:
        """ 2nd, slow implementation of the sync algorithm for testing"""

        def __init__(self, seed=None, n_it=3, n_pos=10, n_chr=2, n_int=5):
            self.seed = seed
            if seed:
                random.seed(seed)
            self.dat = {}
            self.minmax = {}
            for it in range(n_it):
                self.dat[f'it{it}'] = {}
                for chrom in range(n_chr):
                    self.dat[f'it{it}'][f'c{chrom}'] = self.create_rnd_int(it, f'c{chrom}', n_int, n_pos)

        def __repr__(self):
            return f"SyncPerPositionIteratorTestDataset({self.seed})"

        def create_rnd_int(self, it, chrom, n_int, n_pos):
            random.seed(self.seed)
            ret = []
            for i in range(random.randrange(n_int)):
                start = random.randrange(n_pos)
                end = random.randrange(start, n_pos)
                g = test_feature.from_gi(gi(chrom, start, end), feature_id=it,
                                         name=f'it{it}_{chrom}:{start}-{end}_{len(ret)}')
                if g.chromosome not in self.minmax:
                    self.minmax[g.chromosome] = range(g.start, g.end)
                self.minmax[g.chromosome] = range(min(self.minmax[g.chromosome].start, g.start),
                                                  max(self.minmax[g.chromosome].stop, g.end))
                ret.append(g)
            return list(sorted(ret))

        def expected(self):
            ret = {}
            for chrom in sorted(self.minmax):
                for p in range(self.minmax[chrom].start, self.minmax[chrom].stop + 1):
                    pos = gi(chrom, p, p)

                    found = SortedList()
                    for i, d in enumerate(self.dat.values()):
                        for g in d[chrom]:
                            if g.overlaps(pos):
                                found.add(g.name)
                    ret[pos] = found
            return ret

        def found(self):
            """ Iterate with SyncPerPositionIterator() over MemoryIterators """
            itdata = {}
            for it in self.dat:
                gis = []
                for c in self.dat[it]:
                    gis.extend(self.dat[it][c])
                itdata[it] = gis
            ret = {}
            for loc, item in SyncPerPositionIterator([MemoryIterator(itdata[it]) for it in itdata]):
                ret[loc] = [it[0][0].name for it in item if len(it) > 0]
            return ret

    # test with random datasets
    found_differences = set()
    for seed in range(0, 100):
        print(f"======================================={seed}============================")
        t = SyncPerPositionIteratorTestDataset(seed)
        # print('found', t.found())
        # print('expected', t.expected())
        assert len(t.found()) == len(t.expected()), f"invalid length for {t}, {len(t.found())} != {len(t.expected())}"
        for a, b in zip(t.found(), t.expected()):
            if a != b:
                if SortedSet(a[1]) != SortedSet(b[1]):
                    found_differences.add(seed)
    assert len(found_differences) == 0
    # use more intervals, iterators, chromosomes; heavy overlaps
    found_differences = set()
    for seed in range(0, 10):
        print(f"======================================={seed}============================")
        t = SyncPerPositionIteratorTestDataset(seed, n_it=5, n_pos=100, n_chr=5, n_int=500)
        assert len(t.found()) == len(t.expected()), f"invalid length for {t}, {len(t.found())} != {len(t.expected())}"
        for a, b in zip(t.found(), t.expected()):
            if a != b:
                if SortedSet(a[1][0]) != SortedSet(b[1][0]):
                    found_differences.add(seed)
    assert len(found_differences) == 0
    # for seed in found_differences:
    #     t=SyncPerPositionIteratorTestDataset(seed)
    #     print(f"differences in {t}")
    #     for a, b in zip(t.found(), t.expected()):
    #         if a != b:
    #             print('>', a, b)


def test_PyrangesIterator():
    exons, cpg = pr.data.exons(), pr.data.cpg()
    # get exons with same start but different end coords
    res = []
    for mloc, (locs, ex) in BlockLocationIterator(PandasIterator(exons.df, 'Name')):
        endpos = {loc.end for loc in locs}
        if len(endpos) > 1:
            res += [(mloc, (locs, ex))]
    assert len(res) == 5
    # second test
    with PyrangesIterator(get_resource('pybedtools_snps')) as it:
        stats = Counter([dat.Strand for loc, dat in it])
    assert stats == Counter({'+': 740893, '-': 59107})


@dataclass
class MockRead:
    tags: dict

    def has_tag(self, tag):
        return tag in self.tags

    def get_tag(self, tag):
        return self.tags.get(tag)


def test_TagFilter():
    assert TagFilter('xx', [1, 12, 13], False).filter(MockRead({'xx': 12}))  # filter if values is found
    assert not TagFilter('xx', [12], False, inverse=True).filter(
        MockRead({'xx': 12}))  # inverse filter: filter if values is not found!


def test_ReadIterator():
    with ReadIterator(get_resource('rogue_read_bam'), 'SIRVomeERCCome') as it:
        for _ in it:
            pass
        assert it.stats['yielded_items', 'SIRVomeERCCome'] == 1
    stats = {x: Counter() for x in ['all', 'def', 'mq20', 'tag']}
    with open_file_obj(get_resource('small_example_bam')) as bam:
        for chrom in ReferenceDict.load(bam):
            with ReadIterator(bam, chrom, flag_filter=0) as it:
                it.to_list()
                stats['all'].update(it.stats)
            with ReadIterator(bam, chrom) as it:
                it.to_list()
                stats['def'].update(it.stats)
            with ReadIterator(bam, chrom, min_mapping_quality=20) as it:
                it.to_list()
                stats['mq20'].update(it.stats)
            with ReadIterator(bam, chrom, tag_filters=[TagFilter('MD', ['100'])]) as it:
                it.to_list()
                stats['tag'].update(it.stats)
    # print( stats['tag'])
    assert stats['all']['yielded_items', '1'] == 31678  # samtools view -c small_example.bam -> 31678
    assert stats['def']['yielded_items', '1'] == 21932  # samtools view -c small_example.bam -F 3844 -> 21932
    assert stats['mq20']['yielded_items', '1'] == 21626  # samtools view -c small_example.bam -F 3844 -q 20 -> 21626
    assert stats['tag'][
               'yielded_items', '1'] == 7388  # samtools view small_example.bam -F 3844 | grep -v "MD:Z:100" | wc -l -> 7388
    # count t/c mismatches
    tc_conv = {}
    for _, (r, mm) in ReadIterator(get_resource('small_example_bam'), report_mismatches=True, min_base_quality=10):
        if len(mm) > 0:
            is_rev = not r.is_reverse if r.is_read2 else r.is_reverse
            refc = "A" if is_rev else "T"
            altc = "G" if is_rev else "C"
            mm_tc = [(off, pos1, ref, alt) for off, pos1, ref, alt in mm if ref == refc and alt == altc]
            if len(mm_tc) > 0:
                tc_conv[r.query_name, not r.is_read2] = mm_tc
    # overlapping mate pair: both  contain T/C snp
    assert tc_conv['HWI-ST466_135068617:8:2209:6224:33460', True] == [(71, 22432587, 'T', 'C')]
    assert tc_conv['HWI-ST466_135068617:8:2209:6224:33460', False] == [(29, 22432587, 'T', 'C')]
    #  a read with 2 A/G conversions
    assert tc_conv['HWI-ST466_135068617:8:2316:4251:54002', False] == [(2, 22443997, 'A', 'G'), (5, 22444000, 'A', 'G')]
    # test aliasing
    res = ReadIterator(get_resource('small_example_bam'), 'chr1', fun_alias=toggle_chr).to_list()
    assert len(res) == 21932
    assert res[0].location.chromosome == 'chr1'
    # TODO add data from /groups/.../ref/_delme_testdata/smallbams/


def slow_pileup(bam, chrom, start, stop):
    """ Runs pysam pileup for reference """
    ac = Counter()
    for pu in bam.pileup(contig=chrom, start=start - 1, stop=stop - 1, flag_filter=DEFAULT_FLAG_FILTER,
                         truncate=True, mark_ends=True, add_indels=True, min_base_quality=0, min_mapping_quality=0,
                         ignore_overlaps=False, ignore_orphans=False,
                         max_depth=100000):
        pos = pu.reference_pos + 1
        ac[pos] = Counter()
        for r in pu.pileups:
            if r.is_refskip:
                continue
            # print(r.alignment.query_name, r.query_position)
            if r.is_del:
                ac[pos][None] += 1
            else:
                ac[pos][r.alignment.query_sequence[r.query_position]] += 1
    return [(gi(chrom, gpos, gpos), ac[gpos] if gpos in ac else Counter()) for gpos in range(start, stop)]


def test_FastPileupIterator():
    with open_file_obj(get_resource('small_example_bam')) as bam:
        # A T/C SNP
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, '1', {22432587})] == \
               [(22432587, Counter({'C': 4}))]
        # 2 positions with  single MM
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, '1', {22433446, 22433447})] == [
            (22433446, Counter({'G': 3, 'T': 1})), (22433447, Counter({'C': 3, 'G': 1}))]
        # A G/T SNP with 3 low-quality bases
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, '1', {22418286}, min_base_quality=10)] \
               == [(22418286, Counter({'T': 12, 'G': 2}))]
        # position with 136 Ts and 1 deletion
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, '1', {22418244})] == [
            (22418244, Counter({'T': 136, None: 1}))]
        # assert that also uncovered positions are reported
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, '1', range(22379012, 22379015))] == [
            (22379012, Counter()), (22379013, Counter()), (22379014, Counter({'C': 1}))]
        # assert equal to slow pysam pileup. This region contains uncovered areas, insertions and deletions: chr1:22,408,208-22,408,300
        assert FastPileupIterator(bam, '1', range(22408208, 22408300)).to_list() == slow_pileup(bam, '1', 22408208,
                                                                                                22408300)
        # test aliasing
        assert [(loc.start, c) for loc, c in FastPileupIterator(bam, 'chr1', {22418244}, fun_alias=toggle_chr)] == [
            (22418244, Counter({'T': 136, None: 1}))]


def test_FastqIterator():
    assert len(FastqIterator(get_resource('small_fastq'))) == 4
    assert [len(x[1]) for x in FastqIterator(get_resource('small_fastq'))] == [34, 26, 24, 37]
    # iterate PE reads and assert names contain 1/2
    for r1, r2 in zip(FastqIterator(get_resource('small_PE_fastq1')), FastqIterator(get_resource('small_PE_fastq2'))):
        n1 = r1.name.split(' ')[1].split(':')[0]
        n2 = r2.name.split(' ')[1].split(':')[0]
        assert n1 == '1' and n2 == '2'


def test_gt2zyg():
    assert gt2zyg('.') == (0, 0)
    assert gt2zyg('1') == (2, 1)
    assert gt2zyg('2') == (2, 1)
    assert gt2zyg('1/.') == (2, 1)
    assert gt2zyg('.|2') == (2, 1)
    assert gt2zyg('1/1/2') == (1, 1)
    assert gt2zyg('0/0/.') == (2, 0)


def test_VcfIterator():
    """TODO: test INDELs"""
    with VcfIterator(get_resource('test_vcf')) as it:
        assert [v.GT for _, v in it.to_list()] == [{'SAMPLE': '1/1'}] * 4
    with VcfIterator(get_resource('test_vcf')) as it:
        assert [v.CS for _, v in it.to_list()] == [{'SAMPLE': 'A'}, {'SAMPLE': 'B'}, {'SAMPLE': 'C'}, {'SAMPLE': 'D'}]
    with VcfIterator(get_resource('test_vcf')) as it:
        assert [v.zyg for _, v in it.to_list()] == [{'SAMPLE': 2}] * 4
    with VcfIterator(get_resource('test_vcf')) as it:
        assert [loc.start for loc, _ in it.to_list()] == [100001, 200001, 300001,
                                                      1000]  # first 3 are deletions, so the genomic pos is +1

    # with sample filtering
    with VcfIterator(get_resource('dmel_multisample_vcf'), samples=['DGRP-208', 'DGRP-325', 'DGRP-721'],
                     filter_nocalls=True) as it:
        dat = it.to_list()
        assert len(dat) == 58  # there are 58 variants called in at least one of the 3 samples
        # NOTE: this sample contains some hom-ref call (0/0) !
        assert {v.data.zyg['DGRP-325'] for v in dat} == {2, 0}

    # test missing chromosome, i.e., a chrom that is in the refdict but not in the tabix index!
    assert VcfIterator(get_resource('test_snps_vcf'), region=gi('3')).to_list() == []


def test_BedIterator():
    # simple test
    assert len(BedIterator(get_resource('test_bed')).to_list()) == 3
    # bed12 test
    assert len(BedIterator(get_resource('test_bed12')).to_list()) == 1


def test_vcf_and_gff_it():
    """TODO: expand"""
    gff_file = get_resource('flybase_gtf')
    vcf_file = get_resource('dmel_multisample_vcf')
    for _ in AnnotationIterator(GFF3Iterator(gff_file, '2L', 574299, 575733),
                                VcfIterator(vcf_file, samples=['DGRP-208', 'DGRP-325', 'DGRP-721'])):
        pass


def test_set_chrom(testdata):
    d, df = testdata
    its = [BedIterator(get_resource('test_bed')),
           BedGraphIterator(get_resource('test_bedgraph')),
           MemoryIterator(d),
           GFF3Iterator(get_resource('gencode_gff')),
           PandasIterator(df, 'Name')]
    for it in its:
        all_int = it.to_list()
        per_chrom = []
        for c in it.chromosomes:
            it.set_region(gi(c))
            per_chrom.append(it.to_list())
        per_chrom = list(itertools.chain(*per_chrom))
        assert [loc for loc, _ in all_int] == [loc for loc, _ in per_chrom]


def test_pybedtools_it(testdata):
    # test whether they return the same locations
    for x, y in zip(BedIterator(get_resource('test_bed')), PybedtoolsIterator(get_resource('test_bed'))):
        assert (x.location == y.location)
    # test filtering, NOTE that the used BED file is not bgzipped and indexed, so its unsafe to use and a
    # warning is issued
    assert [x.location for x in
            PybedtoolsIterator(get_resource("pybedtools::a.bed"), region=gi.from_str("chr1:1-300"))] == \
           from_str("chr1:2-100 (+), chr1:101-200 (+), chr1:151-500 (-)")
    # now we filter before and then just iterate with our iterator
    bt = pybedtools.BedTool(get_resource("pybedtools::a.bed")).intersect([pybedtools.Interval('chr1', 1, 300)], u=True)
    with PybedtoolsIterator(bt) as it:
        assert [x.location for x in it] == from_str(
            "chr1:2-100 (+), chr1:101-200 (+), chr1:151-500 (-)")
        print(it.stats)


def test_pybedtools_it_anno(testdata):
    # combine with annotationiterator
    # Annotate intervals from a bed file with values from a bedgraph file
    bed_file = get_resource('test_bed')  # anno
    bedg_file = get_resource('test_bedgraph')  # scores
    # overlap with bedgraph file, calculate overlap and sum scores
    # NOTE bedgraph file contains interval (1:7-10, 0.3)
    print(AnnotationIterator(PybedtoolsIterator(bed_file), PybedtoolsIterator(bedg_file), labels=['scores']).to_list())
    with AnnotationIterator(PybedtoolsIterator(bed_file), PybedtoolsIterator(bedg_file), labels=['scores']) as it:
        assert ([(i.anno.name, sum([float(x.data.name) * loc.overlap(x.location) for x in i.scores])) for loc, i in
                 it]) == \
               [('int1', 1.408), ('int2', 0.3), ('int3', 0)]
        print(it.stats)


def test_BioframeIterator(testdata):
    bedgraph_file = get_resource('dmel_randomvalues')
    refdict = ReferenceDict.load(bedgraph_file)
    for roi in [None, gi('2L', start=1), gi(), gi('2L', 10000, 20000)]:  # test with different filter regions
        mean_pgl = {}
        for chrom in refdict:
            with BedGraphIterator(bedgraph_file, chromosome=chrom, region=roi) as it:
                mean_pgl[chrom] = np.nanmean([v for loc, v in it])
                if np.isnan(
                        mean_pgl[chrom]):  # if the roi contains no data we get an empty list and np.nanmean returns nan
                    del mean_pgl[chrom]
        mean_bf = BioframeIterator(bedgraph_file, region=roi).df.groupby('chrom')['name'].mean().to_dict()
        assert mean_bf == mean_pgl
