import pysam

from pygenlib.iterators import *
from pygenlib.utils import TagFilter
import pytest
import os
from pathlib import Path
from itertools import product
from more_itertools import take
import pandas as pd
from dataclasses import dataclass

@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir

@pytest.fixture(autouse=True)
def testdata() -> pd.DataFrame:
    # Some overlap tests
    # .........1........  ....2......
    # |-a-|
    #     |-b-|
    #   |-c-|
    #           |-d-|
    #                         |-e-|
    #                     |-f--------|
    #                         |-g-|
    #                         |--h---|
    d = {
        'a': loc_obj('1', 1, 10),
        'b': loc_obj('1', 11, 20),
        'c': loc_obj('1', 5, 15),
        'd': loc_obj('1', 30, 40),
        'e': loc_obj('2', 21, 30),
        'f': loc_obj('2', 1, 50),
        'g': loc_obj('2', 21, 30),
        'h': loc_obj('2', 21, 50),
    }
    df = pd.DataFrame([(loc.chromosome, loc.start, loc.end, name) for name, loc in d.items()],
                      columns=['Chromosome', 'Start', 'End', 'Name'])  # note: this df is not sorted!
    return d, df

def loc_list(s):
    return [loc_obj.from_str(x) for x in s.split(',')]

def test_FastaIterator(base_path):
    fasta_file='ACTB+SOX2.fa'
    # read seq via pysam
    with pysam.Fastafile(fasta_file) as fh:
        ref={c:fh.fetch(c) for c in fh.references}
    # consume all
    all=''.join([s for _,s in FastaIterator(fasta_file, 'chr3', width=1, step=1).take()])
    assert(all==ref['chr3'])
    # some edge cases where the provided sequence is shorter than the requested window size
    assert FastaIterator(fasta_file, 'chr7', 3, 6, width=5, step=3, padding=False).take() == [(loc_obj('chr7', 3, 7), 'GTGCN')] # 5-mer from region of size 4, wo padding
    assert FastaIterator(fasta_file, 'chr7', 3, 6, width=5, step=3, padding=True).take() == [(loc_obj.from_str('chr7:1-5'), 'NNGTG'), (loc_obj.from_str('chr7:4-8'), 'TGCNN')] # 5-mer from region of size 4, wiwth padding
    # consume in tiling windows
    tiled = ''.join([s for _, s in FastaIterator(fasta_file, 'chr7', None, None, width=3, step=3).take()])
    assert(tiled[:-1]==ref['chr7']) # NOTE cut last char in tiled as it is padded by a single N (as len(ref['chr7']) % 3 = 2)
    # get the first 10 5-mers with and w/o padding
    fivemers = [s for _,s in FastaIterator(fasta_file, 'chr7', None, None, width=5, step=2, padding=False).take()][:10]
    assert fivemers==['TTGTG','GTGCC','GCCAT','CATTA','TTACA','ACACT','ACTCC','TCCAG','CAGCC','GCCTG']
    fivemers= [s for _,s in FastaIterator(fasta_file, 'chr7', None, None, width=5, step=2, padding=True).take()][:10]
    assert fivemers==['NNTTG','TTGTG','GTGCC','GCCAT','CATTA','TTACA','ACACT','ACTCC','TCCAG','CAGCC']
    # get 11-mers with padding
    ctx=[s for _, s in FastaIterator(fasta_file, 'chr7', 1, 10, width=11, step=1, padding=True)]
    assert ctx[:5]==['NNNNNTTGTGC','NNNNTTGTGCC','NNNTTGTGCCA','NNTTGTGCCAT','NTTGTGCCATT']
    assert ''.join([x[5] for x in ctx])==ref['chr7'][:10]


def test_TabixIterator(base_path):
    vcf_file = 'test_snps.vcf.gz'
    bed_file = 'test.bed.gz'
    bedg_file = 'test.bedgraph.gz'  # includes track header
    # read VCF file as TSV
    ti=TabixIterator(vcf_file, region='1:1-10', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert(merge_yields(ti.take())[0] == loc_obj('1', 1, 10))
    ti=TabixIterator(vcf_file, chromosome='1', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert(merge_yields(ti.take())[0] == loc_obj('1', 1, 20))
    ti = TabixIterator(vcf_file, chromosome='2', coord_inc=[0, 0], pos_indices=[0, 1, 1])
    assert len([(l,t) for l, t in ti.take()])==1
    with pytest.raises(AssertionError) as e_info:
        TabixIterator(vcf_file, 'unknown_contig',5,10)
    print(f'Expected assertion: {e_info}')
    # BED file
    ti=TabixIterator(bed_file, '1', 1, 10, coord_inc = [1, 0])
    assert(merge_yields(ti.take())[0]==loc_obj('1',6,15)) # start is 0-based, end is 1-based
    # bedgraph file
    assert sum([float(r[3]) for _, r in TabixIterator(bedg_file, coord_inc=[1, 0]).take()])==0.625


def test_PandasIterator(base_path, testdata):
    d,df=testdata
    it = PandasIterator(df, 'Name')
    assert {k:v for v,k in it}==d

def test_BlockLocationIterator(base_path, testdata):
    with BlockLocationIterator(TabixIterator('test.bed.gz', coord_inc = [1, 0]), strategy=BlockStrategy.OVERLAP) as it:
        locs=[l for l,_ in it]
        assert locs == loc_list('1:6-15,2:10-150')
    d, df = testdata
    assert [l for l, _ in BlockLocationIterator(PandasIterator(df, 'Name'), strategy=BlockStrategy.OVERLAP)] == loc_list('1:1-20,1:30-40 ,2:1-50')
    assert BlockLocationIterator(PandasIterator(df, 'Name'), strategy=BlockStrategy.LEFT).take()[-1][1][1] == ['e', 'g', 'h'] # same start coord
    assert BlockLocationIterator(PandasIterator(df, 'Name'), strategy=BlockStrategy.RIGHT).take()[-2][1][1] == ['e', 'g'] # same end coord
    right_sorted= BlockLocationIterator(PandasIterator(df.sort_values(['Chromosome', 'End']), 'Name', is_sorted=True), strategy=BlockStrategy.RIGHT)
    assert [x[1] for _, x in right_sorted.take()[-2:]] ==  [['e', 'g'], ['f', 'h']]

def test_PyrangesIterator(base_path):
    exons, cpg = pr.data.exons(), pr.data.cpg()
    # get exons with same start but different end coords
    res=[]
    for mloc, (locs, ex) in BlockLocationIterator(PandasIterator(exons.df, 'Name')):
        endpos={ l.end for l in locs}
        if len(endpos)>1:
            res+=[(mloc, (locs, ex))]
    assert len(res)==5

@dataclass
class MockRead:
    tags: dict
    def has_tag(self, tag):
        return tag in self.tags
    def get_tag(self, tag):
        return self.tags.get(tag)

def test_TagFilter():
    assert TagFilter('xx', [1,12,13], False).filter(MockRead({'xx':12}))  # filter if values is found
    assert not TagFilter('xx', [12], False, inverse=True).filter(MockRead({'xx':12})) # inverse filter: filter if values is not found!

def test_ReadIterator(base_path):
    with ReadIterator('rogue_read.bam', 'SIRVomeERCCome') as it:
        for l,r in it:
            pass
        assert it.stats['n_reads', 'SIRVomeERCCome']==1
    stats={x:Counter() for x in ['all', 'def', 'mq20', 'tag']}
    with open_pysam_obj('small_example.bam') as bam:
        for chrom in get_reference_dict(bam):
            with ReadIterator(bam, chrom, flag_filter=0) as it:
                it.take()
                stats['all'].update(it.stats)
            with ReadIterator(bam, chrom) as it:
                it.take()
                stats['def'].update(it.stats)
            with ReadIterator(bam, chrom, min_mapping_quality=20) as it:
                it.take()
                stats['mq20'].update(it.stats)
            with ReadIterator(bam, chrom, tag_filters=[TagFilter('MD', ['100'])]) as it:
                it.take()
                stats['tag'].update(it.stats)
    print( stats['tag'])
    assert stats['all']['n_reads', '1']==31678 # samtools view -c small_example.bam -> 31678
    assert stats['def']['n_reads', '1']==21932 # samtools view -c small_example.bam -F 3844 -> 21932
    assert stats['mq20']['n_reads', '1']==21626 # samtools view -c small_example.bam -F 3844 -q 20 -> 21626
    assert stats['tag']['n_reads', '1'] == 7388  # samtools view  small_example.bam -F 3844 | grep -v "MD:Z:100" | wc -l -> 7388
    # count t/c mismatches
    tc_conv={}
    for l,r,mm in ReadIterator('small_example.bam',report_mismatches=True, min_base_quality=10):
        if len(mm)>0:
            is_rev = not r.is_reverse if r.is_read2 else r.is_reverse
            refc = "A" if is_rev else "T"
            altc = "G" if is_rev else "C"
            mm_tc=[(off, pos1, ref, alt) for off, pos1, ref, alt in mm if ref==refc and alt==altc]
            if len(mm_tc) > 0:
                tc_conv[r.query_name, not r.is_read2]=mm_tc
    # overlapping mate pair: both  contain T/C snp
    assert tc_conv['HWI-ST466_135068617:8:2209:6224:33460', True]== [(71, 22432587, 'T', 'C')]
    assert tc_conv['HWI-ST466_135068617:8:2209:6224:33460', False] == [(29, 22432587, 'T', 'C')]
    #  a read with 2 A/G conversions
    assert tc_conv['HWI-ST466_135068617:8:2316:4251:54002', False]==[(2, 22443997, 'A', 'G'), (5, 22444000, 'A', 'G')]

def test_FastPileupIterator(base_path):
    # A T/C SNP
    assert [(l.start,c) for l,c in FastPileupIterator('small_example.bam', '1', {22432587}).take()]==[(22432587, Counter({'C': 4}))]
    # 2 positions with  single MM
    assert [(l.start,c) for l,c in FastPileupIterator('small_example.bam', '1', {22433446,22433447}).take()]==[(22433446, Counter({'G': 3, 'T': 1})), (22433447, Counter({'C': 3, 'G': 1}))]
    # A G/T SNP with 3 low-quality bases
    assert [(l.start,c) for l,c in FastPileupIterator('small_example.bam', '1', {22418286}, min_base_quality=10).take()]==[(22418286, Counter({'T': 12, 'G': 2}))]
    # position with 136 Ts and 1 deletion
    assert [(l.start,c) for l,c in FastPileupIterator('small_example.bam', '1', {22418244}).take()]==[(22418244, Counter({'T': 136, None: 1}))]