from pygenlib.utils import *
import pytest
from pathlib import Path
import json
import pysam
import numpy as np
import os
import tempfile
import shutil

@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir

def test_reverse_complement():
    assert reverse_complement('ACTG') =="CAGT"
    assert reverse_complement('ACUG', TMAP['rna']) == "CAGU"
    assert reverse_complement('ACTG', TMAP['rna']) == "C*GU"
    assert complement('ACTG', TMAP['dna']) == "TGAC"
    assert complement('ACUG', TMAP['rna']) == "UGAC"
    tmap=ParseMap(TMAP['rna'], missing_char='?') # custom unknown-base char
    assert complement('ACTG', tmap) == "UG?C"
    tmap = TMAP['rna']
    assert complement('ACTG', tmap) == "UG*C"


def test_get_config():
    config = json.loads('{ "obj1": { "obj2": { "prop1": 1 }}, "list1": [ "a", "b"], "prop2": 2 }')
    assert get_config(config, ['obj1', 'obj2', 'prop1']), 1
    assert get_config(config, 'obj1/obj2/prop1'), 1
    assert get_config(config, ['list1']), ["a", "b"]
    assert get_config(config, 'prop2'), 2
    assert get_config(config, 'prop3', 15), 15
    assert get_config(config, 'prop3') is None


def test_parse_gff_info(base_path):
    """ shallow test of GFF/GTF infor parsing.
    """
    gene_gff_file = 'gencode.v39.ACTB+SOX2.gff3.gz'  # 275 records
    with pysam.TabixFile(gene_gff_file, mode="r") as f:
        for row in f.fetch(parser=pysam.asTuple()):
            reference, source, ftype, fstart, fend, score, strand, phase, info = row
            pinfo = parse_gff_info(info)
            min_exp_fields=set('ID,gene_id,gene_type,gene_name,level'.split(','))
            shared, _, _ = cmp_sets(set(pinfo.keys()), min_exp_fields)
            assert shared==min_exp_fields

def test_longest_hp_gc_len():
    assert longest_hp_gc_len("AAAAAA"), (6, 0)
    assert longest_hp_gc_len("AAAGCCAA"), (3, 3)
    assert longest_hp_gc_len("GCCGCGCGCGCGCGCAAAGCCAA"), (3, 15)
    assert longest_hp_gc_len("GCCGCGCGCGCGCCCCGCAAAGCCAA"), (4, 18)
    assert longest_hp_gc_len("ACTGNNNACTGC"), (3, 2)


def test_align_guide():
    assert align_sequence('ACTG', 'AAATTTCCCACTGAAATTTCCC'), (1.0, 9, 13)
    _, x, y = align_sequence('ACTG', 'AAATTTCCCACTGAAATTTCCC')
    assert 'AAATTTCCCACTGAAATTTCCC'[x:y], 'ACTG'
    assert align_sequence('ACTG', 'AAAAAAAA'), (0.25, 0, 4)
    s, x, y = align_sequence('ACTG', 'AACATTTCCCAAACTTTCCC')  # find the second longer hit
    assert (s, x, y) == (0.75, 12, 16) and 'AACATTTCCCAAACTTTCCC'[x:y] == 'ACTT'


def test_kmer_search():
    seq = "ACTGATACGATGCATCGACTAGCATCGACTACGATCAGCTACGATCGACTAACGCGAGCAC"
    res = kmer_search(seq, ['TACGA', 'ATCAG', 'AACGC'])
    for k in res:
        for s in res[k]:
            assert seq[s:s + len(k)], k

def test_split_list():
    split_list([1, 2, 3, 4, 5, 6], 3, is_chunksize=True)
    for i in [0, 1, 5, 20, 30]:
        assert len(list(split_list(range(i), 3))) == 3
    for x, y, z in [(0, 1, 0), (1, 1, 1)]:
        # print(list(split_list(range(x), y, is_chunksize=True)))
        assert len(list(split_list(range(x), y, is_chunksize=True))) == z

def test_intersect_lists():
    assert intersect_lists()==[]
    assert intersect_lists([1,2,3,4],[1,4],[3,1], check_order=True)==[1]
    assert intersect_lists([1,2,3,4],[1,4])==[1,4]
    assert intersect_lists([1, 2, 3], [3, 2, 1])==[3,2,1]
    with pytest.raises(AssertionError) as e_info: # assert that assertion error is raised
        intersect_lists([1, 2, 3], [3, 2, 1], check_order=True)
    print(f'Expected assertion: {e_info}')
    assert intersect_lists((1,2,3,5),(1,3,4,5))==[1,3,5]

def test_to_str():
    assert to_str(), 'NA'
    assert to_str(na='*'), '*'
    assert to_str([12, [None, [1, '', []]]]), '12,NA,1,NA,NA'
    assert to_str(12, [None, [1, '', []]]), '12,NA,1,NA,NA'
    assert to_str(range(3), sep=';'), '0;1;2'
    assert to_str(1, 2, [3, 4], 5, sep=''), '12345'
    assert to_str([1, 2, 3][::-1], sep=''), '321'
    assert to_str((1, 2, 3)[::-1], sep=''), '321'

def test_rnd_seq():
    # we expect 50% GC
    gc_perc = np.array([count_gc(s)[1] for s in rnd_seq(100, m=1000)])
    assert 0.45 < np.mean(gc_perc) < 0.55
    # we expect 60% GC
    gc_perc=np.array([count_gc(s)[1] for s in rnd_seq(100, 'GC'*60 + 'AT'*40, 1000)])
    assert 0.55<np.mean(gc_perc)<0.65

def test_bgzip_and_tabix(base_path):
    # create temp dir, gunzip a GFF3 file and bgzip+tabix via pysam.
    # just asserts that file exists.
    with tempfile.TemporaryDirectory() as tmp:
        gene_gff_file = 'gencode.v39.ACTB+SOX2.gff3.gz'  # 275 records
        gunzip(gene_gff_file, tmp+'/test.gff3')
        print('created temporary file', tmp+'/test.gff3')
        bgzip_and_tabix(tmp+'/test.gff3')
        assert os.path.isfile(tmp+'/test.gff3.gz') and os.path.isfile(tmp+'/test.gff3.gz.tbi')
        print_dir_tree(tmp)

def test_count_reads(base_path):
    assert count_lines('Test01_L001_R1_001.top20.fastq'), 80
    assert count_reads('Test01_L001_R1_001.top20.fastq'), 20

def test_write_data():
    assert write_data([1,2,['a','b', None], None], sep=';'), "1;2;a,b,NA;NA"

def test_slugify():
    assert slugify("this/is/an invalid filename!.txt"), "thisisan_invalid_filenametxt"

def test_reference_dict(base_path):
    r1 = ReferenceDict("A", {'chr1': 1, 'chr2':2, 'chrM': 23, 'chrX': 24})
    r2 = ReferenceDict("B", {'chr1': 1, 'chrX': 24})
    r3 = ReferenceDict("C", {'chr1': 1, 'chrX': 24, 'chrM': 23}) # different order
    r4 = ReferenceDict("B", {'chr1': 1, 'chrX': 25}) # different length
    assert ReferenceDict.merge_and_validate() is None
    assert ReferenceDict.merge_and_validate(r1) == r1
    assert ReferenceDict.merge_and_validate(r1, r2) == {'chr1': 1, 'chrX': 24}
    with pytest.raises(AssertionError) as e_info:
        ReferenceDict.merge_and_validate(r1, r3)
    print(f'Expected assertion: {e_info}')
    assert ReferenceDict.merge_and_validate(r1, r2) == {'chr1': 1, 'chrX': 24}
    with pytest.raises(AssertionError) as e_info:
        ReferenceDict.merge_and_validate(r1, r4)
    print(f'Expected assertion: {e_info}')
    ReferenceDict.merge_and_validate(r1,None,r2)
