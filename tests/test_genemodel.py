from pygenlib.genemodel import *
import pytest
from pathlib import Path
from itertools import product

@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir
def test_geneid2symbol():
    res=geneid2symbol(['ENSMUSG00000029580', 60])
    assert res['60'].symbol=='ACTB' and res['60'].taxid==9606
    assert res['ENSMUSG00000029580'].symbol == 'Actb' and res['ENSMUSG00000029580'].taxid == 10090

def test_genemodel(base_path):
    """ complex transcriptome test """
    config={
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'gene_gff':  'gencode.v39.ACTB+SOX2.gff3.gz',
        'load_intron_sequences': True
    }


    t=Transcriptome(config)
    assert len(t.genes)==2 and len(t.transcripts)==24
    assert t.genes['ENSG00000181449.4'].transcripts['ENST00000325404.3'].location == gi('chr3', 181711925, 181714436, '+')
    t.load_sequences()

    tx = t.transcripts['ENST00000676189.1']
    assert tx.exons[0].get_seq(mode='rna') == 'ACCACCGCCGAGACCGCGTCCGCCCCGCGAGCACAGAGCCTCGCCTTTGCCGATCCGCCGCCCGTCCACACCCGCCGCCAG' # BLAT
    assert tx.get_seq(mode='rna').startswith('ACCA')
    assert tx.get_seq(mode='rna').endswith('TGAG')
    assert tx.introns[-1].get_seq(mode='rna') == "GTGGGTGTCTTTCCTGCCTGAGCTGACCTGGGCAGGTCGGCTGTGGGGTCCTGTGGTGTGTGGGGAGCTGTCACATCCAGGGTCCTCACTGCCTGTCCCCTTCCCTCCTCAG"
    # length of spliced tx seq = sum of length of exon locations
    for tx in t.transcripts.values():
       assert sum([len(ex.location) for ex in tx.exons]) == len(tx.get_spliced_seq())
    # introns 4+5 are in 3'-UTR
    assert [i.rnk for i in tx.introns if i.location.overlaps(gi.merge([u.location for u in tx.utr3]))] == [4,5]