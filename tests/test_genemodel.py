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
        'genome_fa': 'ACTB+SOX2.fa',
        'gene_gff':  'gencode.v39.ACTB+SOX2.gff3.gz'
    }
    t=transcriptome(config)
    assert len(t.genes)==2 and len(t.transcripts)==24
    assert t.genes['ENSG00000181449.4'].transcripts['ENST00000325404.3'].location == loc_obj('chr3', 181711925, 181714436, '+')
