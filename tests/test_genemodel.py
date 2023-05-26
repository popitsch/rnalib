from pygenlib.genemodel import *
import pytest
import tempfile
from pathlib import Path
from itertools import product

from pygenlib.utils import print_dir_tree, cmp_sets


@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir


@pytest.fixture(autouse=True)
def testdata() -> dict:
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'gencode.v39.ACTB+SOX2.gff3.gz',
        'annotation_flavour': 'gencode'
    }
    return config


def test_geneid2symbol():
    res = geneid2symbol(['ENSMUSG00000029580', 60])
    assert res['60'].symbol == 'ACTB' and res['60'].taxid == 9606
    assert res['ENSMUSG00000029580'].symbol == 'Actb' and res['ENSMUSG00000029580'].taxid == 10090

def test_eq(base_path, testdata):
    t1 = Transcriptome(testdata)
    t2 = Transcriptome(testdata)
    assert t1 == t2
    # __eq__ not implemented yet for Features.
    for o1, o2 in zip(t1.genes.values(), t2.genes.values()):
        assert o1 == o2
    for o1, o2 in zip(t1.transcripts.values(), t2.transcripts.values()):
        assert o1 == o2
    for o1, o2 in pairwise(t1.genes.values()):
        assert o1 != o2

def test_eq_PAR_genes(base_path):
    # special cases with duplicated genes in PAR regions
    config = {
        'genome_fa': '/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.fna',
        'annotation_gff': '/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/annotation/gencode.v39.annotation.sorted.gff3.gz',
        'annotation_flavour': 'gencode',
        'transcript_filter': {
            '#included_tids': ['ENST00000483079.6', 'ENST00000496301.6'],
            'included_chrom': ['chrX','chrY']
        },
        'copied_fields': ['gene_type'],
        'drop_empty_genes': False,
        'load_sequences': False
    }
    t=Transcriptome(config)
    t.drop_empty_genes()
    t.check_stats()

def test_genemodel(base_path, testdata):
    """ complex transcriptome test """
    t = Transcriptome(testdata)
    assert len(t.genes) == 2 and len(t.transcripts) == 24
    assert t.genes['ENSG00000181449.4'].transcripts['ENST00000325404.3'].location() == gi('chr3', 181711925, 181714436,
                                                                                          '+')
    t.load_sequences()

    # length of spliced tx seq = sum of length of exon locations
    for tx in t.transcripts.values():
        assert sum([len(ex) for ex in tx.exons]) == len(tx.get_spliced_seq())

    tx = t.transcripts['ENST00000676189.1']
    assert tx.exons[0].get_seq(
        mode='rna') == 'ACCACCGCCGAGACCGCGTCCGCCCCGCGAGCACAGAGCCTCGCCTTTGCCGATCCGCCGCCCGTCCACACCCGCCGCCAG'  # BLAT
    assert tx.get_seq(mode='rna').startswith('ACCA')
    assert tx.get_seq(mode='rna').endswith('TGAG')
    assert tx.introns[-1].get_seq(
        mode='rna') == "GTGGGTGTCTTTCCTGCCTGAGCTGACCTGGGCAGGTCGGCTGTGGGGTCCTGTGGTGTGTGGGGAGCTGTCACATCCAGGGTCCTCACTGCCTGTCCCCTTCCCTCCTCAG"
    # introns 4+5 are in 3'-UTR
    assert [i.rnk for i in tx.introns if i.overlaps(gi.merge(tx.utr3))] == [4, 5]
    # assert that links are correct

    config = {
        'genome_fa': 'dmel_r6.36.fa.gz',
        'annotation_gff': 'flybase.dmel-all-r6.51.sorted.gtf.gz',
        'annotation_flavour': 'flybase',
        'transcript_filter': {
            'included_chrom': ['2L'],
            'included_regions': ['2L:9839-21376']
        }
    }
    t = Transcriptome(config)
    t.drop_empty_genes()
    assert t.transcripts['FBtr0306591'].parent==t.genes['FBgn0002121']
    for g in t.genes.values():
        for tx in g.transcripts.values():
            print( g.name, tx.tid, tx.gene_type, tx.parent, tx.parent.name)
            assert g.name==tx.parent.name


def test_itree(base_path, testdata):
    t = Transcriptome(testdata)
    # 1 gene, exons of 5 tx but only one 3'UTR annotation at this coordinate
    assert {e.name for e in t.query(gi('chr7', 5529026, 5529026), Gene)} == {'ACTB'}
    assert {e.parent.tid for e in t.query(gi('chr7', 5529026, 5529026), Exon)} == \
           {'ENST00000477812.2', 'ENST00000484841.6', 'ENST00000425660.5', 'ENST00000645025.1', 'ENST00000462494.5'}
    # envelop
    assert {e.parent.tid for e in t.query(gi('chr7', 5528947, 5529104), Exon, envelop=True)} == \
           {'ENST00000425660.5', 'ENST00000484841.6'}
    # get 3'utrs
    assert [u.parent.tid for u in t.query(gi('chr7', 5529026, 5529026), Utr3)] == ['ENST00000425660.5']
    # check edges
    assert [len(t.query(gi('chr7', pos, pos), Gene)) for pos in [5526408, 5526409, 5563902, 5563903]] == [0, 1, 1, 0]


def test_genmodel_persistence(base_path, testdata):
    # save and load
    with tempfile.TemporaryDirectory() as tmp:
        pkfile = os.path.join(tmp, 'transcriptome.pk')
        t1 = Transcriptome(testdata)
        t1.save(pkfile)
        t2 = Transcriptome.load(pkfile)
        print_dir_tree(tmp)
        assert t1 == t2


def test_genmodel_gff3(base_path, testdata):
    """ Write to GFF3, load and compare"""
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'gencode.v39.ACTB+SOX2.gff3.gz',
        'annotation_flavour': 'gencode',
        'copied_fields': ['gene_type', 'tag'],
        'transcript_filter': {
            'included_tags': ['Ensembl_canonical']
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        gff3file = os.path.join(tmp, 'transcriptome.gff3')
        t1 = Transcriptome(config)
        t1.to_gff3(gff3file, bgzip=True)
        config2 = config.copy()
        config2['annotation_gff'] = gff3file + '.gz'
        t2 = Transcriptome(config2)
        assert t1 == t2


def test_filter(base_path, testdata):
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'gencode.v39.ACTB+SOX2.gff3.gz',
        'annotation_flavour': 'gencode',
        'transcript_filter': {
            'included_tids': ['ENST00000325404.3', 'ENST00000674681.1']
        }
    }
    t = Transcriptome(config)
    assert t.log == {'parsed_gff_lines': 275, 'filtered_exon': 99, 'filtered_five_prime_UTR': 30, 'filtered_transcript': 22,
                     'filtered_three_prime_UTR': 18}
    assert len(t.transcripts) == 2
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'gencode.v39.ACTB+SOX2.gff3.gz',
        'annotation_flavour': 'gencode',
        'copied_fields': ['gene_type'],
        'transcript_filter': {
            'included_tags': ['Ensembl_canonical']
        }
    }
    t = Transcriptome(config)
    assert t.log == {'parsed_gff_lines': 275, 'filtered_exon': 99, 'filtered_five_prime_UTR': 30, 'filtered_transcript': 22,
                     'filtered_three_prime_UTR': 18}
    assert len(t.transcripts) == 2
    # assert copied fields
    assert all([hasattr(tx, 'gene_type') for tx in t.transcripts.values()])


def test_triples(base_path, testdata):
    """ TODO: more max_dist checks (same chrom) """

    def get_name(x):
        return None if x is None else x.name

    t = Transcriptome(testdata)
    assert [(get_name(x), get_name(y), get_name(z)) for x, y, z in t.gene_triples()] == \
           [(None, 'SOX2', 'ACTB'), ('SOX2', 'ACTB', None)]
    assert [(get_name(x), get_name(y), get_name(z)) for x, y, z in t.gene_triples(max_dist=1000)] == \
           [(None, 'SOX2', None), (None, 'ACTB', None)]


def test_utility_functions(base_path, testdata):
    t = Transcriptome(testdata)
    # test whether returned 3'end intervals are in sum 200bp long or None (if tx too short)
    for tx in t.transcripts.values():
        assert calc_3end(tx) is None or sum([len(x) for x in calc_3end(tx)]) == 200


def test_annotation_flavours(base_path):
    # Ensembl with chrom aliasing
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'ensembl_Homo_sapiens.GRCh38.109.ACTB+SOX2.gtf.gz',
        'annotation_flavour': 'ensembl',
        'annotation_fun_alias': 'toggle_chr',
        '#transcript_filter': { 'included_tids': ['ENST00000676319']}
    }
    t = Transcriptome(config)
    assert t.check_stats()
    assert len(t.genes) == 2
    # UCSC with tx filtering
    config = {
        'genome_fa': 'ACTB+SOX2.fa.gz',
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': 'UCSC.hg38.ncbiRefSeq.ACTB+SOX2.sorted.gtf.gz',
        'annotation_flavour': 'ucsc',
        'transcript_filter': {
            'included_tids': ['NM_001101.5']
        }
    }
    t = Transcriptome(config)
    assert t.check_stats()
    assert len(t.genes) == 1
    # Mirgenedb
    config = {
        'genome_fa': 'dmel_r6.36.fa.gz',
        'annotation_gff': 'mirgenedb.dme.sorted.gff3.gz',
        'annotation_flavour': 'mirgenedb',
        '#transcript_filter': {
            'included_regions': ['X:4368325-4368346']
        }
    }
    t = Transcriptome(config)
    assert t.check_stats()
    assert len(t.genes) == 483
    # flybase
    config = {
        'genome_fa': 'dmel_r6.36.fa.gz',
        'annotation_gff': 'flybase.dmel-all-r6.51.sorted.gtf.gz',
        'annotation_flavour': 'flybase',
        'transcript_filter': {
            'included_chrom': ['2L'],
            'included_regions': ['2L:1-10000']
        }
    }
    t = Transcriptome(config)
    assert t.check_stats()
    assert len(t.genes) == 3515
    t.drop_empty_genes() # drop
    assert t.check_stats()
    assert len(t.genes) == 2 and len(t.transcripts)==12
    cnt = Counter()
    for tx in t.transcripts.values():
        cnt[tx.gene_type] += 1
    assert cnt == {'mRNA': 11, 'pseudogene': 1}
    assert {l.chromosome for l in t.transcripts.values()} == {'2L'}
