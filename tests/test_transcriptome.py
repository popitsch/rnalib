"""
Tests for the genemodel module
"""
import io
import tempfile
from collections import Counter
from itertools import pairwise

import IPython.terminal.shortcuts.filters
import pytest
import os
import copy
import biotite.sequence as seq

import rnalib
from rnalib import gi, GI, BedGraphIterator, ReadIterator, read_alias_file, norm_gn, Transcriptome, Feature, \
    TranscriptomeIterator, print_dir_tree, TranscriptFilter, AbstractFeatureFilter
from rnalib.testdata import get_resource

assert rnalib.__RNALIB_TESTDATA__ is not None, ("Please set rnalib.__RNALIB_TESTDATA__ variable to the testdata "
                                                "directory path")


@pytest.fixture(autouse=True)
def default_testdata() -> dict:
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome', rnalib.__RNALIB_TESTDATA__),  # get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('gencode_gff', rnalib.__RNALIB_TESTDATA__),  # get_resource('gencode_gff'),,
        'annotation_flavour': 'gencode',
        'feature_filter': {'gene': {'included': {'gene_type': ['protein_coding']}}}
    }
    return config


def test_eq(default_testdata):
    t1 = Transcriptome(**default_testdata)
    t2 = Transcriptome(**default_testdata)
    assert t1 != t2  # different hash
    # __eq__ based on coordinates only
    for o1, o2 in zip(t1.genes, t2.genes):
        assert o1 == o2
        assert o1.get_location() == o2.get_location()
    for o1, o2 in zip(t1.transcripts, t2.transcripts):
        assert o1.get_location() == o2.get_location()
    for o1, o2 in pairwise(t1.genes):
        assert o1.get_location() != o2.get_location()

    f1 = Feature('chr1', 1, 10, '+', feature_id=1, feature_type='a', parent=1)
    f2 = Feature('chr2', 1, 10, '+', feature_id=1, feature_type='a', parent=1)
    assert f1 != f2
    # compare with GI
    assert f1.overlap(gi('chr1', 1, 10)) and not f1.overlap(gi('chr1', 11, 20))
    assert f1 < gi('chr1', 11, 20) and f1 > gi('chr1', 0, 5)


def test_transcriptome(default_testdata):
    """ complex transcriptome test """
    t = Transcriptome(**default_testdata)
    assert len(t.genes) == 2 and len(t.transcripts) == 24
    assert t.transcript['ENST00000325404.3'].location == gi('chr3', 181711925, 181714436, '+')
    t.load_sequences()

    # get SOX2 sequence
    assert t.gene['SOX2'].sequence[:10] == 'GATGGTTGTC'
    assert t.gene['SOX2'].sequence[-10:] == 'GACACTGAAA'

    # translate
    assert str(seq.NucleotideSequence(t.gene['SOX2'].transcript[0].translated_sequence).translate(complete=True)) == \
           'MYNMMETELKPPGPQQTSGGGGGNSTAAAAGGNQKNSPDRVKRPMNAFMVWSRGQRRKMAQENPKMHNSEISKRLGAEWKLLSETEKRPFIDEAKRLRALHMKEHPDYKYRPRRKTKTLMKKDKYTLPGGLLAPGGNSMASGVGVGAGLGAGVNQRMDSYAHMNGWSNGSYSMMQDQLGYPQHPGLNAHGAAQMQPMHRYDVSALQYNSMTSSQTYMNGSPTYSMSYSQQGTPGMALGSMGSVVKSEASSSPPVVTSSSHSRAPCQAGDLRDMISMYLPGAEVPEPAAPSRLHMSQHYQSGPVPGTAINGTLPLSHM*'

    # length of spliced tx seq = sum of length of exon locations
    for tx in t.transcripts:
        assert sum([len(ex) for ex in tx.exon]) == len(t.get_sequence(tx, mode='spliced'))

    # checkstart/end of translated sequence
    assert t.transcript['ENST00000325404.3'].translated_sequence[:10] == 'ATGTACAACA'  # starts with ATG
    assert t.transcript['ENST00000325404.3'].translated_sequence[-10:] == 'ACACATGTGA'  # ends with TGA (stop)

    # n_introns = n_exons -1
    for tx in t.transcripts:
        assert len(tx.exon) == len(tx.intron) + 1

    tx = t.transcript['ENST00000676189.1']
    assert t.get_sequence(tx.exon[0], mode='rna') == \
           'ACCACCGCCGAGACCGCGTCCGCCCCGCGAGCACAGAGCCTCGCCTTTGCCGATCCGCCGCCCGTCCACACCCGCCGCCAG'  # confirmed by BLAT
    assert t.get_sequence(tx, mode='rna').startswith('ACCA')
    assert t.get_sequence(tx, mode='rna').endswith('TGAG')
    assert t.get_sequence(tx.intron[-1], mode='rna') == \
           "GTGGGTGTCTTTCCTGCCTGAGCTGACCTGGGCAGGTCGGCTGTGGGGTCCTGTGGTGTGTGGGGAGCTGTCACATCCAGGGTCCTCACTGCCTGTCCCCTTCCCTCCTCAG"
    # introns 4+5 are in 3'-UTR
    assert [i.get_rnk() for i in tx.intron if i.overlaps(GI.merge(tx.three_prime_UTR))] == [4, 5]
    # assert that links are correct
    config = {
        'genome_fa': get_resource('dmel_genome'),
        'annotation_gff': get_resource('flybase_gtf'),
        'annotation_flavour': 'flybase',
        'feature_filter': {'location': {'included': {'region': ['2L:1-21376']}}},
        'copied_fields': ['gene_type'],
        'load_sequence_data': True
    }
    t = Transcriptome(**config)
    assert t.transcript['FBtr0306591'].parent == t.gene['FBgn0002121']
    assert t.transcript['FBtr0306591'].gff_feature_type == 'mRNA'
    assert t.gene['FBgn0031208'].sequence[:10] == 'CTACTCGCAT' and t.gene['FBgn0031208'].sequence[-10:] == 'TTCCCCAAGT'
    for tx in t.transcripts[:10]:
        # check for premature stop codons which indicate wrong in-silico splicing/translation
        # note that this is not a safe test for any annotated tx as there are special cases, see e.g., FBtr0329895
        assert len(seq.NucleotideSequence(tx.translated_sequence).translate(complete=True)) == len(
            tx.translated_sequence) // 3


def test_to_dataframe(default_testdata):
    t = Transcriptome(**default_testdata)
    TranscriptomeIterator(t).to_dataframe()


def test_itree(default_testdata):
    t = Transcriptome(**default_testdata)
    # 1 gene, exons of 5 tx but only one 3'UTR annotation at this coordinate
    assert {e.gene_name for e in t.query(gi('chr7', 5529026, 5529026), 'gene')} == {'ACTB'}
    assert {e.parent.feature_id for e in t.query(gi('chr7', 5529026, 5529026), 'exon')} == \
           {'ENST00000477812.2', 'ENST00000484841.6', 'ENST00000425660.5', 'ENST00000645025.1', 'ENST00000462494.5'}
    # envelop
    assert {e.parent.feature_id for e in t.query(gi('chr7', 5528947, 5529104), 'exon', envelop=True)} == \
           {'ENST00000425660.5', 'ENST00000484841.6'}
    # get 3'utrs
    assert [u.parent.feature_id for u in t.query(gi('chr7', 5529026, 5529026), 'three_prime_UTR')] == [
        'ENST00000425660.5']
    # check edges
    assert [len(t.query(gi('chr7', pos, pos), 'gene')) for pos in [5526408, 5526409, 5563902, 5563903]] == [0, 1, 1, 0]


def test_clear_annotations(default_testdata):
    t = Transcriptome(**default_testdata)
    # there should be no annotation keys
    assert {len(info) for feature, info in TranscriptomeIterator(t, feature_types=['gene'])} == {0}
    # load sequwnces, now each gene should have 1 annotation key 'dna_seq'
    t.load_sequences()
    assert {len(info) for feature, info in TranscriptomeIterator(t, feature_types=['gene'])} == {1}

    # now, let's annotate each gene with its length
    def anno_len(item):
        loc, (anno, overlapping) = item
        anno['len'] = len(loc)

    t.annotate(anno_its=TranscriptomeIterator(t, feature_types=['gene']), fun_anno=anno_len, feature_types=['gene'])
    assert {len(info) for feature, info in TranscriptomeIterator(t, feature_types=['gene'])} == {2}
    # now we clear all annotations but keep the dna_seq anno (default behaviour)
    t.clear_annotations()
    assert {len(info) for feature, info in TranscriptomeIterator(t, feature_types=['gene'])} == {1}
    # now lets clear all annotations
    t.clear_annotations(retain_keys=None)
    assert {len(info) for feature, info in TranscriptomeIterator(t, feature_types=['gene'])} == {0}


# @pytest.mark.skip(reason="Currently broken")
def test_genmodel_persistence(default_testdata):
    # by_ref is currently broken, see https://oegedijk.github.io/blog/pickle/dill/python/2020/11/10/serializing-dill-references.html#Workaround:-move-definitions-to-__main__:
    # save and load
    with tempfile.TemporaryDirectory() as tmp:
        pkfile = os.path.join(tmp, 'transcriptome.pk')
        t1 = Transcriptome(**default_testdata)
        t1.save(pkfile)
        t2 = Transcriptome.load(pkfile)
        print_dir_tree(tmp)
        assert len(t1) == len(t2)


def test_genmodel_anno_persistence(default_testdata):
    # save and load
    with tempfile.TemporaryDirectory() as tmp:
        pkfile = os.path.join(tmp, 'transcriptome_anno.pk')
        t = Transcriptome(**default_testdata)
        for f in t.genes[0].features():
            t.anno[f]['test'] = 1
        # some annotations are set
        t.save_annotations(pkfile)
        # change annos for all of them
        for f in t.anno.values():
            f['test'] = 'testvalue'
        for f in t.anno:
            assert t.anno[f]['test'] == 'testvalue'
        # load and update
        t.load_annotations(pkfile, update=True)
        for f in t.anno:
            if f in t.genes[0].features():
                assert t.anno[f]['test'] == 1
            else:
                assert t.anno[f]['test'] == 'testvalue'
        # load and replace
        t.load_annotations(pkfile)
        for f in t.anno:
            if f in t.genes[0].features():
                assert t.anno[f]['test'] == 1
            else:
                assert t.anno[f] == {}

        oldid = id(t.genes[0].__class__)
        # can we load for new model? (here the ids of all feature classes differ!)
        t = Transcriptome(**default_testdata)
        t.load_annotations(pkfile)
        for f in t.genes[0].features():
            assert t.anno[f]['test'] == 1
        newid = id(t.genes[0].__class__)
        assert oldid != newid  # different 'feature' classes!


def test_to_gff3(default_testdata):
    """ Write to GFF3, load and compare"""
    with tempfile.TemporaryDirectory() as tmp:
        gff3file = os.path.join(tmp, 'transcriptome.gff3')
        config = default_testdata
        config['copied_fields'] = ['tag',
                                   'gene_type']  # NOTE we must copy these fields or remove the filter for config2!
        config['calc_introns'] = True
        t1 = Transcriptome(**config)
        t1.to_gff3(gff3file, bgzip=True)
        config2 = config.copy()
        config2['annotation_gff'] = gff3file + '.gz'
        config2['calc_introns'] = False
        t2 = Transcriptome(**config2)
        assert len(t1) == len(t2)
        assert Counter([f.feature_type for f in t1.anno]) == Counter([f.feature_type for f in t2.anno])
    # test with custom annotations
    with tempfile.TemporaryDirectory() as tmp:
        gff3file = os.path.join(tmp, 'transcriptome.gff3')
        config = default_testdata
        t1 = Transcriptome(**config)
        reg = t1.add(gi('chr3:5000-6000'), feature_id='my_reg_region',
                    feature_type='regulatory_region')  # create a 'regulatory_region' feature and add to the transcriptome

        t1.anno[reg]['test'] = (4, 5, 6)  # add a 'test' annotation
        t1.to_gff3(gff3file, bgzip=True, feature_types=('regulatory_region'))
        #rnalib.print_small_file(gff3file + '.gz')
        config2 = config.copy()
        config2['annotation_gff'] = gff3file + '.gz'
        t2 = Transcriptome(**config2)
        reg1,_ = zip(*t1.iterator(feature_types='regulatory_region'))
        reg2, _ = zip(*t1.iterator(feature_types='regulatory_region'))
        assert reg1 == reg2



def test_to_bed(default_testdata):
    t = Transcriptome(**default_testdata)
    with io.StringIO() as out:
        t.to_bed(out)
        print(out.getvalue())


def test_filter(default_testdata):
    # simple filter tests
    tf = TranscriptFilter(
        {
            'gene': {
                'included': {'gene_id': ['ENSG...', None], 'tag': ['Ensembl_canonical', None]}
            },
            'transcript': {
                'excluded': {'transcript_id': ['x']}
            },
            'location': {
                'included': {'chromosomes': ['2L']},
                'excluded': {'regions': ['2L:1-21376']}
            }
        })
    # test location filter
    assert tf.filter(gi('2L', 1, 1000), {'feature_type': 'gene'}) == (True, 'excluded_region')
    assert tf.filter(gi('2R', 1, 1000), {'feature_type': 'gene'}) == (True, 'included_chromosome')
    assert tf.filter(gi('2L', 50000, 100000), {'feature_type': 'gene'}) == (False, 'passed')
    assert tf.filter(gi('2L', 50000, 100000), {'feature_type': 'gene', 'gene_id': 'ENSG...'}) == (False, 'passed')
    assert tf.filter(gi('2L', 50000, 100000), {'feature_type': 'gene', 'gene_id': 'ENSG...', 'tag': 'A,B'}) == (
        True, 'missing_tag_value')
    assert tf.filter(gi('2L', 50000, 100000),
                     {'feature_type': 'gene', 'gene_id': 'ENSG...', 'tag': 'Ensembl_canonical,A,B'}) == (
               False, 'passed')
    assert tf.get_chromosomes() == {'2L'}
    # test filter builder
    tf = TranscriptFilter().include_chromosomes(['2L'])
    assert tf.filter(gi('2R', 1, 1000), {'feature_type': 'gene'}) == (True, 'included_chromosome')
    # test with transcriptome
    config = copy.deepcopy(default_testdata)
    config['feature_filter'] = TranscriptFilter({'gene': {'included': {'tags': [
        'protein_coding']}}})
    # config = {
    #     'genome_fa': '_delme_testdata/static_test_files/ACTB+SOX2.fa.gz',  # get_resource('ACTB+SOX2_genome'),
    #     'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
    #     'annotation_gff': '_delme_testdata/gff/gencode_44.ACTB+SOX2.gff3.gz',  # get_resource('gencode_gff'),,
    #     'annotation_flavour': 'gencode'
    # }
    # filter for non-existent tag, all tx should be filtered as well
    t = Transcriptome(**config)
    assert t.log == {'parsed_gff_lines': 345,
                     'filtered_transcript_parent_gene_filtered': 89,
                     'filtered_gene_missing_tags': 5} and len(t.transcripts) == 0
    # filter for list of transcript ids.
    config['feature_filter'] = TranscriptFilter().include_transcript_ids(['ENST00000325404.3', 'ENST00000674681.1'])
    t = Transcriptome(**config)
    assert t.log == {'parsed_gff_lines': 345,
                     'filtered_transcript_missing_transcript_id_value': 87,
                     'dropped_empty_genes': 3}
    assert len(t.transcripts) == 2 and len(t.genes) == 2
    assert {tx.feature_id for tx in t.transcripts} == {'ENST00000325404.3', 'ENST00000674681.1'}
    # filter for gene_type
    config['feature_filter'] = TranscriptFilter().include_gene_types(['protein_coding'])
    t = Transcriptome(**config)
    assert t.log == {'parsed_gff_lines': 345,
                     'filtered_transcript_parent_gene_filtered': 65,
                     'filtered_gene_missing_gene_type_value': 3}
    assert len(t.transcripts) == 24 and len(t.genes) == 2
    # assert copied fields
    assert all([hasattr(tx, 'gene_type') for tx in t.transcripts])
    assert set([tx.gene_type for tx in t.transcripts]) == {'protein_coding'}

    # test custom filter class
    class MyFilter(AbstractFeatureFilter):
        """A simple custom example filter that rejects all non protein coding genes (but keeps entries
        without gene_type annotation) and drops genes outside the first 6 Mb of each chromosome.
        """

        def filter(self, loc, info):  # -> bool, str:
            if info.get('feature_type', '') == 'gene':
                if info.get('gene_type', 'protein_coding') != 'protein_coding':
                    return True, 'not_protein_coding'
                if loc.end > 6000000:  # consider only first 6 Mb
                    return True, 'out_of_bounds'
            return False, 'passed'

    config['feature_filter'] = MyFilter()
    t = Transcriptome(**config)
    assert all([f.end <= 6000000 for f, dat in t.iterator()])


def test_triples(default_testdata):
    """ TODO: more max_dist checks (same chrom) """

    def get_name(x):
        return None if x is None else x.gene_name

    t = Transcriptome(**default_testdata)
    assert [(get_name(x), get_name(y), get_name(z)) for x, y, z in t.gene_triples()] == \
           [(None, 'SOX2', 'ACTB'), ('SOX2', 'ACTB', None)]
    assert [(get_name(x), get_name(y), get_name(z)) for x, y, z in t.gene_triples(max_dist=1000)] == \
           [(None, 'SOX2', None), (None, 'ACTB', None)]


def test_gff_flavours():
    # Ensembl with chrom aliasing
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),  # get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('ensembl_gff'),
        'annotation_flavour': 'ensembl',
        'annotation_fun_alias': 'toggle_chr'
    }
    t = Transcriptome(**config)
    assert len(t.genes) == 5
    assert 'gene:ENSG00000181449' in t.gene

    # UCSC with tx filtering
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('ucsc_gtf'),
        'annotation_flavour': 'ucsc',
        'feature_filter': TranscriptFilter({'transcript': {'included': {'transcript_id': ['NM_001101.5']}}})
    }
    t = Transcriptome(**config)
    assert Counter([f.feature_type for f in t.anno]) == {'exon': 6, 'CDS': 5, 'intron': 5, 'five_prime_UTR': 2,
                                                         'three_prime_UTR': 1, 'gene': 1, 'transcript': 1}
    assert len(t.genes) == 1

    # Mirgenedb
    config = {
        'genome_fa': get_resource('dmel_genome'),
        'annotation_gff': get_resource('mirgendb_dme_gff'),
        'annotation_flavour': 'mirgenedb',
    }
    t = Transcriptome(**config)  # 322 miRNA entries, 161 pre_miRNA entries
    assert Counter([f.feature_type for f in t.anno]) == {'gene': 483, 'transcript': 483}

    # flybase
    config = {
        'genome_fa': get_resource('dmel_genome'),
        'annotation_gff': get_resource('flybase_gtf'),
        'annotation_flavour': 'flybase',
        'feature_filter': TranscriptFilter({
            'location': {'included': {'regions': ['2L:1-10000']}}
        })
    }
    t = Transcriptome(**config)
    print(t.log)
    assert len(t.genes) == 2 and len(t.transcripts) == 12
    assert (Counter([f.feature_type for f in t.anno]) ==
            {'exon': 107,
             'intron': 95,
             'CDS': 84,
             'five_prime_UTR': 32,
             'transcript': 12,
             'three_prime_UTR': 11,
             'gene': 2})
    assert Counter([tx.gff_feature_type for tx in t.transcripts]) == {'mRNA': 11, 'pseudogene': 1}
    assert {loc.chromosome for loc in t.transcripts} == {'2L'}
    # Generic
    config = {
        'genome_fa': get_resource('dmel_genome'),
        'annotation_gff': get_resource('generic_gff3'),
        'annotation_flavour': 'generic',
        'copied_fields': ['Alias', 'ID']
    }
    t = Transcriptome(**config)
    assert 'feature_type' in t.genes[0].__dict__, "Did not copy feature_type field"
    # NOTE there should be *no* intron as the exons are directly adjacent to each other
    assert Counter([f.feature_type for f in t.anno]) == {'exon': 2, 'gene': 1, 'transcript': 1}  # , 'intron': 1}

    # Chess 3 GFF/GTF
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('chess_gff'),
        'annotation_flavour': 'chess',
        'feature_filter': {
            'location': {'included': {'chromosomes': ['chr3']}}
        },
        'copied_fields': ['gene_type', 'source'],
    }
    t = Transcriptome(**config)
    assert len(t.genes) == 2 and len(t.transcripts) == 8
    assert t.gene['SOX2'].gene_type == 'protein_coding'
    assert Counter(tx.source for tx in t.transcripts) == {'RefSeq': 6, 'GENCODE': 1, 'MANE': 1}
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('chess_gtf'),
        'annotation_flavour': 'chess',
        'feature_filter': {
            'location': {'included': {'chromosomes': ['chr3']}}
        },
        'copied_fields': ['gene_type', 'source'],
    }
    t = Transcriptome(**config)
    assert len(t.genes) == 2 and len(t.transcripts) == 8
    assert t.gene['SOX2'].gene_type == 'protein_coding'
    assert Counter(tx.source for tx in t.transcripts) == {'RefSeq': 6, 'GENCODE': 1, 'MANE': 1}


def test_iterator():
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('gencode_gff'),
        'annotation_flavour': 'gencode'
    }
    t = Transcriptome(**config)
    print(t.merged_refdict)
    print(t.iterator('chr3', feature_types=['gene']).to_list())
    assert len(t.iterator('chr3', feature_types=['gene']).to_list()) == 2  # 2 annotated genes on chr3


def test_annotate():
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('gencode_gff'),
        'annotation_flavour': 'gencode',
        'feature_filter': {'transcript': {'included': {'transcript_id': ['ENST00000473257.3']}}},
        'load_sequence_data': True
    }
    t = Transcriptome(**config)
    assert Counter([f.location.feature_type for f in TranscriptomeIterator(t, feature_types=['exon', 'intron'])]) == \
           {'exon': 5, 'intron': 4}

    def calc_mean_score(label='score'):
        """Factory creating annotation functions"""

        def calc_mean(item):
            loc, (anno, scores) = item
            anno[label] = sum([score * loc.overlap(sloc) for sloc, score in scores]) / len(loc)

        return calc_mean

    # annotate exons and introns with mappability scores
    # this is just an example, it would be more efficient to annotate genes with
    # mappability score arrays and calculate for each exon from there (as for sequences)
    t.annotate(anno_its=BedGraphIterator(get_resource('human_umap_k24')),
               fun_anno=calc_mean_score('mappability'),
               feature_types=['exon', 'intron'])
    # for feature, anno in TranscriptomeIterator(t, feature_types=['exon', 'intron']):
    #     print(feature, anno['mappability'], anno)
    ex = t.transcript['ENST00000473257.3'].exon[0]  # check this exon
    assert t.anno[ex]['mappability'] == \
           sum([0.958 * 2 + 0.917 * 23 + 1 * (len(ex) - 25)]) / len(ex)  # checked in IGV


def test_annotate_read_counts():
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('gencode_gff'),
        'annotation_flavour': 'gencode',
        'load_sequence_data': False
    }
    t = Transcriptome(**config)

    # annotate genes with read counts
    def count_reads(item):
        """Read counter"""
        loc, (anno, reads) = item
        if 'rc' not in anno:
            anno['rc'] = Counter()
        anno['rc']['reads'] += len(reads)
        anno['rc']['reads_ss'] += len([r for l1, r in reads if l1.strand == loc.strand])
        anno['rc']['reads_ss_tc'] += len(
            [r for l1, r in reads if l1.strand == loc.strand and r.has_tag('xc') and r.get_tag('xc') > 0])

    # coungt the reads
    t.annotate(anno_its=ReadIterator(get_resource('small_ACTB+SOX2_bam')),
               fun_anno=count_reads)
    # no reads in SOX2
    assert t.gene['SOX2'].rc['reads'] == 0
    # manually checked in IGV
    # 13 reads (7 on minus strand and 3 of those have a t/c conversion) in 1st exon ACTB/ENST00000674681.1
    assert t.transcript['ENST00000674681.1'].exon[0].rc == {'reads': 13, 'reads_ss': 7, 'reads_ss_tc': 3}


def test_eq_PAR_genes():  # needs access to /Volumes
    if not os.path.isfile(
            '/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.fna'):
        pytest.skip("No access to /Volumes/groups/ameres/Niko/, test skipped")
    # special cases with duplicated genes in PAR regions
    # here we test that this does not break GFF parsing and that par_regions are accessible after transcriptome
    # building.
    config = {
        'genome_fa': '/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.fna',
        'annotation_gff': '/Volumes/groups/ameres/Niko/ref/genomes/GRCh38/annotation/gencode.v39.annotation.sorted.gff3.gz',
        'annotation_flavour': 'gencode',
        'transcript_filter': {
            '#included_tids': ['ENST00000483079.6', 'ENST00000496301.6'],
            'included_chrom': ['chrX', 'chrY']
        },
        'copied_fields': ['gene_type'],
        'load_sequence_data': False
    }
    t = Transcriptome(**config)
    par_genes = [g for g in t.genes if hasattr(g, 'par_regions')]
    for chroms in [(g.chromosome, g.par_regions[0].chromosome) for g in par_genes]:
        assert chroms == ('chrX', 'chrY')
    # check for false introns between two PAR locations (several have len==0 as start>end coordinate)
    false_introns = [i for i, _ in t.iterator(feature_types=['intron']) if len(i) == 0]
    assert len(false_introns) == 0


# def test_full_gff3:
#     # Example from https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
#     config = {
#         'genome_fa': 'ACTB+SOX2.fa.gz',
#         'annotation_gff': 'gff_examples/gff3_example1.sorted.gff3.gz',
#         'annotation_flavour': 'ensembl'
#     }
#     t = Transcriptome(**config)

def test_read_alias_file():
    aliases, current_symbols = read_alias_file(get_resource('hgnc_gene_aliases'))
    assert [norm_gn(g, current_symbols, aliases) for g in ['A2MP', 'FLJ23569', 'p170']] == \
           ['A2MP1', 'A1BG-AS1', 'A2ML1']
