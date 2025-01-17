"""
Tests for the utils module
"""
import gzip
import json
import os
import tempfile
from collections import defaultdict

import dill
import numpy as np
import pysam
import pytest

import rnalib
import rnalib as rna

assert rnalib.__RNALIB_TESTDATA__ is not None, (
    "Please set rnalib.__RNALIB_TESTDATA__ variable to the testdata " "directory path"
)


def from_str(s):
    return [rna.gi(x) for x in s.split(",")]


def test_rna_complement():
    assert rna.reverse_complement("ACTG") == "CAGT"
    assert rna.reverse_complement("ACUG", rna.TMAP["rna"]) == "CAGU"
    assert rna.reverse_complement("ACTG", rna.TMAP["rna"]) == "C*GU"
    assert rna.complement("ACTG", rna.TMAP["dna"]) == "TGAC"
    assert rna.complement("ACUG", rna.TMAP["rna"]) == "UGAC"
    tmap = rna.ParseMap(rna.TMAP["rna"], missing_char="?")  # custom unknown-base char
    assert rna.complement("ACTG", tmap) == "UG?C"
    tmap = rna.TMAP["rna"]
    assert rna.complement("ACTG", tmap) == "UG*C"


def test_get_config():
    config = json.loads(
        '{ "obj1": { "obj2": { "prop1": 1 }}, "list1": [ "a", "b"], "prop2": 2 }'
    )
    assert rna.get_config(config, ["obj1", "obj2", "prop1"]), 1
    assert rna.get_config(config, "obj1/obj2/prop1"), 1
    assert rna.get_config(config, ["list1"]), ["a", "b"]
    assert rna.get_config(config, "prop2"), 2
    assert rna.get_config(config, "prop3", 15), 15
    assert rna.get_config(config, "prop3") is None


def test_parse_gff_attributes():
    """shallow test of GFF/GTF info field parsing."""
    for fn, dialect in [
        (rna.get_resource("gencode_gff"), rna.GFF_FLAVOURS["gencode", "gff"]),
        (rna.get_resource("ucsc_gtf"), rna.GFF_FLAVOURS["ucsc", "gtf"]),
        (rna.get_resource("ensembl_gff"), rna.GFF_FLAVOURS["ensembl", "gff"]),
        (rna.get_resource("flybase_gtf"), rna.GFF_FLAVOURS["flybase", "gtf"]),
    ]:
        expected_fields = {
            v
            for k, v in dialect.items()
            if k in ["gid", "tid", "tx_gid", "feat_tid"] and v is not None
        }
        parsed_attributes = set()
        with pysam.TabixFile(fn, mode="r") as f:
            for row in f.fetch(parser=pysam.asTuple()):
                reference, source, ftype, fstart, fend, score, strand, phase, info = row
                parsed_attributes |= rna.parse_gff_attributes(
                    info, fmt=rna.guess_file_format(fn)
                ).keys()
        shared, _, _ = rna.cmp_sets(set(parsed_attributes), expected_fields)
        assert shared == expected_fields, f"missing fields in {fn}"


def test_get_reference_dict():
    """Test reference dict implementation and aliasing"""
    assert rna.RefDict.load(
        rna.get_resource("ensembl_gff"), fun_alias=rna.toggle_chr
    ).keys() == {"chr3", "chr7"}
    assert (
            rna.RefDict.load(rna.get_resource("ensembl_gff")).orig.keys()
            == rna.RefDict.load(
        rna.get_resource("ensembl_gff"), fun_alias=rna.toggle_chr
    ).orig.keys()
    )
    assert (
            rna.RefDict.load(
                rna.get_resource("ensembl_gff"), fun_alias=rna.toggle_chr
            ).alias("1")
            == "chr1"
    )
    # compare 2 refdicts, one w/o chr prefix (ensembl) and one with (fasta file)
    assert rna.RefDict.merge_and_validate(
        rna.RefDict.load(rna.get_resource("ensembl_gff"), fun_alias=rna.toggle_chr),
        rna.RefDict.load(rna.get_resource("ACTB+SOX2_genome")),
    ).keys() == {"chr3", "chr7"}
    assert rna.RefDict.merge_and_validate(
        rna.RefDict.load(rna.get_resource("ensembl_gff")),
        rna.RefDict.load(
            rna.get_resource("ACTB+SOX2_genome"), fun_alias=rna.toggle_chr
        ),
    ).keys() == {"3", "7"}


def test_longest_hp_gc_len():
    assert rna.longest_hp_gc_len("AAAAAA"), (6, 0)
    assert rna.longest_hp_gc_len("AAAGCCAA"), (3, 3)
    assert rna.longest_hp_gc_len("GCCGCGCGCGCGCGCAAAGCCAA"), (3, 15)
    assert rna.longest_hp_gc_len("GCCGCGCGCGCGCCCCGCAAAGCCAA"), (4, 18)
    assert rna.longest_hp_gc_len("ACTGNNNACTGC"), (3, 2)


def test_kmer_search():
    seq = "ACTGATACGATGCATCGACTAGCATCGACTACGATCAGCTACGATCGACTAACGCGAGCAC"
    res = rna.kmer_search(seq, ["TACGA", "ATCAG", "AACGC"])
    for k in res:
        for s in res[k]:
            assert seq[s: s + len(k)], k


def test_split_list():
    rna.split_list([1, 2, 3, 4, 5, 6], 3, is_chunksize=True)
    for i in [0, 1, 5, 20, 30]:
        assert len(list(rna.split_list(range(i), 3))) == 3
    for x, y, z in [(0, 1, 0), (1, 1, 1)]:
        # print(list(split_list(range(x), y, is_chunksize=True)))
        assert len(list(rna.split_list(range(x), y, is_chunksize=True))) == z


def test_intersect_lists():
    assert rna.intersect_lists() == []
    assert rna.intersect_lists([1, 2, 3, 4], [1, 4], [3, 1], check_order=True) == [1]
    assert rna.intersect_lists([1, 2, 3, 4], [1, 4]) == [1, 4]
    assert rna.intersect_lists([1, 2, 3], [3, 2, 1]) == [3, 2, 1]
    with pytest.raises(
            AssertionError
    ) as e_info:  # assert that assertion error is raised
        rna.intersect_lists([1, 2, 3], [3, 2, 1], check_order=True)
    print(f"Expected assertion: {e_info}")
    assert rna.intersect_lists((1, 2, 3, 5), (1, 3, 4, 5)) == [1, 3, 5]


def test_to_str():
    assert rna.to_str() == "NA"
    assert rna.to_str(na="*") == "*"
    assert rna.to_str([12, [None, [1, "", []]]]) == "12,NA,1,NA,NA"
    assert rna.to_str(12, [None, [1, "", []]]) == "12,NA,1,NA,NA"
    assert rna.to_str(range(3), sep=";") == "0;1;2"
    assert rna.to_str(1, 2, [3, 4], 5, sep="") == "12345"
    assert rna.to_str([1, 2, 3][::-1], sep="") == "321"
    assert rna.to_str((1, 2, 3)[::-1], sep="") == "321"


def test_to_list():
    assert rna.to_list(None) == []
    assert rna.to_list(1) == [1]
    assert rna.to_list([1, 2]) == [1, 2]
    assert rna.to_list((1, 2)) == [1, 2]
    assert rna.to_list({1: 2}) == [{1: 2}]
    assert rna.to_list("1,2,3", sep=",") == ["1", "2", "3"]
    assert rna.to_list("1,2,3", sep=";") == ["1,2,3"]


def test_to_set():
    assert rna.to_set(None) == set()
    assert rna.to_set(1) == {1}
    assert rna.to_set([1, 2]) == {1, 2}
    assert rna.to_set((1, 2)) == {1, 2}
    assert rna.to_set("1,2,3", sep=",") == {"1", "2", "3"}
    assert rna.to_set("1,2,3", sep=";") == {"1,2,3"}


def test_rnd_seq():
    # we expect 50% GC
    gc_perc = np.array([rna.count_gc(s)[1] for s in rna.rnd_seq(100, m=1000)])
    assert 0.45 < np.mean(gc_perc) < 0.55
    # we expect 60% GC
    gc_perc = np.array(
        [rna.count_gc(s)[1] for s in rna.rnd_seq(100, "GC" * 60 + "AT" * 40, 1000)]
    )
    assert 0.55 < np.mean(gc_perc) < 0.65


def test_random_sample():
    assert rna.random_sample(12) == 12  # constant
    uni100 = rna.random_sample("uniform(1,2,100)")  # uniform sampling
    assert len(uni100) == 100 and 1 <= min(uni100) <= max(uni100) <= 2


def test_bgzip_and_tabix():
    # create temp dir, gunzip a GFF3 file and bgzip+tabix via pysam.
    # just asserts that file exists.
    with tempfile.TemporaryDirectory() as tmp:
        rna.gunzip(rna.get_resource("gencode_gff"), tmp + "/test.gff3")
        print("created temporary file", tmp + "/test.gff3")
        rna.bgzip_and_tabix(tmp + "/test.gff3")
        assert os.path.isfile(tmp + "/test.gff3.gz") and os.path.isfile(
            tmp + "/test.gff3.gz.tbi"
        )
        rna.print_dir_tree(tmp)
    # test sorting
    with tempfile.TemporaryDirectory() as tmp:
        rna.gunzip(rna.get_resource("gencode_gff"), tmp + "/test.gff3")
        # read all lines and randomly shuffle
        with open(tmp + "/test.gff3", "r") as f:
            shuffeled_lines = f.readlines()
            np.random.shuffle(shuffeled_lines)
        with open(tmp + "/test.gff3", "w") as f:
            f.writelines(shuffeled_lines)
        rna.bgzip_and_tabix(
            tmp + "/test.gff3", sort=True, seq_col=0, start_col=3, end_col=4
        )
        assert os.path.isfile(tmp + "/test.gff3.gz") and os.path.isfile(
            tmp + "/test.gff3.gz.tbi"
        )
        rna.print_dir_tree(tmp)
        # compare with original file using two blockLocationIterators
        orig_blocks = {
            loc
            for loc, _ in rna.GroupedLocationIterator(
                rna.GFF3Iterator(rna.get_resource("gencode_gff")), strategy="start"
            )
        }
        sort_blocks = {
            loc
            for loc, _ in rna.GroupedLocationIterator(
                rna.GFF3Iterator(tmp + "/test.gff3.gz"), strategy="start"
            )
        }
        assert orig_blocks == sort_blocks


def test_count_reads():
    assert rna.count_lines(rna.get_resource("small_PE_fastq1")), 80
    assert rna.count_reads(rna.get_resource("small_PE_fastq1")), 20


def test_write_data():
    assert rna.write_data([1, 2, ["a", "b", None], None], sep=";"), "1;2;a,b,NA;NA"


def test_slugify():
    assert rna.slugify(
        "this/is/an invalid filename!.txt"
    ), "thisisan_invalid_filenametxt"


def test_refdict():
    r1 = rna.RefDict({"chr1": 1, "chr2": 2, "chrM": 23, "chrX": 24}, "A", None)
    r2 = rna.RefDict({"chr1": 1, "chrX": 24}, "B", None)
    r3 = rna.RefDict({"chr1": 1, "chrX": 24, "chrM": 23}, "C", None)  # different order
    r4 = rna.RefDict({"chr1": 1, "chrX": 25}, "D", None)  # different length
    assert rna.RefDict.merge_and_validate() is None
    assert rna.RefDict.merge_and_validate(r1) == r1
    assert rna.RefDict.merge_and_validate(r1, r2) == {"chr1": 1, "chrX": 24}
    with pytest.raises(AssertionError) as e_info:
        rna.RefDict.merge_and_validate(r1, r3, check_order=True)
    print(f"Expected assertion: {e_info}")
    assert rna.RefDict.merge_and_validate(r1, r2) == {"chr1": 1, "chrX": 24}
    with pytest.raises(AssertionError) as e_info:
        rna.RefDict.merge_and_validate(r1, r4)
    print(f"Expected assertion: {e_info}")
    rna.RefDict.merge_and_validate(r1, None, r2)
    # test iter_blocks()
    r5 = rna.RefDict(
        {"chr1": 10, "chr2": 20, "chrM": 23, "chrX": 12}, "test_refdict", None
    )
    assert list(r5.tile(10)) == from_str(
        "chr1:1-10, chr2:1-10,  chr2:11-20, chrM:1-10,  chrM:11-20, chrM:21-23, chrX:1-10,  chrX:11-12"
    )


def test_annodict():
    # test key type checking
    d = rna.FixedKeyTypeDefaultdict(defaultdict, allowed_key_type=int)
    d[1] = 1
    with pytest.raises(TypeError) as e_info:
        d["x"] = 2
    print("expected exception:", e_info.value)
    d.allowed_key_type = str
    d["x"] = 2
    # test pickling
    d = rna.FixedKeyTypeDefaultdict(defaultdict, allowed_key_type=rna.Feature)
    reg = rna.Feature(
        chromosome="1",
        start=10,
        end=20,
        strand=None,
        transcriptome=None,
        feature_type="reg",
    )
    d[reg]["x"] = 1
    d2 = dill.loads(dill.dumps(d))
    assert d == d2
    assert d.allowed_key_type == d2.allowed_key_type
    assert d[reg]["x"] == d2[reg]["x"]


def test_calc_3end():
    config = {
        "genome_fa": rna.get_resource(
            "ACTB+SOX2_genome"
        ),  # rna.get_resource('ACTB+SOX2_genome'),
        "genome_offsets": {"chr3": 181711825, "chr7": 5526309},
        "annotation_gff": rna.get_resource(
            "gencode_gff"
        ),  # rna.get_resource('gencode_gff'),,
        "annotation_flavour": "gencode",
        "feature_filter": {"gene": {"included": {"gene_type": ["protein_coding"]}}},
    }
    t = rna.Transcriptome(**config)
    # test whether returned 3'end intervals are in sum 200bp long or None (if tx too short)
    for tx in t.transcripts:
        assert (
                rna.calc_3end(tx) is None or sum([len(x) for x in rna.calc_3end(tx)]) == 200
        )


def test_geneid2symbol():  # needs internet connection
    res = rna.geneid2symbol(["ENSMUSG00000029580", 60])
    assert res["60"].symbol == "ACTB" and res["60"].taxid == 9606
    assert (
            res["ENSMUSG00000029580"].symbol == "Actb"
            and res["ENSMUSG00000029580"].taxid == 10090
    )


def test_build_amplicon_resources():
    with tempfile.TemporaryDirectory(dir=rna.__RNALIB_TESTDATA__) as tmpdir:
        # create a small BED file with two defined amplicons that map to the genomic positions in the used FASTA file
        with open(f"{tmpdir}/test.bed", "wt") as out:
            rna.MemoryIterator(
                {
                    "amp1.1": rna.gi("chr3:1-100"),
                    "amp1.2": rna.gi("chr3:50-150"),
                    "amp2": rna.gi("chr7:10-50"),
                }
            ).to_bed(out, no_header=True)
        test_bed = rna.bgzip_and_tabix(f"{tmpdir}/test.bed")
        with gzip.open(f"{tmpdir}/test.bed.gz", "rt") as infile:
            print("".join(infile.readlines()))
        assert rna.tools.build_amplicon_resources(
            "test_transcriptome",
            bed_file=test_bed,
            fasta_file=rna.get_resource("ACTB+SOX2_genome"),
            padding=50,
            amp_extension=10,
            out_dir=tmpdir,
        ) == {
                   "mean_amplicon_length": 340.0,
                   "parsed_intervals": 3,
                   "written_amplicons": 2,
               }
        rna.print_dir_tree(tmpdir)


def print_bam_file(bam_file):
    with pysam.AlignmentFile(bam_file, "rb") as f:
        print(f.header, end="")
        for r in f:
            print(r.to_string())


def test_bam_writer():
    # load transcriptome
    t = rna.Transcriptome(
        genome_fa="/Users/niko/Desktop/data/ref/genomes/GRCh38/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.fna",
        annotation_gff=rna.get_resource("gencode_gff"),
        annotation_flavour="gencode",
        load_sequence_data=True,
        feature_filter=rna.TranscriptFilter(
            config={"gene": {"included": {"gene_type": ["protein_coding"]}}}
        ),
    )
    with tempfile.TemporaryDirectory(dir=rna.__RNALIB_TESTDATA__) as tmpdir:
        # tmpdir="/Users/niko/Desktop/data/projects/Ameres/slamlib/data/delme"
        out_bam = f"{tmpdir}/test.bam"
        cr = 0.05
        with rna.BamWriter(t.genome_fa, out_bam) as wout:
            read_len = 200
            for tx in [t.transcripts[0], t.transcripts[1]]:
                splice_seq, idx, idx0 = rna.get_tx_indices(tx)
                for start in range(0, len(splice_seq) - read_len, 50):
                    read_seq, blocks = rna.get_aligned_blocks(
                        tx, start, start + read_len, splice_seq, idx, idx0
                    )
                    if blocks is None:
                        continue
                    # add T/C mismatches
                    read_tpos = np.array(
                        [i for (i, x) in enumerate(read_seq) if x == 'T']
                    )
                    read_tc_pos = np.array(
                        [
                            i
                            for i, p in zip(
                            read_tpos,
                            rna.random_sample(f'uniform(0,1,{len(read_tpos)})'),
                        )
                            if p <= cr
                        ]
                    )
                    for i in read_tc_pos:
                        read_seq = read_seq[:i] + 'C' + read_seq[i + 1:]
                    query_sequence = (
                        rna.reverse_complement(read_seq)
                        if tx.strand == "-"
                        else read_seq
                    )
                    wout.write(aligned_blocks=blocks, query_sequence=query_sequence)
        # print_bam_file(out_bam)


def test_get_covered_regions():
    assert list(rna.get_covered_regions(rna.get_resource("small_example_bam"))) == [
        rna.gi("1:22376052-22474991 (+)")
    ]


def test_SeqErrProfile():
    se = rna.MismatchProfile(
        {
            ('T', 'C', '+'): 10,
            ('T', 'T', '+'): 990,
            ('T', 'A', '+'): 0,
            ('T', 'G', '+'): 0,
        }
    )
    assert se.get_prob('T', 'C') == 10 / 1000
    assert se.get_prob('T', 'A') == 0
    se = rna.MismatchProfile(
        {
            ('T', 'C', '+'): 10,
            ('T', 'T', '+'): 975,
            ('T', 'A', '+'): 10,
            ('T', 'G', '+'): 5,
        }
    )
    assert se.get_prob('T', 'C') == 10 / 1000
    assert se.get_prob('T', 'A') == 10 / 1000
    assert se.get_prob('T', None) == 25 / 1000
    assert se.get_prob('x', None) == 0
    se = rna.MismatchProfile.get_flat_profile(0.1)
    assert se.get_mismatch_prob("+") == 0.1
    assert se.get_prob('T', 'C', "-") == 0.1 / 3
    se = rna.MismatchProfile.from_bam(
        rna.get_resource("small_example_bam"), features=None, min_cov=1, max_mm_frac=1
    )
    print(
        [
            se.get_prob(ref, alt, strand)
            for (ref, alt, strand) in [
            ('N', 'A', "+"),
            ('N', 'T', "+"),
            ('N', 'G', "+"),
            ('N', 'C', "+"),
        ]
        ]
    )
    print(
        [
            se.get_prob(ref, alt, strand)
            for (ref, alt, strand) in [
            ('N', 'A', "-"),
            ('N', 'T', "-"),
            ('N', 'G', "-"),
            ('N', 'C', "-"),
        ]
        ]
    )
    assert np.allclose(
        a=[
            se.get_prob(ref, alt, strand, revcomp=False)
            for (ref, alt, strand) in [
                ('N', 'A', "+"),
                ('N', 'T', "+"),
                ('N', 'G', "+"),
                ('N', 'C', "+"),
            ]
        ],
        b=[0.28, 0.31, 0.21, 0.20],
        atol=0.03,
    )
    assert np.allclose(
        a=[
            se.get_prob(ref, alt, strand, revcomp=False)
            for (ref, alt, strand) in [
                ('N', 'A', "-"),
                ('N', 'T', "-"),
                ('N', 'G', "-"),
                ('N', 'C', "-"),
            ]
        ],
        b=[0.28, 0.31, 0.21, 0.20],
        atol=0.03,
    )


def test_is_paired():
    assert rna.is_paired(rna.get_resource("small_Actb_PE_bam"))
    assert not rna.is_paired(rna.get_resource("small_ACTB+SOX2_clean_MD_bam"))
    assert rna.is_paired(rna.get_resource("small_example_bam"))


def test_extract_aligned_reads_from_fastq():
    with tempfile.TemporaryDirectory() as tmp:
        assert (
                rna.extract_aligned_reads_from_fastq(
                    bam_file=rna.get_resource("small_remapped_bam"),
                    fastq1_file=rna.get_resource("small_remapped_fastq1"),
                    fastq2_file=rna.get_resource("small_remapped_fastq2"),
                    out_file_prefix=tmp + "/test",
                )[0]
                == 66
        )
        rna.print_dir_tree(tmp)
        rna.print_small_file(tmp + "/test.r1.fq.gz", False, 8)
        rna.print_small_file(tmp + "/test.r2.fq.gz", False, 8)


def test_compact_bedgraph_file():
    with tempfile.TemporaryDirectory() as tmp:
        bgf = rna.compact_bedgraph_file(
            rna.get_resource("human_umap_k24"), out_file=tmp + "/test.bedgraph"
        )
        all_loc = [loc for loc, _ in rna.BedGraphIterator(bgf)]
        assert rna.gi("chr7:5529166-5529166") in all_loc  # assert some interval
        assert rna.gi("chr7:5529202-5529205") in all_loc  # assert merged interval
        assert rna.gi("chr7:5530527-5530549") in all_loc  # assert merged interval
        assert rna.gi("chr7:5530551-5531863") in all_loc  # assert last interval
        rna.print_small_file(bgf, True, 400)


def test_get_rel_coord():
    config = {
        "genome_fa": rna.get_resource("ACTB+SOX2_genome"),
        "genome_offsets": {"chr3": 181711825, "chr7": 5526309},
        "annotation_gff": rna.get_resource(
            "gencode_gff"
        ),  # rna.get_resource('gencode_gff'),,
        "annotation_flavour": "gencode",
        "disable_progressbar": True,  # no progressbars
        "feature_filter": {"gene": {"included": {"gene_type": ["protein_coding"]}}},
        "load_sequence_data": True
    }
    t = rna.Transcriptome(**config)
    tx = t['ENST00000674681.1']
    tx_len = sum([len(e) for e in tx.exon])
    # this should be the 1st nucleotide in this transcript
    assert rna.get_rel_coord(tx, "exon", rna.gi("chr7:5530601-5530601")) == \
           rna.gi(chromosome="ENST00000674681.1", start=1, end=1)
    # this should be the last nucleotide in this transcript
    assert rna.get_rel_coord(tx, "exon", rna.gi("chr7:5526409-5526409")) == \
           rna.gi(chromosome="ENST00000674681.1", start=tx_len, end=tx_len)

    # strand specificity
    assert rna.get_rel_coord(tx, "exon", rna.gi("chr7:5530601-5530601"), True) is None
    assert rna.get_rel_coord(tx, "exon", rna.gi("chr7:5530601-5530601 (-)"), True) == \
           rna.gi(chromosome="ENST00000674681.1", start=1, end=1)

    # codons that should translate to SKR
    aa_seq = tx.get_protein_sequence(rna.gi("chr7:5529339-5529346"))
    assert str(aa_seq) == "SKR"

    # fly genome tests
    config = {
        "genome_fa": rna.get_resource('dmel_genome'),  # genome FASTA
        "annotation_gff": rna.get_resource('flybase_gtf'),  # gene annotation GTF/GFF file
        "annotation_flavour": 'flybase',  # flavour of the annotation file.
        "load_sequence_data": True,  # load sequences from configured genome FASTA file
        "disable_progressbar": True,  # no progressbars
        "feature_filter": {'location': {'included': {'chromosomes': ['2L']}}}
        # a simple filter that will include only annotations from 2L
    }
    t = rna.Transcriptome(**config)
    # minus tx test
    tx = t['FBtr0306591']
    # overlapping tx end
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:21,372-21,411"))
    assert tx.spliced_sequence[rel.start-1:rel.end] == "GCACA"
    # start in intron
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:21,061-21,100"))
    assert tx.spliced_sequence[rel.start-1:rel.end] == "CACCGCCTTAACCGTACAAACGGAAATACATAAAG"
    # end in intron
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:9,825-9,864"))
    assert tx.spliced_sequence[rel.start-1:rel.end] == "TTAATAAAACATGTATTTTATTTAAC"
    # overlapping tx start
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:15,709-15,713"))
    assert tx.spliced_sequence[rel.start-1:rel.end] == "ACG"

    # plus strand tests
    tx = t['FBtr0306539']
    # overlapping tx start
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:66,449-66,488"))
    assert tx.spliced_sequence[rel.start - 1:rel.end] == "GACAACT"
    # over SJ
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:67,502-67,578"))
    assert tx.spliced_sequence[rel.start - 1:rel.end] == "ACAAGACATGTTGGGC"
    # both endpoints in introns - should return exon2
    rel = tx.get_rel_coordinates('exon', rna.gi("2L:67,378-67,562"))
    assert tx.spliced_sequence[rel.start - 1:rel.end] == tx.exon[2].sequence

def test_get_protein_sequence():
    config = {
        "genome_fa": rna.get_resource('dmel_genome'),  # genome FASTA
        "annotation_gff": rna.get_resource('flybase_gtf'),  # gene annotation GTF/GFF file
        "annotation_flavour": 'flybase',  # flavour of the annotation file.
        "load_sequence_data": True,  # load sequences from configured genome FASTA file
        "disable_progressbar": True,  # no progressbars
        "feature_filter": {'location': {'included': {'chromosomes': ['2L']}}}
        # a simple filter that will include only annotations from 2L
    }
    t = rna.Transcriptome(**config)
    # minus tx test
    tx = t['FBtr0306591']
    # CDS access
    assert len(tx.get_rel_coordinates('CDS', rna.gi("2L:17,132-17,219"))) == 5
    assert [str(tx.get_protein_sequence(rna.gi("2L", s, 17219))) for s in range(17130, 17137)] == \
           ['MLK', 'ML', 'ML', 'ML', 'M', 'M', 'M']
