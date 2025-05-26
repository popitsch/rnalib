"""
    Provides access to test files and can be used to initially build the testdata folder.

    testdata creation:
        Rnalib tests use various test data files that can be created by running this python script.
        The contained test_resources dict describes the various test resources and their origin.
        Briefly, this script does the following for each configured resource:

        * Download source file from a public URL or copy from the static_test_files directory
        * Ensure that the files are sorted by genomic coordinates and are compressed and indexed with bgzip and tabix.
        * Slice a genomic subregions from the files if configured
        * Copy the resultfiles and corresponding indices to the testdata directory

    testdata access:
        Once you have created the testdata folder, you can get the filenames of test resources via the
        `get_resource(<resource_id>)` method. If `<resource_id>` starts with 'pybedtools::<id>' then this method
        will return the filename of the respective pybedtools test file.
        To list the ids of all available test resources, use the list_resources() method.

    Examples
    --------
    >>> get_resource("gencode_gff")
    >>> get_resource("pybedtools::hg19.gff")
    >>> list_resources()
    >>> ['gencode_gff','gencode_gtf', ... ,'pybedtools::y.bam']

    Note that for testdata creation, the  following external tools need to be installed:

    * samtools
    * bedtools
    * htslib (bgzip, tabix)

"""
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pybedtools
import pysam
from tqdm.auto import tqdm

import rnalib as rna

"""
    Predefined test resources.
    Each resource is defined by a dictionary with the following keys: 
    - uri: the source URI, either an http(s) URL for remote data or a file URL for (static) test resources
    - regions: a list of genomic regions to slice from the source file. If not set, the whole file is used.
        make sure that this list is sorted by genomic coordinates.
    - filename: the target filename in the testdata directory
    - recreate: if set to True, the resource will be recreated even if the target file already exists. Default: False  
"""
test_resources = {
    # -------------- GTF/GFF -------------------------------
    "gencode_gff": {
        "uri": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gff3.gz",
        "regions": ["chr3:181711825-181714536", "chr7:5526309-5564002"],  # sox2 + actb
        "filename": "gff/gencode_44.ACTB+SOX2.gff3.gz",
    },
    "gencode_gtf": {
        "uri": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
        "regions": ["chr3:181711825-181714536", "chr7:5526309-5564002"],  # sox2 + actb
        "filename": "gff/gencode_44.ACTB+SOX2.gtf.gz",
    },
    "flybase_gtf": {
        "uri": "ftp://ftp.flybase.net/genomes/Drosophila_melanogaster/dmel_r6.36_FB2020_05/gtf/dmel-all-r6.36.gtf.gz",
        "regions": ["2L:1-100000"],
        "filename": "gff/flybase_dmel-2L-r6.36.gtf.gz",
    },
    "wormbase_gff": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/wormbase_cele_WS293.head.gff3.gz",
        "filename": "gff/wormbase_cele_WS293.head.gff3.gz",
        "description": """
            wormbase gff was preprocessed with:
            with open("c_elegans.PRJEB28388.WS293.annotations.fixed.gff3", 'wt') as out:
                with open("c_elegans.PRJEB28388.WS293.annotations.gff3") as file:
                    while line := file.readline():
                        tok = line.rstrip().split('\t')
                        if len(tok) == 8:
                            tok+=['.']
                        if not tok[0].startswith("#"):
                            assert len(tok) == 9, f"Wrong nuber of tokens: {tok}"
                        print('\t'.join(tok), file=out)
            and then sorted with bedtools sort -faidx c_elegans.PRJEB28388.WS293.genomic.fa.fai -i ... , bgzipped and tabixed
        """,
    },
    "ensembl_gff": {
        "uri": "https://ftp.ensembl.org/pub/release-110/gff3/homo_sapiens/Homo_sapiens.GRCh38.110.chr.gff3.gz",
        "regions": ["3:181711825-181714536", "7:5526309-5564002"],
        "filename": "gff/ensembl_Homo_sapiens.GRCh38.110.ACTB+SOX2.gff3.gz",
        "recreate": False,
    },
    "ucsc_gtf": {
        "uri": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz",
        "regions": ["chr3:181711825-181714536", "chr7:5526309-5564002"],  # sox2 + actb
        "filename": "gff/UCSC.hg38.ncbiRefSeq.ACTB+SOX2.sorted.gtf.gz",
        "recreate": False,
    },
    "chess_gff": {
        "uri": "https://github.com/chess-genome/chess/releases/download/v.3.0.1/chess3.0.1.gff.gz",
        "regions": ["chr3:181711825-181714536", "chr7:5526309-5564002"],
        "filename": "gff/chess3.GRCh38.ACTB+SOX2.gff3.gz",
        "recreate": False,
    },
    "chess_gtf": {
        "uri": "https://github.com/chess-genome/chess/releases/download/v.3.0.1/chess3.0.1.gtf.gz",
        "regions": ["chr3:181711825-181714536", "chr7:5526309-5564002"],
        "filename": "gff/chess3.GRCh38.ACTB+SOX2.gtf.gz",
        "recreate": False,
    },
    "mirgenedb_dme_gff": {
        "uri": "https://mirgenedb.org/gff/dme?sort=pos&all=1",
        "filename": "gff/mirgenedb.dme.sorted.gff3.gz",
    },
    "mirbase_hsa_gff": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/mirbase_hsa.sorted.gff3.gz",
        "filename": "gff/mirbase_hsa.sorted.gff3.gz",
    },
    "generic_gff3": {
        # small GFF file for testing the generic GFF parser
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/generic.gff3.gz",
        "filename": "gff/generic.gff3.gz",
    },
    "pybedtools_gff": {
        "uri": f"file:///{pybedtools.filenames.example_filename('hg19.gff')}",
        "format": "gff",
        "filename": "gff/pybedtools_gff.gff3.gz",
    },
    # -------------- FASTQ -------------------------------
    "small_fastq": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test.fq.gz",
        "filename": "fastq/test.fq.gz",
    },
    "small_PE_fastq1": {
        # example FASTQ files (paired-end), mate1
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/Test01_L001_R1_001.top20.fastq",
        "regions": 20,
        "filename": "fastq/Test01_L001_R1_001.top20.fastq",
    },
    "small_PE_fastq2": {
        # example FASTQ files (paired-end), mate2
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/Test01_L001_R2_001.top20.fastq",
        "regions": 20,
        "filename": "fastq/Test01_L001_R2_001.top20.fastq",
    },
    "small_remapped_fastq1": {
        # example FASTQ file with remapped reads, mate1
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small_remapped.r1.fq.gz",
        "filename": "fastq/small_remapped.r1.fq.gz",
    },
    "small_remapped_fastq2": {
        # example FASTQ file with remapped reads, mate2
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small_remapped.r1.fq.gz",
        "filename": "fastq/small_remapped.r2.fq.gz",
    },
    # -------------- BAM -------------------------------
    "rogue_read_bam": {
        # example BAM file with a rogue read
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/rogue_read.bam",
        "filename": "bam/rogue_read.bam",
    },
    "ncbi_test_bam": {
        "uri": "https://ftp.ncbi.nlm.nih.gov/toolbox/gbench/tutorial/Tutorial6/BAM_Test_Files/scenario1_with_index/mapt.NA12156.altex.bam",
        "regions": ["NT_167251.1:302188-302337"],
        "filename": "bam/mapt.NA12156.altex.small.bam",
        "recreate": False,
    },
    "deepvariant_test_bam": {
        "uri": "https://github.com/nf-core/test-datasets/raw/deepvariant/testdata/NA12878_S1.chr20.10_10p1mb.bam",
        "filename": "bam/NA12878_S1.chr20.10_10p1mb.bam",
    },
    "small_example_bam": {
        # small GRCh37(!) mapped example BAM file
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small_example.bam",
        "filename": "bam/small_example.bam",
    },
    "small_ACTB+SOX2_bam": {
        # Small example BAM file containing SLAM-seq data (i.e., T/C conversions) for ACTB+SOX2. Included tags: xc, YC
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small.ACTB+SOX2.bam",
        "filename": "bam/small.ACTB+SOX2.bam",
    },
    "small_ACTB+SOX2_clean_MD_bam": {
        # as above, but including MD tags and no xc/YC tags
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small.ACTB+SOX2.clean.MD.bam",
        "filename": "bam/small.ACTB+SOX2.clean.MD.bam",
    },
    "small_Actb_PE_bam": {
        # small example BAM file containing paired-end reads for mouse Actb
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/GSM3185042.Actb.bam",
        "filename": "bam/GSM3185042.Actb.bam",
    },
    "small_remapped_bam": {
        # small example BAM file with remapped reads
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/small_remapped.bam",
        "filename": "bam/small_remapped.bam",
    },
    # -------------- VCF -------------------------------
    "dmel_multisample_vcf": {
        "#uri": "https://www.ebi.ac.uk/eva/webservices/vcf-dumper/v1/segments/2L%3A574291-575734/variants?species=dmelanogaster_6&studies=SNPASSAY_4_SUBMISSION_2010_11_14",
        "uri": "https://www.ebi.ac.uk/eva/webservices/vcf-dumper/v1/segments/2L%3A1-30000/variants?species=dmelanogaster_6&studies=SNPASSAY_4_SUBMISSION_2010_11_14",
        "format": "vcf",
        "filename": "vcf/dmelanogaster_6_exported_20230523.vcf.gz",
    },
    "test_vcf": {
        # small VCFv4.2 example file with a single sample
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test.vcf.gz",
        "filename": "vcf/test.vcf.gz",
    },
    "test_snps_vcf": {
        # small VCF file containing some SNP calls
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test_snps.vcf.gz",
        "filename": "vcf/test_snps.vcf.gz",
    },
    "HG003_GIAB_benchmark_vcf": {
        "uri": "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG003_NA24149_father/latest/GRCh38/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz",
        "regions": [
            "chr1:31647596-31647635",  # MNV
            "chr1:31653729-31653768",  # DEL
            "chr1:31654457-31654496",  # INS
            "chr1:31670549-31670588",  # SNP, 2 ALTs
        ],
        "filename": "vcf/HG003_GIAB_benchmark.vcf.gz",
    },
    # -------------- FAST5 -------------------------------
    "nanoseq_fast5_raw": {
        "uri": "https://github.com/nf-core/test-datasets/raw/nanoseq/fast5/barcoded/003c04de-f704-491e-8d0c-33ffa269423d.fast5",
        "filename": "fast5/003c04de-f704-491e-8d0c-33ffa269423d.fast5",
    },
    "nanoseq_fast5_basecalled": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/FAT61995_a1291c8f_5.fast5",
        "filename": "fast5/FAT61995_a1291c8f_5.fast5",
    },
    # -------------- BED -------------------------------
    "test_bed": {
        # small example bed file
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test.bed.gz",
        "filename": "bed/test.bed.gz",
    },
    "test_bed12": {
        # small example bed12 file
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test_bed12.bed.gz",
        "filename": "bed/test_bed12.bed.gz",
    },
    "deepvariant_test_bed": {
        "uri": "https://github.com/nf-core/test-datasets/raw/deepvariant/testdata/test_nist.b37_chr20_100kbp_at_10mb.bed",
        "filename": "bed/test_nist.b37_chr20_100kbp_at_10mb.bed",
    },
    "gencode_bed": {
        # a bedops-converted gencode_gff file: gff2bed -d < gencode_44.ACTB+SOX2.gff3 | cut -f1-6 | bgzip > gencode_44.ACTB+SOX2.bed.gz
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/gencode_44.ACTB+SOX2.bed.gz",
        "filename": "bed/gencode_44.ACTB+SOX2.bed.gz",
    },
    "example_bedgraph": {
        # small example bedgraph file
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/example.bedgraph.gz",
        "filename": "bed/example.bedgraph.gz",
    },
    # -------------- BEDGRAPH --------------------------
    "test_bedgraph": {
        # small example bedgraph file with track header
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/test.bedgraph.gz",
        "filename": "bed/test.bedgraph.gz",
        "tabix_options": "-S 1",  # skip header
    },
    "human_umap_k24": {
        #  actb ex1+2 (on chr7) umap data from https://bismap.hoffmanlab.org/raw/hg38/k24.umap.bedgraph.gz
        "uri": "https://bismap.hoffmanlab.org/raw/hg38/k24.umap.bedgraph.gz",
        "#uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/GRCh38.chr7.k24.umap.bedgraph.gz",
        "format": "bed",
        "regions": ["chr7:5529160-5531863"],  # actb ex1+2
        "filename": "bed/GRCh38.k24.umap.ACTB_ex1+2.bedgraph.gz",
        "tabix_options": "-S 1",  # skip header
        "include": True,
    },
    "pybedtools_snps": {
        "uri": f"file:///{pybedtools.filenames.example_filename('snps.bed.gz')}",
        "format": "bed",
        "filename": "bed/pybedtools_snps.bed.gz",
    },
    "dmel_randomvalues": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/dmel_randomvalues.bedgraph.gz",
        "filename": "bed/dmel_randomvalues.bedgraph.gz",
    },
    # -------------- FASTA -------------------------------
    "ACTB+SOX2_genome": {
        # GRCh38 derived mFASTA with genomic sequences +/- 100bp of ACTB and SOX2
        # samtools faidx <GRCh38.fa> chr3:181711825-181714536 > ACTB+SOX2.fa
        # samtools faidx <GRCh38.fa> chr7:5526309-5564002    >> ACTB+SOX2.fa
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/ACTB+SOX2.fa.gz",
        "filename": "fasta/ACTB+SOX2.fa.gz",
    },
    "dmel_genome": {
        "uri": "https://ftp.flybase.net/genomes/Drosophila_melanogaster/dmel_r6.36_FB2020_05/fasta/dmel-all-chromosome-r6.36.fasta.gz",
        "filename": "fasta/dmel_r6.36.fa.gz",
    },
    # -------------- DIV -------------------------------
    "hgnc_gene_aliases": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/hgnc_complete_set.head.txt.gz",
        "filename": "div/hgnc_complete_set.head.txt.gz",
    },
    "slamseq_example_featureCounts_all_reads": {
        # a featureCounts output file with multiple samples
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_featureCounts_all_reads.txt",
        "filename": "div/slamseq_example_featureCounts_all_reads.txt",
    },
    "slamseq_example_featureCounts_tc_reads": {
        # a featureCounts output file with multiple samples
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_featureCounts_tc_reads.txt",
        "filename": "div/slamseq_example_featureCounts_tc_reads.txt",
    },
    "slamseq_example_gff": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_mm10.gff3.gz",
        "filename": "gff/slamseq_example_mm10.gff3.gz",
    },
    "slamseq_example_pe_no4sU_bam": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/GSM3185046.pe_example.bam",
        "filename": "bam/GSM3185046.pe_example.bam",
    },
    "slamseq_example_pe_no4sU_vcf": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/GSM3185046.pe_example.vcf.gz",
        "filename": "bam/GSM3185046.pe_example.vcf.gz",
    },
    "slamseq_example_pe_4sU_bam": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/GSM3185043.pe_example.bam",
        "filename": "bam/GSM3185043.pe_example.bam",
    },
    # -------------- BigWig/BigBed -------------------------------
    "test_bigwig": {
        "uri": f"https://github.com/deeptools/pyBigWig/raw/master/pyBigWigTest/test.bw",
        "filename": "bigwig/test.bw",
    },
    "test_bigbed": {
        "uri": f"https://github.com/deeptools/pyBigWig/raw/master/pyBigWigTest/test.bigBed",
        "filename": "bigwig/test.bigBed",
    },
}

large_test_resources = {
    # -------------- Full gencode39 annotation -------------------------------
    "full_gencode_gff": {
        "uri": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_39/gencode.v39.annotation.gff3.gz",
        "filename": "bigfiles/gencode_39.gff3.gz",
    },
    # -------------- Full chess annotation -------------------------------
    "full_chess_gtf": {
        "uri": "https://github.com/chess-genome/chess/releases/download/v.3.0.1/chess3.0.1.gtf.gz",
        "filename": "bigfiles/chess3.0.1.gtf.gz",
    },
    # -------------- Gene name aliases -------------------------------
    "gene_aliases": {
        "uri": "https://ftp.ebi.ac.uk/pub/databases/genenames/out_of_date_hgnc/tsv/hgnc_complete_set.txt",
        "filename": "bigfiles/hgnc_complete_set.txt",
    },
    # -------------- GRCh38 chr20 -------------------------------
    "grch38_chr20": {
        "uri": "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr20.fa.gz",
        "filename": "bigfiles/grch38_chr20.fa.gz",
    },
    # -------------- GRCh38 mappability scores -------------------------------
    "grch38_umap": {
        "uri": "https://bismap.hoffmanlab.org/raw/hg38/k24.umap.bed.gz",
        "filename": "bigfiles/GRCh38.k24.umap.bed.gz",
        "tabix_options": "-p bed -S 1",  # this is a bedgraph file. Skip header.
    },
    # -------------- SLAM-seq example data -------------------------------
    "slamseq_example_tp0": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_0.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_0.cr0.05.STAR.bam",
    },
    "slamseq_example_tp5": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_5.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_5.cr0.05.STAR.bam",
    },
    "slamseq_example_tp10": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_10.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_10.cr0.05.STAR.bam",
    },
    "slamseq_example_tp15": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_15.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_15.cr0.05.STAR.bam",
    },
    "slamseq_example_tp20": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_20.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_20.cr0.05.STAR.bam",
    },
    "slamseq_example_tp25": {
        "uri": f"file:///{os.path.dirname(os.path.realpath(__file__))}/static_test_files/slamseq_example_25.cr0.05.STAR.bam",
        "filename": "bam/slamseq_example_25.cr0.05.STAR.bam",
    },
}


def list_resources():
    """returns a List of all available test resources"""
    lst = [k for k in test_resources.keys()]
    lst += [k for k in large_test_resources.keys()]
    lst += [f"pybedtools::{k}" for k in pybedtools.filenames.list_example_files()]
    return lst


def get_resource(k, data_dir: Path = None, conf=None, create_resource=False):
    """
    Return a file link to the test resource with the passed key.
    If the passed key starts with 'pybedtools::<filename>', then the respective pybedtools test file will be
    returned.

    Examples
    --------
    >>> get_resource("gencode_gff")
    >>> get_resource("pybedtools::snps.bed.gz")

    """
    if data_dir is None:
        data_dir = rna.__RNALIB_TESTDATA__
    if conf is None:
        conf = test_resources
        conf.update(large_test_resources)
    if k.startswith("pybedtools::"):
        k = k[len("pybedtools::") :]
        return pybedtools.filenames.example_filename(k)
    assert k in conf, f"No test resource with key {k} defined in passed config"
    resfile = str(data_dir) + f"/{conf[k]['filename']}"
    if create_resource:
        try:
            # this method will download only if the resource doe not (completely) exist
            download_bgzip_slice(conf, k, data_dir, view_tempdir=False)
        except Exception as ex:
            logging.debug(traceback.format_exc())
            logging.error(
                f"Error creating resource {k}: {ex}. Some tests may not work..."
            )
            return None
    return resfile


def download_bgzip_slice(
    config, resource_name, outdir, view_tempdir=False, show_progress=True
):
    """
    Download the resource with the passed name from the passed config dict.
    The resource will be downloaded to the testdata directory and will be sorted, compressed and indexed.
    If the resource is a genomic file (gff, gtf, bed, vcf, fasta, bam), then a subregion can be sliced from the
    file by passing a list of regions in the 'regions' key of the resource dict.

    This method requires the following external tools to be installed:
    * samtools
    * bedtools
    * htslib (bgzip, tabix)

    Examples
    --------
    >>> download_bgzip_slice(test_resources, "gencode_gff", "/path/to/testdata")
    """
    res = config[resource_name]
    outfile = f"{outdir}/{res['filename']}"
    outfiledir, outfilename = os.path.dirname(outfile), os.path.basename(outfile)
    ff = res.get("format", rna.guess_file_format(outfile))
    tabix_options = res.get("tabix_options", "")
    if ff == "bedgraph":
        tabix_options += " -p bed"  # ensure BED coordinate presets for bedgraph format
    # check expected resources
    outfiles = [outfile]  # list of created files (e.g., bam + bai)
    if ff in ["gff", "gtf", "bed", "bedgraph", "vcf"]:
        outfiles += [outfile + ".tbi"]  # tabix index
    elif ff == "fasta":
        outfiles += [outfile + ".fai"]  # fasta index
        if outfile.endswith(".gz"):
            outfiles += [outfile + ".gzi"]  # fasta index
    elif ff == "bam":
        outfiles += [outfile + ".bai"]  # bam index
    # check whether all expected resources already exist
    if not res.get("include", True):
        logging.debug("Not included. Skipping...")
        return
    if all([os.path.isfile(of) for of in outfiles]) and not res.get("recreate", False):
        logging.debug("Resource already exists, skipping...")
        return
    logging.debug(f"Creating testdataset {resource_name}")
    rna.ensure_outdir(outfiledir)  # ensure outdir
    # download data
    with tempfile.TemporaryDirectory() as tempdirname:
        try:
            logging.debug(f"Downloading {res['uri']} to {tempdirname}/{outfilename}")
            f = rna.download_file(
                res["uri"], f"{tempdirname}/{outfilename}", show_progress=show_progress
            )
            assert os.path.isfile(f), "Could not download file..."
            if ff in [
                "gff",
                "gtf",
                "bed",
                "bedgraph",
                "vcf",
            ]:  # use bgzip and index with tabix, sort, slice
                tmpfile = f"{tempdirname}/sorted.{ff}.gz"
                subprocess.call(
                    f"bedtools sort -header -i {f} | bgzip > {tmpfile}", shell=True
                )  # index

                subprocess.call(f'tabix {tabix_options} {tmpfile}', shell=True)  # index
                f = tmpfile
            elif ff == "fasta":
                if outfile.endswith(".gz"):
                    tmpfile = f"{tempdirname}/bgzip.{ff}.gz"
                    subprocess.call(
                        f"gunzip -c {f} | bgzip > {tmpfile}", shell=True
                    )  # index
                    f = tmpfile
                logging.debug(f"Indexing FASTA file {f}")
                subprocess.call(f"samtools faidx {f}", shell=True)  # index
            elif ff == "bam":
                logging.debug(f"Indexing BAM file {f}")
                subprocess.call(f"samtools index {f}", shell=True)  # index
            # slice and copy result files
            if ff in ["gff", "gtf", "bed", "bedgraph", "vcf"]:
                if ff == "vcf":
                    # copy header of VCF file
                    vcf_header = pysam.VariantFile(f, "r").header
                if res.get("regions", None) is not None:
                    logging.debug(f"Slicing {ff}")
                    tmpfile = f"{tempdirname}/sliced.{ff}"
                    if ff == "vcf":  # copy header of VCF file
                        vcf_header = str(pysam.VariantFile(f, "r").header).strip()
                        with open(tmpfile, "wt") as out:
                            print(vcf_header, file=out)
                    for reg in res.get("regions"):  # TODO: ensure regions are sorted
                        subprocess.call(f"tabix {f} {reg} >> {tmpfile}", shell=True)
                    subprocess.call(f"bgzip {tmpfile}", shell=True)
                    subprocess.call(f"tabix {tmpfile}.gz", shell=True)
                    f = tmpfile + ".gz"
                shutil.copy(f, outfile)
                shutil.copy(f + ".tbi", outfile + ".tbi")
            elif ff == "fastq":
                if res.get("regions", None) is not None:
                    tmpfile = f"{tempdirname}/sliced.{ff}"
                    n_lines = res.get("regions")
                    subprocess.call(f"head -n {n_lines} {f} > {tmpfile}", shell=True)
                    f = tmpfile
                shutil.copy(f, outfile)
            elif ff == "fasta":
                if res.get("regions", None) is not None:
                    tmpfile = f"{tempdirname}/sliced.{ff}"
                    subprocess.call(f"touch {tmpfile}")
                    for reg in res.get("regions"):  # TODO: ensure regions are sorted
                        subprocess.call(
                            f"samtools faidx {f} {reg} >> {tmpfile}", shell=True
                        )
                    subprocess.call(f"samtools faidx {tmpfile}", shell=True)
                    f = tmpfile
                shutil.copy(f, outfile)
                shutil.copy(f + ".fai", outfile + ".fai")
                if os.path.isfile(f + ".gzi"):
                    shutil.copy(f + ".gzi", outfile + ".gzi")
            elif ff == "bam":
                if res.get("regions", None) is not None:
                    samfile = f"{tempdirname}/tmp.sam"
                    subprocess.call(
                        f"samtools view -H {f} > {samfile}", shell=True
                    )  # header
                    for reg in res.get("regions"):  # TODO: ensure regions are sorted
                        subprocess.call(
                            f"samtools view {f} {reg} >> {samfile}", shell=True
                        )
                    subprocess.call(
                        f"samtools view -hSb --output {f} {samfile}", shell=True
                    )
                    subprocess.call(f"samtools index {f}", shell=True)  # index
                shutil.copy(f, outfile)
                shutil.copy(f + ".bai", outfile + ".bai")
            else:  # default: just copy the downloaded file
                shutil.copy(f, outfile)
            # check expected  output
            assert all(
                [os.path.isfile(of) for of in outfiles]
            ), f"Missing output files: {outfiles}"

        finally:
            # allow inspection of tempdir for  debugging purposes
            if view_tempdir:
                input(
                    f"You can view your temporary files at {tempdirname}. Press enter to delete the files."
                )
        return outfile


def create_testdata(out_dir, resources=None, show_data_dir=False, mkdir=True):
    """Downloads test resources configured in the passed dicts if not existing yet.

    Parameters
    ----------
    out_dir : str
        The directory where the test data will be created.
    resources : dict, optional
        A dict or a list of dicts configureing the test resources. If None, the test_resources dict will be used.
    show_data_dir : bool, optional
        If True, the resulting test data directory will be printed.
    mkdir : bool, optional
        If True, the output directory will be created if it does not exist.
    """
    if resources is None:
        resources = test_resources
    resources = rna.to_list(resources)
    if not os.path.isdir(out_dir):
        if mkdir:
            os.makedirs(out_dir)
        else:
            raise FileNotFoundError(f"Directory {out_dir} does not exist.")
    nerr = 0
    logging.debug(f"============= Output directory: {out_dir}=========")
    for res in resources:
        with tqdm(res, desc="Creating testdata") as pbar:
            for resname in pbar:
                pbar.set_description(f"Creating test dataset '{resname}'")
                if get_resource(resname, out_dir, res, create_resource=True) is None:
                    nerr += 1
        if show_data_dir:
            print(f"============= resulting test data dir: =========")
            rna.print_dir_tree(out_dir)
    logging.info(f"========= All done with {nerr} errors  ==========")


def make_random_intervals(n=1000, chroms=("chr1",), max_coord=None, max_length=100):
    """
    Adapted from bioframe's make_random_intervals method.
    Creates a random set of intervals with the passed parameters.
    """
    n, max_length = int(n), int(max_length)
    n_chroms = len(chroms)
    max_coord = (n // n_chroms) if max_coord is None else int(max_coord)
    chroms = np.array(chroms)[np.random.randint(0, n_chroms, n)]
    starts = np.random.randint(0, max_coord, n)
    ends = starts + np.random.randint(1, max_length, n)
    values = np.random.randint(0, 1000, n)
    df = (
        pd.DataFrame({"chrom": chroms, "start": starts, "end": ends, "value": values})
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )
    return df
