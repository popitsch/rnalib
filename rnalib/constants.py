from enum import IntEnum, Enum

from IPython.core.display import Markdown

#: Convenience lists of canonical chromosome names
CANONICAL_CHROMOSOMES = {
    'GRCh38': [f"chr{c}" for c in list(range(1, 23)) + ['X', 'Y', 'M']],
    'GRCm38': [f"chr{c}" for c in list(range(1, 20)) + ['X', 'Y', 'MT']],
    'dmel': ['2L', '2R', '3L', '3R', '4', 'X', 'Y']
}

#: maximum integer value, assuming 32-bit ints
MAX_INT = 2 ** 31 - 1

#: Maps valid sub-feature types (e.g., 'exon', 'CDS') types to Sequence Ontology terms (e.g., '3UTR' -> 'three_prime_UTR')
FTYPE_TO_SO = {
    'gene': 'gene', 'ncRNA_gene': 'gene',
    'transcript': 'transcript', 'mRNA': 'transcript', 'ncRNA': 'transcript', 'lnc_RNA': 'transcript',
    'pseudogenic_transcript': 'transcript', 'pre_miRNA': 'transcript', 'rRNA': 'transcript',
    'snRNA': 'transcript', 'snoRNA': 'transcript', 'tRNA': 'transcript', 'miRNA': 'transcript',
    'exon': 'exon',
    'intron': 'intron',
    'CDS': 'CDS',
    'three_prime_UTR': 'three_prime_UTR', '3UTR': 'three_prime_UTR', 'UTR3': 'three_prime_UTR',
    'five_prime_UTR': 'five_prime_UTR', '5UTR': 'five_prime_UTR', 'UTR5': 'five_prime_UTR'
}

#: Maps info field names for various GFF fto feature types.
GFF_FLAVOURS = {
    ('gencode', 'gff'): {
        'gid': 'ID', 'tid': 'ID', 'tx_gid': 'Parent', 'feat_tid': 'Parent', 'gene_name': 'gene_name',
        'ftype_to_SO': FTYPE_TO_SO
    },
    ('gencode', 'gtf'): {
        'gid': 'gene_id', 'tid': 'transcript_id', 'tx_gid': 'gene_id', 'feat_tid': 'transcript_id',
        'gene_name': 'gene_name',
        'ftype_to_SO': FTYPE_TO_SO
    },
    ('ensembl', 'gff'): {
        'gid': 'ID', 'tid': 'ID', 'tx_gid': 'Parent', 'feat_tid': 'Parent', 'gene_name': 'Name',
        'ftype_to_SO': FTYPE_TO_SO | {'pseudogene': 'gene'}
        # 'pseudogene': maps to 'gene' in ensembl but to tx in flybase
    },
    ('flybase', 'gtf'): {
        'gid': 'gene_id', 'tid': 'transcript_id', 'tx_gid': 'gene_id', 'feat_tid': 'transcript_id',
        'gene_name': 'gene_symbol',
        'ftype_to_SO': FTYPE_TO_SO | {'pseudogene': 'transcript'}
        # 'pseudogene': maps to 'gene' in ensembl but to tx in flybase
    },
    ('ucsc', 'gtf'): {
        'gid': None, 'tid': 'transcript_id', 'tx_gid': 'gene_id', 'feat_tid': 'transcript_id', 'gene_name': 'gene_name',
        'ftype_to_SO': FTYPE_TO_SO
    },
    ('chess', 'gff'): {
        'gid': None, 'tid': 'ID', 'tx_gid': 'Parent', 'feat_tid': 'Parent', 'gene_name': 'gene_name',
        'ftype_to_SO': FTYPE_TO_SO
    },
    ('chess', 'gtf'): {
        'gid': None, 'tid': 'transcript_id', 'tx_gid': 'gene_id', 'feat_tid': 'transcript_id', 'gene_name': 'gene_name',
        'ftype_to_SO': FTYPE_TO_SO
    },
    ('mirgenedb', 'gff'): {
        'gid': None, 'tid': 'ID', 'tx_gid': None, 'feat_tid': None, 'gene_name': 'Alias',
        'ftype_to_SO': {'pre_miRNA': 'transcript', 'miRNA': 'transcript'}
    },
    ('generic', 'gff'): {'gid': 'ID', 'tid': 'ID', 'tx_gid': 'Parent', 'feat_tid': 'Parent',
                         'gene_name': 'gene_name', 'ftype_to_SO': FTYPE_TO_SO},
    ('generic', 'gtf'): {'gid': 'gene_id', 'tid': 'transcript_id', 'tx_gid': 'gene_id', 'feat_tid': 'transcript_id',
                         'gene_name': 'gene_name', 'ftype_to_SO': FTYPE_TO_SO}
}


class BamFlag(IntEnum):
    """BAM flags, @see https://broadinstitute.github.io/picard/explain-flags.html"""
    BAM_FPAIRED = 0x1  # the read is paired in sequencing, no matter whether it is mapped in a pair
    BAM_FPROPER_PAIR = 0x2  # the read is mapped in a proper pair
    BAM_FUNMAP = 0x4  # the read itself is unmapped; conflictive with BAM_FPROPER_PAIR
    BAM_FMUNMAP = 0x8  # the mate is unmapped
    BAM_FREVERSE = 0x10  # the read is mapped to the reverse strand
    BAM_FMREVERSE = 0x20  # the mate is mapped to the reverse strand
    BAM_FREAD1 = 0x40  # this is read1
    BAM_FREAD2 = 0x80  # this is read2
    BAM_FSECONDARY = 0x100  # not primary alignment
    BAM_FQCFAIL = 0x200  # QC failure
    BAM_FDUP = 0x400  # optical or PCR duplicate
    BAM_SUPPLEMENTARY = 0x800  # optical or PCR duplicate


#: default BAM flag filter (int 3844); comparable to samtools view -F 3844; also used as default filter in IGV.
DEFAULT_FLAG_FILTER = BamFlag.BAM_FUNMAP | BamFlag.BAM_FSECONDARY | BamFlag.BAM_FQCFAIL \
                      | BamFlag.BAM_FDUP | BamFlag.BAM_SUPPLEMENTARY

#: Markdown separator for jupyter notebooks; `display(SEP)`
SEP = Markdown('---')



