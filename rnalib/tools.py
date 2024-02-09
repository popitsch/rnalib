import ast
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import intervaltree
import pandas as pd
import pysam

import rnalib as rna
from tqdm.auto import tqdm

from rnalib import DEFAULT_FLAG_FILTER


def prune_tags(bam_in, bam_out, kept_tags=('NH', 'HI', 'AS')):
    """
    Parameters
    ----------
    bam_in
        input bam file
    bam_out
        output bam file
    kept_tags
        list of tags to be kept

    Example
    -------
    >>> prune_tags('/Users/niko/projects/rnalib/rnalib/static_test_files/small.ACTB+SOX2.bam', \
    >>> '/Users/niko/projects/rnalib/rnalib/static_test_files/small.ACTB+SOX2.clean.bam')

    """
    with rna.ReadIterator(bam_in, flag_filter=0) as it:
        with pysam.AlignmentFile(bam_out, "wb", template=it.file) as out:
            for _, r in tqdm(it):
                for tag in r.get_tags():
                    if tag[0] not in kept_tags:
                        r.set_tag(tag[0], None)
                out.write(r)


def tag_tc(bam_file, included_chrom=None, snp_vcf_file=None, out_prefix=None, fractional_counts=False,
           write_density_histogram=True, tags=None, min_mapping_quality=0, flag_filter=DEFAULT_FLAG_FILTER,
           min_base_quality=10):
    """
    Determines the T/C status per read while masking SNP positions and create density histogram per chrom.
    Will output two files: a BAM file with the T/C status encoded in the XC tag and a TSV file containing the T/C
    density per chrom.

    Parameters
    ----------
    bam_file : str
        bam file to be analysed
    included_chrom : list
        list of chromosomes to be included (subset of reference dict)
    snp_vcf_file : str
        VCF file containing variant calls from the respective cell line. T/C and A/G SNPs will be masked
    out_prefix : str
        output file prefix
    fractional_counts : bool
        if set, reads will be counted by 1/NH
    write_density_histogram : bool
        if set, a (gzipped) TSV file containing T/C density data will be written. default is True
    tags : dict
        dictionary containing the BAM tags to be used for the output BAM file.
        Default is {'ntc': 'xc', 'ntt': 'xt', 'col': 'YC'}
    min_mapping_quality : int
        minimum mapping quality
    flag_filter : int
        flag filter for filtering reads.
    min_base_quality: int
        minimum base quality

    """
    if tags is None:
        tags = {'ntc': 'xc', 'ntt': 'xt', 'col': 'YC'}  # Used BAM tags
    elif isinstance(tags, str):
        tags = ast.literal_eval(tags)  # parse from string
    if included_chrom is not None and isinstance(included_chrom, str):
        try:
            included_chrom = ast.literal_eval(included_chrom)
        except ValueError as e:
            included_chrom = included_chrom.split(',')  # parse from string
    out_prefix = rna.remove_extension(bam_file) if out_prefix is None else out_prefix
    out_bam_file = f'{out_prefix}+tc.bam'
    out_tsv_file = f'{out_prefix}.density.tsv'
    profile = Counter()
    # reference dict containing canonical chrom in correct order
    if snp_vcf_file is None:
        refdict = rna.RefDict.merge_and_validate(
            rna.RefDict.load(bam_file),
            included_chrom=included_chrom)
    else:
        refdict = rna.RefDict.merge_and_validate(
            rna.RefDict.load(bam_file), rna.RefDict.load(snp_vcf_file),
            included_chrom=included_chrom)
    # print(f'Considered chromosomes: {refdict.keys()}')
    with rna.ReadIterator(bam_file,
                          report_mismatches=False,
                          min_mapping_quality=min_mapping_quality,
                          flag_filter=flag_filter,
                          min_base_quality=min_base_quality) as it:
        with pysam.AlignmentFile(out_bam_file, "wb", template=it.file) as out:  # @UndefinedVariable
            with tqdm(total=it.max_items()) as pbar:
                for chrom in refdict:
                    reg = rna.gi(chrom)
                    it.set_region(reg)
                    # get genomic positions of all T/C and A/G snps in the considered region
                    masked_pos = set([loc.start for loc, snp in rna.VcfIterator(snp_vcf_file, region=reg) if
                                      ((snp.ref, snp.alt) == ('T', 'C')) or (
                                              (snp.ref, snp.alt) == ('A', 'G'))]) if snp_vcf_file is not None else set()
                    pbar.set_description(f"{chrom} with {len(masked_pos)} masked positions")
                    for loc, r in it:
                        pbar.update(1)
                        is_rev = not r.is_reverse if r.is_read2 else r.is_reverse
                        refc = "A" if is_rev else "T"
                        altc = "G" if is_rev else "C"
                        mm = [(off, pos + 1, ref.upper(), r.query_sequence[off]) for (off, pos, ref) in
                              r.get_aligned_pairs(with_seq=True, matches_only=True) if (ref.upper() == refc) and
                              (r.query_qualities[off] >= it.min_base_quality) and (pos + 1 not in masked_pos)]
                        mm_tc = [(off, pos1, ref, alt) for off, pos1, ref, alt in mm if alt == altc]
                        # count value of this read. if fractional_counts is True: count by 1/NH
                        cnt = (1 / r.get_tag("NH") if r.has_tag("NH") else 1) if fractional_counts else 1
                        profile[r.reference_name, refc, len(mm), len(mm_tc)] += cnt
                        # set bam tags, etc.
                        r.set_tag(tag=tags['ntt'], value=len(mm), value_type="i")
                        r.set_tag(tag=tags['ntc'], value=len(mm_tc), value_type="i")
                        if len(mm_tc) > 0:
                            ftc = len(mm_tc) / len(mm)
                            r.set_tag(tag=tags['col'], value=f'{255 - int(255 * ftc)},0,0',
                                      value_type="Z")  # set color relative to conversion rate
                        out.write(r)
    try:
        pysam.index(out_bam_file)  # @UndefinedVariable
    except Exception as e:
        print("error indexing bam: %s" % e)
    # filter for min number of counts
    profile = pd.DataFrame([list(k) + [v] for k, v in profile.items()],
                           columns=['chromosome', 'ref', 'convertible', 'converted', 'count'])
    if write_density_histogram:
        profile.to_csv(out_tsv_file, index=False, sep='\t')
        rna.bgzip_and_tabix(out_tsv_file, create_index=False)  # bgzip
    else:
        out_tsv_file = None
    return profile, out_bam_file, out_tsv_file


def filter_tc(bam_file, out_file=None, min_tc=1, tags=None):
    """
    Filters a T/C annotated BAM file for reads with at least min_tc T/C conversions.

    Parameters
    ----------
    bam_file : str
        T/C annotated bam file to be analysed
    out_prefix : str
        output file prefix
    min_tc : int
        minimum number of T/C conversions
    tags : dict
        dictionary containing the BAM tags to be used for the output BAM file.
        Default is {'ntc': 'xc', 'ntt': 'xt', 'col': 'YC'}
    """
    if tags is None:
        tags = {'ntc': 'xc', 'ntt': 'xt', 'col': 'YC'}  # Used BAM tags)
    elif isinstance(tags, str):
        tags = ast.literal_eval(tags)  # parse from string
    if out_file is None:
        out_file = f'{Path(bam_file).stem}_tc-only.bam'
    with rna.ReadIterator(bam_file,
                          report_mismatches=False,
                          min_mapping_quality=0,
                          flag_filter=0,
                          min_base_quality=0) as it:
        with pysam.AlignmentFile(out_file, "wb", template=it.file) as out:  # @UndefinedVariable
            for loc, r in tqdm(it, total=it.max_items()):
                if r.get_tag(tags['ntc'], 0) >= min_tc:
                    out.write(r)
    try:
        pysam.index(out_file)  # @UndefinedVariable
    except Exception as e:
        print("error indexing bam: %s" % e)
    return out_file


def build_amplicon_resources(transcriptome_name, bed_file, fasta_file, out_dir, padding=100, amp_extension=100):
    """ Builds resources for amplicon analyses from a bed file.
        Intervals are parsed from the bed file (and possibly extended and padded= and sequences are extracted from the
        fasta file. The sequences are written to a FASTA file. For downstream analyses, a chrom.sizes file,
        a .dict file, a GFF3 file and a BED file are created. Additionally, a log file is written to the output
        directory.

        The parsed intervals are merged and only one 'amplicon' per merged interval is written to the FASTA file.
        The GFF3 file contains the amplicon coordinates and the BED file contains the amplicon coordinates and the
        original bed item score and strand information.

        The created amplicon sequences may be padded 3' and 5' with Ns (needed by some mappers). The amplicon sequences
        can also be extended 3' and 5' which is recommended to allow mappers to map reads that extend beyond the
        amplicon coordinates. The created FASTA file is indexed with samtools faidx. The GFF3 and BED files are bgzipped
        and tabix indexed.

        Parameters
        ----------
        transcriptome_name : str
            name of the transcriptome
        bed_file : str
            bed file containing the amplicon coordinates
        fasta_file : str
            fasta file containing the reference genome
        out_dir : str
            output directory
        padding : int
            padding to be added to the amplicon sequences
        amp_extension : int
            extension of the amplicon sequences

        Example
        -------
        >>> build_amplicon_resources('test_amplicon', 'amplicons.bed', 'hg38.fa', ''.)

        Returns
        -------
        log: dict
            a dictionary containing information about the created resources
    """
    print(f'extracting sequences for {transcriptome_name}')
    refdict = rna.RefDict.load(fasta_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log = Counter()
    # get amplicon coordinates
    ampcoords = {}
    for loc, bed_item in rna.BedIterator(bed_file):
        if loc.chromosome not in refdict:
            print(f"Chromosome {loc.chromosome} not found in reference dict. Skipping.")
            continue
        ampchrom = f"{bed_item.name}_{loc.chromosome}_{loc.start}_{loc.end}"
        ampstart = max(1, loc.start - amp_extension)  # avoid negative start
        ampend = loc.end + amp_extension
        if loc.chromosome not in ampcoords:
            ampcoords[loc.chromosome] = intervaltree.IntervalTree()
        ampcoords[loc.chromosome].addi(ampstart, ampend, ampchrom)
        log['parsed_intervals'] += 1
    for chrom in ampcoords:  # merge intervals and concat names
        ampcoords[chrom].merge_overlaps(data_reducer=lambda a, b: '+'.join([a, b]), strict=False)
    # create FASTA + GFF
    out_file_fasta = f"{out_dir}/{transcriptome_name}.fa"
    out_file_chrsize = f"{out_dir}/{transcriptome_name}.fa.chrom.sizes"
    out_file_dict = f"{out_dir}/{transcriptome_name}.fa.dict"
    out_file_gff3 = f"{out_dir}/{transcriptome_name}.gff3"
    out_file_bed = f"{out_dir}/{transcriptome_name}.bed"
    written_fa = set()
    with open(out_file_fasta, 'w') as out_fasta:
        with open(out_file_chrsize, 'w') as out_chrsize:
            with open(out_file_dict, 'w') as out_dict:
                with open(out_file_gff3, 'w') as out_gff3:
                    with open(out_file_bed, 'w') as out_bed:
                        with pysam.FastaFile(fasta_file) as genome:  # @UndefinedVariable
                            for loc, bed_item in rna.BedIterator(bed_file):
                                if loc.chromosome not in refdict:
                                    continue  # skip
                                gi = next(iter(ampcoords[loc.chromosome][
                                               loc.start:loc.end]))  # get amplicon interval (genomic coords)
                                ampstart, ampend, ampchrom = gi.begin, gi.end, gi.data
                                offset = padding + (loc.start - ampstart) + 1
                                seq = 'N' * padding + genome.fetch(loc.chromosome, ampstart - 1, ampend) + 'N' * padding
                                ampchrom_len = len(seq)
                                log['mean_amplicon_length'] += ampchrom_len
                                # FASTA
                                if ampchrom not in written_fa:
                                    log['written_amplicons'] += 1
                                    print('>%s' % ampchrom, file=out_fasta)
                                    print(rna.format_fasta(seq), file=out_fasta)
                                    # chrom.sizes file
                                    print('%s\t%i' % (ampchrom, ampchrom_len), file=out_chrsize)
                                    # DICT
                                    print('@SQ\tSN:%s\tLN:%i\tM5:%s\tUR:file:%s' % (ampchrom,
                                                                                    ampchrom_len,
                                                                                    hashlib.md5(seq.encode(
                                                                                        'utf-8')).hexdigest(),
                                                                                    # M5 MD5 checksum of the sequence in the uppercase, excluding spaces but including pads (as ‘*’s)
                                                                                    os.path.abspath(out_file_fasta)),
                                          file=out_dict)
                                    written_fa.add(ampchrom)
                                # BED
                                print("\t".join([str(x) for x in [
                                    ampchrom,
                                    offset - 1,  # start
                                    offset + loc.end - loc.start,  # end
                                    bed_item.name,
                                    '.' if bed_item.score is None else bed_item.score,
                                    '.' if loc.strand is None else loc.strand
                                ]]), file=out_bed)
                                # GFF3
                                print("\t".join([str(x) for x in [
                                    ampchrom,
                                    'transcriptome_tools',
                                    'transcript',
                                    offset,  # start
                                    offset + loc.end - loc.start,  # end
                                    '.' if bed_item.score is None else bed_item.score,
                                    '.' if loc.strand is None else loc.strand,
                                    0,
                                    f'gene_name={bed_item.name}'
                                ]]), file=out_gff3)
    log['mean_amplicon_length'] = log['mean_amplicon_length'] / log['written_amplicons']
    out_file_log = f"{out_dir}/{transcriptome_name}.log"
    with open(out_file_log, 'wt') as file:
        file.write(json.dumps(log))
    # compress + index output files
    rna.bgzip_and_tabix(out_file_gff3, sort=True, seq_col=0, start_col=3, end_col=4, line_skip=0, zerobased=False)
    rna.bgzip_and_tabix(out_file_bed, sort=True, seq_col=0, start_col=1, end_col=2, line_skip=0, zerobased=True)
    pysam.faidx(out_file_fasta)  # @UndefinedVariable
    print("Created resources:")
    print("FASTA file + idx:\t" + out_file_fasta)
    print("CHROMSIZE file:\t" + out_file_chrsize)
    print("DICT file:\t" + out_file_dict)
    print("GFF file + idx:\t" + out_file_gff3)
    print("BED file + idx:\t" + out_file_bed)
    return log
