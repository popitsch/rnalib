"""
Test external APIs
"""
from collections import Counter

import bioframe
import pybedtools

import rnalib as rna
from rnalib import gi
from rnalib.testdata import get_resource

assert rna.__RNALIB_TESTDATA__ is not None, ("Please set rnalib.__RNALIB_TESTDATA__ variable to the testdata  "
                                             "directory path")


def test_bioframe_pitfall_example():
    bioframe_gff = bioframe.read_table(get_resource("gencode_gff"), schema='gff')[['chrom', 'start', 'end', 'strand']]
    bioframe_bed = bioframe.read_table(get_resource("gencode_bed"), schema='bed')[['chrom', 'start', 'end', 'strand']]
    assert bioframe.is_bedframe(bioframe_gff) and bioframe.is_bedframe(bioframe_bed)  # assert that they are 'bedframes'
    # now assert that the number of differing rows equals the number of input rows
    assert len(bioframe_gff.compare(bioframe_bed).index) == len(bioframe_gff.index), "Does this work with bioframe now?"


def test_pybedtools_pitfall_example():
    """
    The following code block was copy-pasted from the [pybedtools page](https://github.com/daler/pybedtools) and the
    idea is to create a list of gene names that are <5 kb away from intergenic SNPs. However, this example does not
    work properly (with pybedtools v0.9.1) due to inconsistent chromosome order of the two input files
    (hg19.gff: chr1, chr21; snps.bed.gz: chr21, chr1; both are pybedtools test resources).

    As a result, `genes.closest` reports only -1 as distance which is why all closes genes will be reported, not just
    the ones at max 5kb distance.
    Notably, the code runs without errors/warnings and returns a reasonable-sized list of gene names which makes it
    hard to spot the error. Only when omitting the `stream=True flag`, `genes.closest` fails and reports the
    inconsistent chrom order.

    """
    # pybedtools example code with unsorted data
    gff_file = get_resource("pybedtools::hg19.gff")  # same as pybedtools.filenames.example_filename('hg19.gff')
    snp_file = get_resource("pybedtools::snps.bed.gz")
    snps = pybedtools.BedTool(snp_file)
    genes = pybedtools.BedTool(gff_file)
    intergenic_snps = snps.subtract(genes).saveas()
    nearby = genes.closest(intergenic_snps, d=True, stream=True).saveas()
    nbgenes = [gene.name for gene in nearby if int(gene[-1]) < 5000]
    assert len(nbgenes) == 4217, "does this work with this pybedtools version now?"

    # now re-run with sorted input files
    gff_file = get_resource("pybedtools_gff")  # this is a sorted, bgzipped and indexed version of the gff file
    snp_file = get_resource("pybedtools_snps")  # this is a sorted, bgzipped and indexed version of the bed file
    snps = pybedtools.BedTool(snp_file)
    genes = pybedtools.BedTool(gff_file)
    intergenic_snps = snps.subtract(genes).saveas()
    nearby = genes.closest(intergenic_snps, d=True, stream=True).saveas()
    nbgenes = [gene.name for gene in nearby if int(gene[-1]) < 5000]
    assert len(nbgenes) == 2422, "does this work with this pybedtools version now?"

    # NOTE: the snps BED file contains 800000 lines with 799218 unique locations of which 1564
    # share the same start and stop coordinate (i.e., empty intervals). proof:
    # cat bed/pybedtools_snps.bed.gz | gunzip | cut -f1-3  | awk '{if($2==$3){print $1"\t"$2"\t"$3 }}' | wc -l
    esnp = [x for loc, x in rna.BedIterator(snp_file).to_list() if
            loc.is_empty()]  # count 'empty' intervals in this BED
    # file
    assert len(esnp) == 1564
    assert len([loc for loc, _ in rna.BedIterator(snp_file)]) == 800000
    assert len([loc for loc, x in rna.BedIterator(snp_file) if not loc.is_empty()]) == 800000 - 1564

    # now we recreate with rnalib methods.
    # first we collect intergenic snps, i.e., we use an AnnotationIterator and save all snps with no overlapping gene(s)
    isnp = []
    gsnp, esnp = Counter(), Counter()
    with rna.AnnotationIterator(rna.BedIterator(snp_file), rna.GFF3Iterator(gff_file)) as it:
        for loc, (v1, v2) in it:
            if loc.is_empty():
                esnp['empty'] += 1
            elif len(v2) == 0:
                isnp.append(loc)
            else:
                for g in v2:
                    gsnp[g.data['ID']] += 1

    # manually remove these wrong entries
    pbt_loc = {gi(pi.chrom, pi.start, pi.end) for pi in intergenic_snps if pi.start != pi.end}
    pgl_loc = {gi(loc.chromosome, loc.start - 1, loc.end) for loc in isnp}
    # assert we found the same number of intergenic SNPs
    assert len(pbt_loc), len(pgl_loc)

    # now we query for genes +/- 5kb using an interval tree. Note that this would report *all* genes
    # within this genomic window, not just the closest one.
    # However, we assume that this is the intended behaviour of this analysis, and it does not make a
    # difference for this dataset.
    itree = rna.GFF3Iterator(gff_file).to_intervaltrees()  # will drop empty intervals
    close_genes = set()
    for snp in isnp:
        for g in itree[snp.chromosome][snp.start - 5000:snp.end + 5000]:
            close_genes.add(g.data['ID'])
    # now assert we found the same (unique) gene names:
    assert len(set(nbgenes)) == len(set(close_genes))
    # NOTE that this rnalib approach is much slower but also more flexible. We can now easily filter, e.g., only for
    # an upstream window, or we could use intervaltree envelop queries for reporting genes that are fully within the
    # query window or treat genes with different annotated gene_type differently.
    # Note that we also collected the number of genic SNPs per gene in the gsnp Counter.
