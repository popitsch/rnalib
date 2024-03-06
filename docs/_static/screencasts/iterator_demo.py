import rnalib as rna  # load rnalib (t=0)
rna.__RNALIB_TESTDATA__ = "../../../notebooks/rnalib_testdata/"  # point to your testdata directory
from rnalib import gi

# rna.it(...) is a factory method that returns an iterator for any supported data type
regions = {
    "A": gi("chr1:1-100"),
    "B": gi("chr1:101-200"),
    "C": gi("chr1:201-300"),
    }  # a dict of genomic regions
for loc, dat in rna.it(regions):  # use rna.it(...) factory method to iterate any supported data type
    display(loc, dat)  # print location and data (t=1)

# rna.it(<bam_file>) creates an rnalib ReadIterator that enables filtered iteration over BAM files
with rna.it(
        rna.get_resource("small_example_bam"),  # iterate over a BAM file using the default flag filter
        tag_filters=[rna.TagFilter("MD", ["100"], inverse=True)],
    ) as it:  # tag filter: MD tag must be '100'
    display(Counter([r.get_tag("MD") for _, r in it]))  # count MD tags of iterated reads to verify the filter
    display(it.stats)  # print stats of the iteration

# rnalib also implements a fast pileup iterator
for loc, dat in rna.it(rna.get_resource("small_example_bam"),  # iterate over a BAM file
                       style="pileup",  # pileup style
                       region=gi("1:22413312-22413317")):  # region to pileup
    display(f"{loc}: {dat}")  # print location and allele counts


# you can also iterate features in a transcriptome:
t = rna.Transcriptome(  # build a transcriptome
    genome_fa=rna.get_resource("dmel_genome"),  # genome FASTA
    annotation_gff=rna.get_resource("flybase_gtf"),  # Gene annotation GTF/GFF file
    annotation_flavour="flybase",
    )  # flavour of the annotation file

display( Counter([
    len(i) > 1000 for i in rna.it(t, feature_types="intron")  # iterate introns and count those with length > 1000
    ]))

# rnalib iterators can be 'tiled' into windows of a given size
display({ str(loc): len(dat)
          for loc, dat in rna.it(t, feature_types="gene").tile(tile_size=1e4)
          if len(dat) > 0
        })  # tile transcriptome into windows of size 10k, count genes per tile and display (t=2)

# You can also add arbitrary annotations to iterated features
for i, _ in rna.it(t, feature_types="exon"):
    t[i]["my_great_annotation"] = random.uniform(0, 1)

# rnalib iterators can be filtered and converted to other data types
display(
    rna.it(t, feature_types="exon"). # iterate all exons...
    to_dataframe(excluded_columns=["source", "gene_type", "gene_name"]). # ...convert to pandas DataFrame...
    query("my_great_annotation > 0.5")  # ... and filter with pandas
    )

# rna.it(...) can iterate the following genomic region iterables:
# * dict, list, tuple, set
# * BAM, VCF, GFF, GTF, BED, Fasta, Fastq, BigBed, BigWig and any Tabix indexed file
# * Pybedtools, Bioframe, Pyranges, Pandas datasets
# * Transcriptome features
# Iterators can be tiled, grouped and synchronized with other iterators
# See the documentation for more details (t=10)
#
