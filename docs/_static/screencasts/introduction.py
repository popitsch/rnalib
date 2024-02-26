import rnalib as rna  # load rnalib (t=1)
rna.__RNALIB_TESTDATA__ = "../../../notebooks/rnalib_testdata/"  # point to your testdata directory
t = rna.Transcriptome(  # build a transcriptome, (t=5) \
    genome_fa=rna.get_resource('dmel_genome'),  # genome FASTA \
    annotation_gff=rna.get_resource('flybase_gtf'),  # Gene annotation GTF/GFF file \
    annotation_flavour='flybase',  # flavour of the annotation file. \
    load_sequence_data=True,  # load sequences from configured genome FASTA file \
    disable_progressbar=True,  # no progressbars, \
    feature_filter={'location': {'included': {'chromosomes': ['2L']}}} # a simple filter that will include only annotations from 2L \
)
display(t)  # print the transcriptome object (t=3)
display(t.log)  # log of the build process
display(t.get_struct())  # get the structure of the transcriptome
display(dir(t))  # list of methods and attributes
display(t.genes)  # list of genes
display([g.gene_name for g in t.genes])  # list of gene names
display([tx.feature_id for tx in t.gene['l(2)gl'].transcript])  # list of transcript IDs for gene 'l(2)gl'

ex = t['FBtr0078103'].exon[2]  # get the 3rd exon of a transcript with given id
t[ex]['my_great_annotation'] = 24.9  # add arbitrary annotations
display(vars(ex))  # show data that is directly associated with this exon
display(t[ex])  # show custom annotation
display(ex.rnk)  # implicitly modeled annotations: rnk is index in exon list
display(ex.parent.feature_id)  # access via feature hierarchy (gene->transcript->exon)
display(ex.sequence[:10])  # derived annotations: access gene-associated sequence data and slice exon coordinates
display(t.query(ex))  # query for any genomic interval via intervaltrees
display({tx.parent.gene_name for tx in t.transcripts if tx.strand == '-' and len(tx.exon) >= 2 and any([len(i) > 1000 for i in tx.intron])})  # query via python list comprehension
display(rna.it(t, feature_types='exon').to_dataframe().query("my_great_annotation > 20"))  # convert to pandas dataframe and filter
display(rna.it(t, feature_types='exon', region='2L:90000-100000').to_bed(None))  # iterate exons in given region, convert to BED and print as tuples
display([(loc, len(dat)) for loc, dat in rna.it(t, feature_types='gene').tile(tile_size=1e5) if len(dat) > 0])  # iterate all genes, tile into windows of size 100k, count genes and show tile and count
# and much, much more... :-) (t=10)
