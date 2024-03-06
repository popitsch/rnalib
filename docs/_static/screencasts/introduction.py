import rnalib as rna  # load rnalib (t=0)

rna.__RNALIB_TESTDATA__ =  "../../../notebooks/rnalib_testdata/"  # point to your testdata directory


# You can easily build a transcriptome from a genome FASTA and a gene annotation GTF/GFF file...
t = rna.Transcriptome(  # build a transcriptome... (t=2)
    genome_fa=rna.get_resource("dmel_genome"),  # genome FASTA
    annotation_gff=rna.get_resource("flybase_gtf"),  # gene annotation GTF/GFF file
    annotation_flavour="flybase",  # flavour of the annotation file.
    load_sequence_data=True,  # load sequences from configured genome FASTA file
    disable_progressbar=True,  # no progressbars,
    feature_filter={
        "location": {"included": {"chromosomes": ["2L"]}} # filter that includes only chromosome 2L
    })
# ... and inspect it
display(t)  # print the transcriptome object (t=0)
display(t.log)  # log of the build process
display(t.get_struct())  # get the structure of the transcriptome
display(dir(t))  # list of methods and attributes
# You can access genes, transcripts, exons, introns, CDS, UTRs, and much more...
display(t.genes)  # list of genes
display([g.gene_name for g in t.genes])  # list of gene names
display([tx.feature_id for tx in t.gene["l(2)gl"].transcript])  # list of transcript IDs for gene 'l(2)gl'

# You can also access sequences and annotations...
ex = t["FBtr0078103"].exon[2]  # get the 3rd exon of a transcript with given id
t[ex]["my_great_annotation"] = 24.9  # add arbitrary annotations
display(vars(ex))  # show data that is directly associated with this exon
display(t[ex])  # show custom annotation
display(ex.rnk)  # implicitly modeled annotations: rnk is index in exon list
display(ex.parent.feature_id)  # access via feature hierarchy (gene->transcript->exon)
display(ex.sequence[:10])  # derived annotations: access gene-associated sequence data and slice exon coordinates
display(t.query(ex))  # query for any genomic interval via intervaltrees
display({tx.parent.gene_name for tx in t.transcripts
        if tx.strand == "-" and len(tx.exon) >= 2 and any([len(i) > 1000 for i in tx.intron])})  # query via python list comprehension

# and much, much more... :-) (t=10)
#
