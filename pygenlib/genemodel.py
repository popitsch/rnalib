"""

Gene model classes.

"""
from sortedcontainers import SortedList

from pygenlib.utils import gi, reverse_complement, get_config, kmer_search, find_all, to_str, \
    bgzip_and_tabix, get_reference_dict, open_file_obj, ReferenceDict, toggle_chr
from pygenlib.iterators import GFF3Iterator, TranscriptomeIterator

from enum import Enum
from collections import Counter
from dataclasses import dataclass
import pandas as pd
import dill
import pysam
from tqdm import tqdm
import gc
import mygene
from itertools import chain, pairwise
import numpy as np
from multiprocessing import Pool, Process, Queue
import psutil
import os
import re
from more_itertools import interleave_longest, triplewise
from intervaltree import Interval, IntervalTree


# ------------------------------------------------------------------------
# gene symbol abstraction
# ------------------------------------------------------------------------

@dataclass
class gene_symbol:
    """
        Class for representing a gene symbol, name and taxonomy id.
    """
    symbol: str  #
    name: str  #
    taxid: int  #

    def __repr__(self):
        return f"{self.symbol} ({self.name}, tax: {self.taxid})"


def geneid2symbol(gene_ids):
    """
        Queries gene names for the passed gene ids from MyGeneInfo via https://pypi.org/project/mygene/
        Gene ids can be, e.g., EntrezIds (e.g., 60) or ensembl gene ids (e.g., 'ENSMUSG00000029580') or a mixed list.
        Returns a dict { entrezid : gene_symbol }
        Example: geneid2symbol(['ENSMUSG00000029580', 60]) - will return mouse and human actin beta
    """
    mg = mygene.MyGeneInfo()
    galias = mg.getgenes(set(gene_ids), filter='symbol,name,taxid')
    id2sym = {x['query']: gene_symbol(x.get('symbol', x['query']), x.get('name', None), x.get('taxid', None)) for x in
              galias}
    return id2sym


def read_alias_file(gene_name_alias_file) -> (dict, set):
    """ read gene name aliases from the passed file.
        Currently supports the download format from genenames.org and vertebrate.genenames.org.
        Returns an alias dict and a set of currently known (active) gene symbols
    """
    aliases = {}
    current_symbols = set()
    if gene_name_alias_file:
        tab = pd.read_csv(gene_name_alias_file, sep='\t',
                          dtype={'alias_symbol': str, 'prev_symbol': str, 'symbol': str}, low_memory=False,
                          keep_default_na=False)
        for _, r in tqdm(tab.iterrows(), desc='load gene aliases', total=tab.shape[0]):
            current_symbols.add(r['symbol'].strip())
            for a in r['alias_symbol'].split("|"):
                aliases[a.strip()] = r['symbol'].strip()
            for a in r['prev_symbol'].split("|"):
                aliases[a.strip()] = r['symbol'].strip()
    return aliases, current_symbols


def norm_gn(g, current_symbols=None, aliases=None) -> str:
    """
    Normalizes gene names. Will return an uppercase version of the passed gene name.
    If a set of current (up-to date) symbols is
    an alias table is
    """
    g = g.strip()  # remove whitespace
    if (current_symbols is not None) and (g in current_symbols):
        return g  # there is a current symbol so don't look for aliases
    if aliases is None:
        return g
    return g if aliases is None else aliases.get(g, g)


# ------------------------------------------------------------------------
# Transcriptome model
# ------------------------------------------------------------------------
class Feature(gi):
    """
        A genomic (annotation) feature
        TODO: test __eq__
    """

    def __init__(self, parent, chromosome, start, end, strand):
        super().__init__(chromosome, start, end, strand)
        self.parent = parent

    def __eq__(self, other):
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.parent == other.parent) and super().__eq__(other)

    def __repr__(self):
        return f"{type(self)}@{super().__repr__()}"

    def set_location(self, loc):
        super().__init__(loc.chromosome, loc.start, loc.end, loc.strand)
    def location(self):
        return gi(self.chromosome, self.start, self.end, self.strand)

    def get_seq(self, mode='dna'):
        """
            Returns the 5'-3' DNA sequence (as shown in a genome browser) of this feature.
            If mode is rna then the reverse complement of negative-strand features will be returned.
        """
        if hasattr(self, 'dna_seq'):
            seq = self.dna_seq
        elif self.parent is not None:  # get from parent
            off = self.start - self.parent.start
            seq = self.parent.get_seq(mode='dna')[off:off + len(self)]
            # print(f"from parent {self.parent} with off {off}")
        if (seq is not None) and (mode == 'rna') and (self.strand == '-'):
            seq = reverse_complement(seq)
        return seq


class Gene(Feature):
    def __init__(self, gid, name, chromosome, start, end, strand, transcriptome=None):
        super().__init__(None, chromosome, start, end, strand)
        self.gid = gid
        self.name = name
        self.transcripts = {}
        self.dna_seq = None
        self.transcriptome = transcriptome
        self.is_par=False # true if there are multiple entries with the same gid in the annotation

    def __eq__(self, other):
        # NOTE: comparison only done on gid/name/coords
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.gid == other.gid) and (self.name == other.name) and super().__eq__(other)

    def __repr__(self):
        return f"{self.name}({self.gid}) ({len(self.transcripts) if self.transcripts is not None else 0} tx)"


class Transcript(Feature):
    """
        A transcript, identified by its transcript_id (tid).
        Features: exons, introns, utr3 and utr5
    """

    def __init__(self, gene, tid, chromosome, start, end, strand):
        super().__init__(gene, chromosome, start, end, strand)
        self.tid = tid
        self.exons = list()
        self.introns = list()
        self.utr3 = list()
        self.utr5 = list()

    def __repr__(self):
        return f"Tx({self.tid}) ({self.parent.name})"

    def __eq__(self, other):
        # NOTE: comparison only done on gene and tid/coords
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.tid == other.tid) and super().__eq__(other)

    def get_spliced_seq(self, sep=''):
        """The fully spliced sequence, always in 'rna' mode."""
        return sep.join([ex.get_seq(mode='rna') for ex in self.exons])

    def features(self):
        """Returns a sorted list of features (exons, introns, utrs) of this transcript"""
        return sorted(chain(self.exons, self.introns, self.utr5, self.utr3))


class Exon(Feature):
    def __init__(self, transcript, rnk, chromosome, start, end, strand):
        super().__init__(transcript, chromosome, start, end, strand)
        self.rnk = rnk

    def __repr__(self):
        return f"Exon{self.rnk} ({self.parent})"

    def __eq__(self, other):
        # NOTE: comparison only done on tx, rnk and coordinates
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.rnk == other.rnk) and super().__eq__(other)


class Intron(Feature):
    def __init__(self, transcript, rnk, chromosome, start, end, strand):
        super().__init__(transcript, chromosome, start, end, strand)
        self.rnk = rnk

    def __repr__(self):
        return f"Intron{self.rnk} ({self.parent})"

    def __eq__(self, other):
        # NOTE: comparison only done on tx, rnk and coordinates
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.rnk == other.rnk) and super().__eq__(other)


class Utr5(Feature):
    def __init__(self, transcript, chromosome, start, end, strand):
        super().__init__(transcript, chromosome, start, end, strand)

    def __repr__(self):
        return f"5'UTR ({self.parent})"

    def __eq__(self, other):
        # NOTE: comparison only done on tx and coordinates
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return super().__eq__(other)


class Utr3(Feature):
    def __init__(self, transcript, chromosome, start, end, strand):
        super().__init__(transcript, chromosome, start, end, strand)

    def __repr__(self):
        return f"3'UTR ({self.parent})"

    def __eq__(self, other):
        # NOTE: comparison only done on tx and coordinates
        if (other is None) or (not isinstance(other, __class__)):
            return False
        return (self.parent == other.parent) and super().__eq__(other)


class TranscriptFilter:
    """
        For filtering transcript annotations based on GFF/GTF locations, attributes or transcript ids (tid)s.
        Supported (optional) filter sections:
        - included_tags: list of tags that must be set. Use, e.g., ['Ensembl_canonical'] to load
                 only canonical tx or ['basic'] for GENCODE basic entries. default:[]
        - included_tids: list of transcript ids (tids) that will be included.
                if a file path (str) is configured, the list of tids is loaded from the respective file that should
                contain one tid per line. default:[]
        - included_genetypes: list of gene_types to be included.  Use, e.g., ['protein_coding'] to load
                only protein coding transcripts
        - included_chrom: list of chromosomes to be included. default:[]
        - included_regions: list of genomic regions to be included. NOTE: slow if many regions are provide here. default:[]


        NOTE that gene objects that have no associated transcript left after filtering can be dropped by
            configuring 'drop_empty_genes': True or by calling t.drop_empty_genes()

        TODO: add warning if wrong config keys used
    """

    def __init__(self, config, config_section='transcript_filter'):
        self.included_tags = set(get_config(config, [config_section, 'included_tags'], default_value=[]))
        self.included_tids = get_config(config, [config_section, 'included_tids'], default_value=[])
        self.included_genetypes = set(get_config(config, [config_section, 'included_genetypes'], default_value=[]))
        self.included_chrom = get_config(config, [config_section, 'included_chrom'], default_value=[])
        self.included_regions = get_config(config, [config_section, 'included_regions'], default_value=[])
        if len(self.included_regions) > 0:
            self.included_regions = [gi.from_str(s) for s in self.included_regions]
        # load tids from file
        if isinstance(self.included_tids, str):
            with open(self.included_tids) as f:
                tids = [line.rstrip('\n') for line in f]
            self.included_tids = tids

    def filter(self, loc, info):
        if len(self.included_tags) > 0:
            if 'tag' in info:
                tags = set(info['tag'].split(','))
                missing = self.included_tags.difference(tags)
                if len(missing) > 0:
                    return True
            else:
                return True  # no 'tag' found -> filter
        if len(self.included_tids) > 0:
            tid = info.get('transcript_id', None)
            if tid and (tid not in self.included_tids):
                return True
        if len(self.included_genetypes) > 0:
            if 'gene_type' in info:
                gene_types = set(info['gene_type'].split(','))
                missing = self.included_genetypes.difference(gene_types)
                if len(missing) > 0:
                    return True
            else:
                return True  # no 'gene_type' found -> filter
        if len(self.included_chrom) > 0:
            if loc.chromosome not in self.included_chrom:
                return True
        if len(self.included_regions) > 0:
            no_overlap = True
            for reg in self.included_regions:
                if loc.overlaps(reg):
                    no_overlap = False
                    break
            if no_overlap:
                return True
        return False

    def __repr__(self):
        if len(self.included_tags) + len(self.included_tids) + len(self.included_genetypes) + len(
                self.included_chrom) + len(self.included_regions) == 0:
            return "unfiltered"
        return f"Filtered ({len(self.included_tags)} tags, " \
               f"{len(self.included_tids)} tids, " \
               f"{len(self.included_genetypes)} genetypes, " \
               f"{len(self.included_chrom)} chroms, " \
               f"{len(self.included_regions)} regions)."


class Transcriptome:
    """
        Represents a transcriptome as modelled by a GTF/GFF file:
        - Model contains genes, transcripts and their featurs (exons, intron, 3'/5'-UTRs)
        - Feature sequences can be added via load_sequences() and accessed via get_seq(). If mode='rna' is passed, the
            sequence is returned in 5'-3' orientation, i.e., they are reverse-complemented for minus-strand transcripts.
            The returned sequence uses the DNA alphabet (ACTG) to enable direct alignment/comparison with genomic
            sequences.
        - Transcriptome features can be queried via query(). The internally used intervaltrees per feature class
            are built when query() is first called.
        - Contained transcripts can be filtered when loading, @see TranscriptFilter

        Querying and iteration examples
        -------------------------------
        - Get all gene names:
            [g.name for g in t.genes.values()]
        - Get number of exons for all ACTB transcripts
            [len(tx.exons) for tx in t.get_gene('ACTB').transcripts.values()]
        - Gene name of tx ENST00000414620.1
            t.transcripts['ENST00000414620.1'].parent.name
        - Get all transcript ids where exons overlap chr7:5529026
            {e.parent.tid for e in t.query(gi('chr7', 5529026, 5529026), Exon)}
        - Get all gene names where a kmer ("ACTGACTGACTG") if expressed in exons (requires load_sequences() )
            { tx.parent.name for tx in t.transcripts.values() if "ACTGACTGACTG" in tx.get_spliced_seq() }
        - list of genes and their up- and downstream genes if within a given distance:
            [(prv, gene, nxt) for prv, gene, nxt in t.gene_triples(max_dist=10000)]
        - coordinate-sorted list of genes in a 10kb window around ACTB:
            g=t.get_gene('ACTB')
            list(sorted(t.query(gi(g.chromosome, g.start-10000, g.end+10000), Gene)))
        - dict of genes and their unique 200bp 3'UTR intervals per tx (multiple intervals if spliced)
                {g.name: {tx.tid:calc_3end(tx) for tx in g.transcripts.values()} for g in t.genes.values()}

        IntervalTree based examples
        -------------------------------
    """

    def __init__(self, config):
        self.config = config
        self.genes, self.transcripts, self.name2gene = {}, {}, {}
        self.obj = {}
        self.log = Counter()
        self.cached = False  # if true then transcriptome was loaded from a pickled file
        self.has_seq = False
        self.itrees = {}
        self.txfilter = TranscriptFilter(self.config)
        self.merged_refdict = None
        self.build()

    def build(self):
        """
            Builds the transcriptome model from the configured GTF/GFF file.

            mandatory config properties:
                genome_fa: FASTA of reference genome
                annotation_gff: GFF/GTF file with gene model (multiple suppotred flavours)

            optional config properties:
                transcript_filter: optional transcript filter configuration (see @TranscriptFilter)
                copied_fields: field names that will be copied from the GFF info section (including
                GFF3 fields source, score and phase). Example: ['score', 'gene_type']. default: []


            Supported GFF flavours:
            - gencode
            - encode
            - ucsc (gene entries are added automatically)
            - flybase (various transcript types are parsed and gff feature type (e.g., mRNA, tRNA, etc.) is set as genee_type)
            - mirgenedb (pre_miRNA and miRNA are added as gene+transcripts with single exon; gene_type annotation is set)
            TODO:
            - Support for CDS, start_codon stop_codon
            - Add flavour autodetect (from gtf/gff header)?

        """
        # read gene aliases (optional)
        aliases, current_symbols = (None, None) if get_config(self.config, 'gene_name_alias_file',
                                                              default_value=None) is None else read_alias_file(
            get_config(self.config, 'gene_name_alias_file', default_value=None))
        # get GFF flavour
        annotation_flavour = get_config(self.config, 'annotation_flavour', required=True).lower()
        assert annotation_flavour in ['gencode', 'ensembl', 'ucsc', 'mirgenedb', 'flybase']
        # get GFF aliasing function
        annotation_fun_alias = get_config(self.config, 'annotation_fun_alias', default_value=None)
        if annotation_fun_alias is not None:
            # import importlib
            # importlib.import_module('pygenlib.utils')
            assert annotation_fun_alias in globals(), f"fun_alias function {annotation_fun_alias} not defined in pygenlib.utils"
            annotation_fun_alias = globals()[annotation_fun_alias]
            print(f"Using aliasing function for annotation_gff: {annotation_fun_alias}")
        # estimate valid chrom
        rd = [get_reference_dict(open_file_obj(get_config(self.config, 'genome_fa', required=True))),
              get_reference_dict(open_file_obj(get_config(self.config, 'annotation_gff', required=True)),
                                 fun_alias=annotation_fun_alias)]
        self.merged_refdict = ReferenceDict.merge_and_validate(*rd, check_order=False,
                                                               included_chrom=self.txfilter.included_chrom)
        assert len(self.merged_refdict) > 0, "No shared chromosomes!"
        self.log = Counter()
        # iterate gff
        for chrom in tqdm(self.merged_refdict, f"Building transcriptome ({self.txfilter})"):
            self.obj[chrom] = SortedList()
            for loc, info in GFF3Iterator(get_config(self.config, 'annotation_gff', required=True), chrom,
                                          fun_alias=annotation_fun_alias):
                self.log['parsed_gff_lines'] += 1
                if annotation_flavour in ['gencode', 'ensembl', 'ucsc', 'flybase']:
                    gid = info['gene_id']  # mandatory gtf info tag
                    if gid not in self.genes:
                        # UCSC gtf does not contain gene entries and flybase is not sorted hierarchically.
                        # So we first create a 'proxy' gene proxy object that
                        # will later be updated with tx coordinates
                        obj = Gene(gid, None, None, None, None, None, transcriptome=self)
                        self.genes[gid] = obj
                        self.obj[chrom].add(obj)
                    if info['feature_type'] == 'gene':
                        if 'gene_name' in info:  # default for genocde/ucsc
                            gname = norm_gn(info['gene_name'], current_symbols, aliases)  # normalize gene names
                        elif 'gene_symbol' in info:  # default for flybase
                            gname = norm_gn(info['gene_symbol'], current_symbols, aliases)  # normalize gene names
                        else:
                            gname = gid
                        if self.genes[gid].chromosome is None:
                            self.genes[gid].set_location(loc)  # update
                            self.genes[gid].name=gname
                        else: # handle PAR region genes, i.e., same gid but different locations:
                            if not self.genes[gid].is_par:
                                self.genes[gid].is_par = True
                                self.genes[gid].par_regions = []
                            self.genes[gid].par_regions.append(loc)
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            setattr(self.genes[gid], field, info.get(field, None))
                        self.name2gene[gname] = self.genes[gid]
                    elif (info['feature_type'] == 'transcript') or \
                            ((annotation_flavour == 'flybase') and
                             (info['feature_type'] in ['mRNA', 'pre_miRNA', 'miRNA', 'ncRNA', 'pseudogene', 'rRNA',
                                                       'snRNA', 'snoRNA', 'tRNA'])):
                        if self.txfilter.filter(loc, info):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        tid = info['transcript_id']
                        if tid in self.transcripts:
                            # for tx with a single exon it is possible that exon entries
                            # occur prior to tx entries. To resolve this, we create 'proxy' tx objects that are
                            # then updated by the respective 'transcript' entry
                            tx = self.transcripts[tid]
                            tx.set_location(loc)  # update
                        else:
                            obj = Transcript(self.genes[gid], tid, loc.chromosome, loc.start, loc.end, loc.strand)
                            self.transcripts[tid] = obj
                            self.obj[chrom].add(obj)
                        if annotation_flavour == 'ucsc':
                            # UCSC gtf does not contain gene entries. So we first create a 'proxy' gene proxy object that
                            # is here updated with information from the respective transcriupt entry
                            gx = self.genes[gid]
                            gname = norm_gn(info.get('gene_name', 'NA'), current_symbols,
                                            aliases)  # normalize gene names
                            start = min(gx.start,
                                        loc.start) if gx.start else loc.start  # if multiple tx: calc min/max coords
                            end = max(gx.end, loc.end) if gx.start else loc.end
                            gx.set_location(loc)  # update
                            gx.name=gname
                            self.name2gene[gname] = gx
                            for field in get_config(self.config, 'copied_fields', default_value=[]):
                                setattr(gx, field, info.get(field, None))
                        elif annotation_flavour == 'flybase':
                            # flybase contains various transcript-like fetaure types: convert to transcript and set gene_type field
                            self.transcripts[tid].gene_type = info['feature_type']
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            setattr(self.transcripts[tid], field, info.get(field, None))
                        self.genes[gid].transcripts[tid] = self.transcripts[tid]
                    elif info['feature_type'] in ['exon', 'three_prime_UTR', '3UTR', 'five_prime_UTR', '5UTR']:
                        if self.txfilter.filter(loc, info):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        tid = info['transcript_id']
                        if tid not in self.transcripts:
                            obj = Transcript(self.genes[gid], tid, None, None, None, None)  # create proxy obj.
                            self.transcripts[tid] = obj
                            self.obj[chrom].add(obj)
                        if info['feature_type'] == 'exon':
                            feature = Exon(self.transcripts[tid], None, loc.chromosome, loc.start, loc.end, loc.strand)
                            self.transcripts[tid].exons.insert(0, feature) if loc.strand == '-' else self.transcripts[
                                tid].exons.append(feature)
                            self.obj[chrom].add(feature)
                        elif info['feature_type'] in ['three_prime_UTR', '3UTR']:
                            feature = Utr3(self.transcripts[tid], loc.chromosome, loc.start, loc.end, loc.strand)
                            self.transcripts[tid].utr3.insert(0, feature) if loc.strand == '-' else self.transcripts[
                                tid].utr3.append(feature)
                            self.obj[chrom].add(feature)
                        elif info['feature_type'] in ['five_prime_UTR', '5UTR']:
                            feature = Utr5(self.transcripts[tid], loc.chromosome, loc.start, loc.end, loc.strand)
                            self.transcripts[tid].utr5.insert(0, feature) if loc.strand == '-' else self.transcripts[
                                tid].utr5.append(feature)
                            self.obj[chrom].add(feature)
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            setattr(feature, field, info.get(field, None))
                elif annotation_flavour == 'mirgenedb':
                    if info['feature_type'] in ['pre_miRNA', 'miRNA']:
                        # add gene
                        gid, tid, gname, gene_type = info['ID'], info['ID'], info['Alias'], info['feature_type']
                        if self.txfilter.filter(loc, {'transcript_id': tid, 'gene_type': gene_type }):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        gname = norm_gn(gname, current_symbols, aliases)  # normalize gene names
                        obj = Gene(gid, gname, loc.chromosome, loc.start, loc.end, loc.strand, transcriptome=self)
                        self.genes[gid] = obj
                        self.obj[chrom].add(obj)
                        # add tx
                        if gname in self.name2gene:
                            print(f"WARN: duplicate gene names found: {gname}")
                        self.name2gene[gname] = self.genes[gid]
                        obj = Transcript(self.genes[gid], tid, loc.chromosome, loc.start, loc.end,
                                         loc.strand)
                        self.transcripts[tid] = obj
                        self.genes[gid].transcripts[tid] = self.transcripts[tid]
                        self.obj[chrom].add(obj)
                        for obj in [self.genes[gid], self.transcripts[tid]]:
                            for field in get_config(self.config, 'copied_fields', default_value=[]):
                                setattr(obj, field, info.get(field, None))

            # drop gene objs that were not resolved, probably due to tx filtering
            for unresolved_gid in [gid for gid in self.genes if self.genes[gid].chromosome is None]:
                print(f"dropping unresolved gene with id {unresolved_gid}")
                obj = self.genes.pop(unresolved_gid, None)
                self.obj[chrom].remove(obj)
            # set exon rnk
            for tx in self.transcripts.values():
                for rnk, ex in enumerate([ex for ex in tx.exons]):
                    ex.rnk = rnk + 1
            # add intron features
            introns=[]
            for tx in self.transcripts.values():
                strand = tx.strand
                ex = reversed(tx.exons) if strand == '-' else tx.exons
                for rnk, (ex0, ex1) in enumerate(pairwise(tx.exons)):
                    if strand == '-':
                        ex0, ex1 = ex1, ex0
                    loc = gi(tx.chromosome, ex0.end + 1, ex1.start - 1, strand)
                    intron = Intron(tx, rnk + 1, loc.chromosome, loc.start, loc.end, loc.strand)
                    # copy fields from exon
                    for field in get_config(self.config, 'copied_fields', default_value=[]):
                        setattr(intron, field, getattr(ex0, field))
                    tx.introns.append(intron)
                    introns.append(intron)
            self.obj[chrom].update(introns)
        if get_config(self.config, 'drop_empty_genes', default_value=False):
            self.drop_empty_genes()
        if get_config(self.config, 'load_sequences', default_value=False):
            self.load_sequences()

    def check_stats(self):
        """Debugging"""
        n = Counter()
        # count all obj
        all_obj = list(chain.from_iterable([self.obj[chrom] for chrom in self.merged_refdict]))
        n['all_obj'] = len(all_obj)
        n['all_obj_it'] = len(TranscriptomeIterator(self, None, None).take())
        # count linked genes/features
        features_linked = list()
        for g in self.genes.values():
            features_linked.append(g)
            for tx in g.transcripts.values():
                features_linked.append(tx)
                features_linked += tx.exons
                features_linked += tx.introns
                features_linked += tx.utr3
                features_linked += tx.utr5
        n['features_linked'] = len(features_linked)
        # count par genes
        n['par_genes'] = len([len(g.par_regions) + 1 for g in self.genes.values() if g.is_par])
        for g in self.genes.values():
            if g not in self.obj[g.chromosome]:
                print('linked gene but not in obj list', g.location(), g)
        extra_obj = [o for chrom in self.merged_refdict for o in self.obj[chrom]if not o in features_linked]
        extra_lnk = [o for o in features_linked if o not in self.obj[o.chromosome]]
        if (n['all_obj'] != n['all_obj_it']) or (n['all_obj'] != n['features_linked']):
            print(n, 'obj not found in linked obj list', len(extra_obj), 'linked obj not found in obj list', extra_lnk)
            return False
        return True

    def drop_empty_genes(self):
        """Drop all genes w/o associated transcripts"""
        for g in [g for g in self.genes.values() if len(g.transcripts)==0]:
            obj = self.genes.pop(g.gid, None)
            if obj.name in self.name2gene:
                self.name2gene.pop(obj.name)
        for chrom in self.obj:
            for g in [g for g in self.obj[chrom] if isinstance(g, Gene) and len(g.transcripts)==0]:
                self.obj[chrom].discard(g)
        if Gene in self.itrees: # drop itree
            del self.itrees[Gene]

    def annotate(self, feature_types, anno_its, field_names):
        """
            Annotates features of the passed type(s) using the passed annotation (per-position) iterators
        """
        TranscriptomeIterator(self, feature_types=feature_types).annotate(anno_its, field_names)

    def gene_triples(self, max_dist=None):
        """
            Convenience method that yields genes and their neighbouring (up-/downstream) genes.
            If max_dist is set and the neighbours are further away (or on other chromosomes),
            None is returned.

            To iterate over all neighbouring genes within a given genomic window, consider query()
            or implement a custom iterator.
        """
        for (x, y, z) in triplewise(chain([None], self.genes.values(), [None])):
            if max_dist is not None:
                dx = None if x is None else x.distance(y)
                if (dx is None) or (abs(dx) > max_dist):
                    x = None
                dz = None if z is None else z.distance(y)
                if (dz is None) or (abs(dz) > max_dist):
                    z = None
            yield x, y, z

    def build_itree(self, feature_class=Gene):
        """
            Returns a dict {chromosome:IntervalTree) with intervals created from features of the
            configured class.
        """
        itrees = {}
        if feature_class == Gene:
            data = self.genes.values()
        elif feature_class == Transcript:
            data = self.transcripts.values()
        elif feature_class == Exon:
            data = list(chain.from_iterable([tx.exons for tx in self.transcripts.values()]))
        elif feature_class == Intron:
            data = list(chain.from_iterable([tx.introns for tx in self.transcripts.values()]))
        elif feature_class == Utr3:
            data = list(chain.from_iterable([tx.utr3 for tx in self.transcripts.values()]))
        elif feature_class == Utr5:
            data = list(chain.from_iterable([tx.utr5 for tx in self.transcripts.values()]))
        else:
            print(f'Cannot build IntervalTree for class {feature_class.__name__}')
            return None
        for feat in tqdm(data, desc=f"Build {feature_class.__name__} interval tree", total=len(data)):
            if feat.chromosome not in itrees:
                itrees[feat.chromosome] = IntervalTree()
            # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
            # if Interval(feat.start, feat.end + 1) in itrees[feat.chromosome]:
            #     print("WARN: duplicates")
            itrees[feat.chromosome].addi(feat.start, feat.end + 1, feat)
        return itrees

    def query(self, query, feature_class=Gene, envelop=False):
        """
            Query features of the passed class at the passed query location.
            If the respective interval trees are not existing yet, it is built and can directly
            be accessed via <transcriptome>.itrees[feature_class][chromosome].

            if 'envelop' is set, then only features fully contained in the query
            interval are returned.
        """
        if feature_class not in self.itrees:
            self.itrees[feature_class] = self.build_itree(feature_class)
        if query.chromosome not in self.itrees[feature_class]:
            return []
        # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
        if envelop:
            return [x.data for x in self.itrees[feature_class][query.chromosome].envelop(query.start, query.end + 1)]
        else:
            return [x.data for x in self.itrees[feature_class][query.chromosome].overlap(query.start, query.end + 1)]

    def __len__(self):
        return len(self.genes)

    def __hash__(self):
        """ TODO: equality/hash based on genes/tx dict keys and hash_seq attribute only """
        return hash((
            frozenset(sorted(self.genes.keys())),
            frozenset(sorted(self.transcripts.keys())),
            self.has_seq
        ))

    def __eq__(self, other):
        return isinstance(other, __class__) and self.__hash__() == other.__hash__()

    def __repr__(self):
        return f"Transcriptome with {len(self.genes)} genes and {len(self.transcripts)} tx" + (
            " (+seq)" if self.has_seq else "") + (" (cached)" if self.cached else "")

    def get_gene(self, gene_name):
        """ Get gene by name"""
        return self.name2gene.get(gene_name, None)



    def load_sequences(self):
        """Loads feature sequences from a genome FASTA file"""
        genome_offsets = get_config(self.config, 'genome_offsets', default_value={})
        with pysam.Fastafile(get_config(self.config, 'genome_fa', required=True)) as fasta:
            for g in tqdm(self.genes.values(), desc='Load sequence', total=len(self.genes)):
                g.dna_seq = fasta.fetch(reference=g.chromosome,
                                        start=g.start - genome_offsets.get(g.chromosome, 0),
                                        end=g.end - genome_offsets.get(g.chromosome, 0) + 1)

    def to_gff3(self, out_file, bgzip=True, feature_types=[Gene, Transcript, Exon, Utr3, Utr5]):
        """
            Writes a GFF3 file with all features of the configured type (default: Gene, Transcript, Exon, Utr3, Utr5).
            The output file will be bgzipped and tabixed if bgzip is True.
            @see https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md

            Example:
                t.to_gff3('introns.gff3', feature_types=[Intron]) # creates a file introns.gff3.gz containing all intron annotations
        """

        def write_line(o, ftype, info, out):
            print("\t".join([str(x) for x in [
                o.chromosome,
                'pygenlib',
                ftype,
                o.start,  # start
                o.end,  # end
                to_str(o.score if hasattr(o, 'score') else None, na='.'),
                o.strand,
                to_str(o.phase if hasattr(o, 'phase') else None, na='.'),
                to_str([f'{k}={v}' for k, v in info.items()], sep=';')
            ]]), file=out)

        copied_fields = [x for x in get_config(self.config, 'copied_fields', default_value=[]) if
                         x not in ['score', 'phase']]
        with open(out_file, 'w') as out:
            for gid, g in self.genes.items():
                info = {'ID': gid, 'gene_id': gid, 'gene_name': g.name}
                info.update({k: getattr(g, k) for k in copied_fields})  # add copied fields
                if Gene in feature_types:
                    write_line(g, 'gene', info, out)
                for tid, t in g.transcripts.items():
                    info = {'ID': tid, 'gene_id': gid, 'transcript_id': tid, 'Parent': gid}
                    info.update({k: getattr(t, k) for k in copied_fields})  # add copied fields
                    if Transcript in feature_types:
                        write_line(t, 'transcript', info, out)
                    for f in [f for f in t.features() if f.__class__ in feature_types]:
                        class2ftype = {
                            Utr3: 'three_prime_UTR',
                            Utr5: 'five_prime_UTR',
                            Exon: 'exon',
                            Intron: 'intron'
                        }
                        ftype = class2ftype.get(f.__class__, str(f.__class__))
                        info = {'ID': tid, 'gene_id': gid, 'transcript_id': tid, 'Parent': tid}
                        if ftype in ['exon', 'intron']:
                            info['exon_number'] = to_str(f.rnk)
                        info.update({k: getattr(t, k) for k in copied_fields})  # add copied fields
                        write_line(f, ftype, info, out)
        if bgzip:
            bgzip_and_tabix(out_file)
            return out_file + '.gz'
        return out_file

    def save(self, out_file):
        """
            Stores this transcriptome as dill (pickle) object.
            Note that this can be slow for large-scaled transcriptomes,
            e.g., 25min for gencode.v39 with a 3GB output file.
            Loading is considerably faster.
        """
        print(f"Storing {self} to {out_file}")
        with open(out_file, 'wb') as out:
            dill.dump(self, out, recurse=True, byref=True)

    @classmethod
    def load(cls, in_file):
        """Load transcriptome from pickled file"""
        print(f"Loading transcriptome model from {in_file}")
        gc.disable()  # disable garbage collector
        with open(in_file, 'rb') as infile:
            obj = dill.load(infile)
        gc.enable()
        obj.cached = True
        print(f"Loaded {obj}")
        return obj


# --------------------------------------------------------------
# utility functions
# --------------------------------------------------------------

def calc_3end(tx, width=200):
    """
        Returns a list of genomic intervals containing the last <width> bases
        of the passed transcript or None if not possible
    """
    ret = []
    for ex in tx.exons[::-1]:
        if len(ex) < width:
            ret.append(ex.copy())
            width -= len(ex)
        else:
            s, e = (ex.start, ex.start + width - 1) if (ex.strand == '-') else (ex.end - width + 1, ex.end)
            ret.append(gi(ex.chromosome, s, e, ex.strand))
            width = 0
            break
    return ret if width == 0 else None
