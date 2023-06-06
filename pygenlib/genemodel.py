"""

Transcriptome model classes.

"""

from collections import Counter
from dataclasses import dataclass, make_dataclass, field
from itertools import chain, pairwise

import dill
import mygene
import pandas as pd
import pysam
from intervaltree import IntervalTree
from more_itertools import triplewise
from tqdm import tqdm

from pygenlib.iterators import GFF3Iterator
from pygenlib.utils import gi, reverse_complement, get_config, get_reference_dict, open_file_obj, ReferenceDict, to_str, \
    bgzip_and_tabix, toggle_chr


# ------------------------------------------------------------------------
# gene symbol abstraction
# ------------------------------------------------------------------------

@dataclass(frozen=True)
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
@dataclass(frozen=True, repr=False)
class Feature(gi):
    """ A (frozen) genomic feature"""
    feature_type: str = None  # a feature type (e.g., exon, intron, etc.)
    parent: object = field(default=None)  # an optional parent
    subfeature_types: tuple = tuple()  # sub-feature types

    def __repr__(self) -> str:
        return f"{self.feature_type}@{self.chromosome}:{self.start}-{self.end}"

    @classmethod
    def from_gi(cls, loc, ):
        """ Init from gi """
        return cls(loc.chromosome, loc.start, loc.end, loc.strand)

    def location(self):
        """Returns a genomic interval representing the genomic location of this feature."""
        return gi(self.chromosome, self.start, self.end, self.strand)

    def rnk(self):
        """Rank (1-based index) of feature in this feature's parent children list"""
        if not self.parent:
            return None
        return self.parent.__dict__[self.feature_type].index(self) + 1

    def features(self, feature_types=None):
        """ Iterates over all sub-features (no sorted)"""
        for ft in self.subfeature_types:
            for f in self.__dict__[ft]:
                if (not feature_types) or (f.feature_type in feature_types):
                    yield f
                for sf in f.features():  # recursion
                    if (not feature_types) or (sf.feature_type in feature_types):
                        yield sf

    # dynamic feature class creation
    @classmethod
    def create_sub_class(cls, feature_type, annotations: dict = None, child_feature_types: list = None):
        """ Create a subclass of feature with additional fields (as defined in the annotations dict)
            and child tuples
        """
        fields = [('feature_type', str, field(default=feature_type))]
        fields += [(k, v, field(default=None)) for k, v in annotations.items()]
        if child_feature_types is not None:
            fields += [(k, tuple, field(default=tuple(), hash=False, compare=False)) for k in child_feature_types]
        sub_class=make_dataclass(feature_type, fields=fields, bases=(cls,), frozen=True, repr=False)
        return sub_class

# Gene = Feature.create_sub_class('gene', {'name': str}, ['transcripts'])
# g=Gene('chr1', 1, 100)
# print(id(g))
# object.__setattr__(g, 'transcripts', (1,2,3))
# print(id(g))

class _mFeature():
    """
        A mutable genomic (annotation) feature used for building only
    """

    def __init__(self, ftype, loc=None, parent=None, children={}):
        self.loc = loc
        self.ftype = ftype
        self.parent = parent
        if self.parent is not None:
            # assert self.loc.strand == self.parent.loc.strand
            # assert parent envelops (contains) child
            if self.parent.children is None:
                self.parent.children = {}
            if self.ftype not in self.parent.children:
                self.parent.children[self.ftype] = list()
            self.parent.children[self.ftype].append(self)
        self.children = children
        self.anno = {}

    def get_anno_rec(self):
        """compiles a dict containing all annotations of this feature and all its children per feature_type"""
        a = {self.ftype: {k: type(v) for k, v in self.anno.items()}}
        t = {self.ftype: set()}
        s = {self.ftype}
        if self.children:
            for cat in self.children:
                t[self.ftype].add(cat)
                s.add(cat)
                for c in self.children[cat]:
                    x, y, z = c.get_anno_rec()
                    a.update(x)
                    t.update(y)
                    s.update(z)
        return a, t, s

    def set_location(self, loc):
        self.loc = loc

    def __repr__(self):
        return f"{self.ftype}@{super().__repr__()} ({ {k: len(v) for k, v in self.children.items()} if self.children else 'NA'})"

    def freeze(self, ft2class):
        """Create a frozen instance (recursively)"""
        # print(f"Freeze {self}")
        f = ft2class[self.ftype].from_gi(self.loc)
        for k, v in self.anno.items():
            object.__setattr__(f, k, v)
        if self.children:
            object.__setattr__(f, 'subfeature_types', tuple(self.children))
            for k in self.children:
                children = [x.freeze(ft2class) for x in self.children[k]]
                if self.loc.strand == '-':  # reverse order if on neg strand
                    children = list(reversed(children))
                for c in children:
                    object.__setattr__(c, 'parent', f)
                object.__setattr__(f, k, tuple(children))
        return f


# g = mFeature( 'gene', gi('chr1',1,200,'-'), parent=None, children={'transcript': []})
# tx = mFeature( 'transcript', gi('chr1',1,200,'-'), parent=g, children={'exon': [], 'intron': []})
# ex1 = mFeature( 'exon', gi('chr1',1,20,'-'), parent=tx, children=None)
# ex2 = mFeature( 'exon', gi('chr1',50,200,'-'), parent=tx, children=None)
# ex1.anno['gene_type']='snoRNA'
# g.freeze()


class Transcriptome:
    """
        Represents a transcriptome as modelled by a GTF/GFF file:
        - Model contains genes, transcripts and arbitrary features (e.g., exons, intron, 3'/5'-UTRs, CDS) as defined
            in the GFF file
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
            [g.name for g in t.genes]
        - Get number of exons for all ACTB transcripts
            [len(tx.exon) for tx in t.gene_name['ACTB'].transcript]
        - Gene name of tx ENST00000414620.1
            t.transcript['ENST00000414620.1'].parent.name
        - Get all transcript ids where exons overlap chr7:5529026
            {e.parent.transcript_id for e in t.query(gi('chr7', 5529026, 5529026), feature_types=['exon'])}
        - Get all gene names where a kmer ("ACTGACTGACTG") if expressed in exons (requires load_sequences() )
            { tx.parent.name for tx in t.transcripts if "ACTGACTGACTG" in t.get_spliced_seq(tx) }
        - list of genes and their up- and downstream genes if within a given distance:
            [(prv, gene, nxt) for prv, gene, nxt in t.gene_triples(max_dist=10000)]
        - coordinate-sorted list of genes in a 10kb window around ACTB:
            g=t.gene_name['ACTB']
            list(sorted(t.query(gi(g.chromosome, g.start-10000, g.end+10000), feature_types=['gene'])))
        - dict of genes and their unique 200bp 3'UTR intervals per tx (multiple intervals if spliced)
                {g.name: {tx.transcript_id:calc_3end(tx) for tx in g.transcript} for g in t.genes}

        IntervalTree based examples
        -------------------------------

    """

    def __init__(self, config):
        self.config = config
        self.log = Counter()
        self.txfilter = TranscriptFilter(self.config)
        self.merged_refdict = None
        self.gene = {}  # gid: gene
        self.transcript = {}  # tid: gene
        self.gene_name = {}  # gene_name : gene
        self.cached = False  # if true then transcriptome was loaded from a pickled file
        self.has_seq = False
        self.anno = {}
        self.chr2itree = {}
        self.build()

    def build(self):
        """
            Builds a transcriptome model from the configured GTF/GFF file.

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
        genes = {}
        transcripts = {}
        anno = {}
        for chrom in tqdm(self.merged_refdict, f"Building transcriptome ({self.txfilter})"):
            for loc, info in GFF3Iterator(get_config(self.config, 'annotation_gff', required=True), chrom,
                                          fun_alias=annotation_fun_alias):
                self.log['parsed_gff_lines'] += 1
                if annotation_flavour in ['gencode', 'ensembl', 'ucsc', 'flybase']:
                    gid = info['gene_id']  # mandatory gtf info tag
                    if gid not in genes:
                        # UCSC gtf does not contain gene entries and flybase is not sorted hierarchically.
                        # So we first create a 'proxy' gene proxy object that
                        # will later be updated with tx coordinates
                        genes[gid] = _mFeature('gene', None, parent=None, children={})
                        genes[gid].anno['gene_id'] = gid
                    # ---------------------------- genes -----------------------------------
                    if info['feature_type'] == 'gene':
                        if 'gene_name' in info:  # default for genocde/ucsc
                            gname = norm_gn(info['gene_name'], current_symbols, aliases)  # normalize gene names
                        elif 'gene_symbol' in info:  # default for flybase
                            gname = norm_gn(info['gene_symbol'], current_symbols, aliases)  # normalize gene names
                        else:
                            gname = gid
                        if genes[gid].loc is None:
                            genes[gid].set_location(loc)  # update
                            genes[gid].anno['name'] = gname
                        else:  # handle PAR region genes, i.e., same gid but different locations. Copy to par_regions
                            if 'par_regions' not in genes[gid].anno:
                                genes[gid].anno['par_regions'] = []
                            genes[gid].anno['par_regions'].append(loc)
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            genes[gid].anno[field] = info.get(field, None)
                    # ---------------------------- transcripts -----------------------------------
                    elif (info['feature_type'] == 'transcript') or \
                            ((annotation_flavour == 'flybase') and
                             (info['feature_type'] in ['mRNA', 'pre_miRNA', 'miRNA', 'ncRNA', 'pseudogene', 'rRNA',
                                                       'snRNA', 'snoRNA', 'tRNA'])):
                        if self.txfilter.filter(loc, info):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        tid = info['transcript_id']
                        transcript_type = info['feature_type'] if annotation_flavour == 'flybase' else None
                        if tid in transcripts:
                            # for tx with a single exon it is possible that exon entries
                            # occur prior to tx entries. To resolve this, we create 'proxy' tx objects that are
                            # then updated by the respective 'transcript' entry
                            tx = transcripts[tid]
                            tx.set_location(loc)  # update location
                            transcripts[tid].anno['transcript_type'] = transcript_type
                        else:
                            transcripts[tid] = _mFeature('transcript', loc, parent=genes[gid],
                                                         children={'exon': [], 'intron': [], 'three_prime_UTR': [],
                                                                   'five_prime_UTR': []})
                            transcripts[tid].anno['transcript_id'] = tid
                            transcripts[tid].anno['transcript_type'] = transcript_type
                        if annotation_flavour == 'ucsc':
                            # UCSC gtf does not contain gene entries. So we first create a 'proxy' gene proxy object that
                            # is here updated with information from the respective transcriupt entry
                            gx = genes[gid]
                            gname = norm_gn(info.get('gene_name', 'NA'), current_symbols,
                                            aliases)  # normalize gene names
                            start = min(gx.loc.start,
                                        loc.start) if gx.loc else loc.start  # if multiple tx: calc min/max coords
                            end = max(gx.loc.end, loc.end) if gx.loc else loc.end
                            gx.set_location(loc)  # update locations
                            gx.anno['name'] = gname
                            for field in get_config(self.config, 'copied_fields', default_value=[]):
                                gx.anno[field] = info.get(field, None)
                        # copy fields
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            transcripts[tid].anno[field] = info.get(field, None)
                    # ---------------------------- sub-features -----------------------------------
                    elif info['feature_type'] in ['exon', 'three_prime_UTR', '3UTR', 'five_prime_UTR', '5UTR', 'CDS']:
                        if self.txfilter.filter(loc, info):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        tid = info['transcript_id']
                        feature_type = info['feature_type']
                        map_ftypes = {
                            '3UTR': 'three_prime_UTR',
                            '5UTR': 'five_prime_UTR'
                        }
                        feature_type = map_ftypes.get(feature_type, feature_type)
                        if tid not in transcripts:
                            transcripts[tid] = _mFeature('transcript', None, parent=genes[gid],
                                                         children={'exon': [], 'intron': [], 'three_prime_UTR': [],
                                                                   'five_prime_UTR': []})  # create proxy obj.
                            transcripts[tid].anno['transcript_id'] = tid
                        feature = _mFeature(feature_type, loc, parent=transcripts[tid], children=None)
                        for field in get_config(self.config, 'copied_fields', default_value=[]):
                            feature.anno[field] = info.get(field, None)
                elif annotation_flavour == 'mirgenedb':
                    if info['feature_type'] in ['pre_miRNA', 'miRNA']:
                        # add gene
                        gid, tid, gname, gene_type = info['ID'], info['ID'], info['Alias'], info['feature_type']
                        if self.txfilter.filter(loc, {'transcript_id': tid, 'gene_type': gene_type}):
                            self.log[f"filtered_{info['feature_type']}"] += 1
                            continue
                        gname = norm_gn(gname, current_symbols, aliases)  # normalize gene names
                        genes[gid] = _mFeature('gene', loc, parent=None, children={})
                        genes[gid].anno['gene_id'] = gid
                        genes[gid].anno['name'] = gname
                        # add tx
                        transcripts[tid] = _mFeature('transcript', loc, parent=genes[gid])
                        transcripts[tid].anno['transcript_id'] = tid
                        for obj in [genes[gid], transcripts[tid]]:
                            for field in get_config(self.config, 'copied_fields', default_value=[]):
                                setattr(obj, field, info.get(field, None))
            # drop gene objs that were not resolved, probably due to tx filtering
            for unresolved_gid in [gid for gid in genes if genes[gid].loc is None]:
                self.log['dropped_unresolved_genes']+=1
                genes.pop(unresolved_gid, None)
            # add intron features
            for tx in transcripts.values():
                if (not 'exon' in tx.children) or (len(tx.children['exon'])<=1):
                    continue
                strand = tx.loc.strand
                for rnk, (ex0, ex1) in enumerate(pairwise(tx.children['exon'])):
                    loc = gi(tx.loc.chromosome, ex0.loc.end + 1, ex1.loc.start - 1, strand)
                    intron = _mFeature('intron', loc, parent=tx, children=None)
                    # copy fields from previous exon
                    intron.anno = ex0.anno.copy()
        # drop genes w/o transcripts (e.g., after filtering)
        for k in [k for k,v in genes.items() if len(v.children)==0]:
            self.log['dropped_empty_genes'] += 1
            obj = genes.pop(k, None)
        # step1: create custom dataclasses
        ft2anno_class = {}
        ft2child_ftype = {}
        fts = set()
        for g in genes.values():
            a, t, s = g.get_anno_rec()
            ft2anno_class.update(a)
            ft2child_ftype.update(t)
            fts.update(s)
        ft2class = {
            ft: Feature.create_sub_class(ft, ft2anno_class.get(ft, {}), ft2child_ftype.get(ft, [])) for ft in fts
        }
        # step2: freeze and add to auxiliary data structures
        self.genes = [g.freeze(ft2class) for g in genes.values()]
        all_features = list()
        for g in self.genes:
            all_features.append(g)
            for f in g.features():
                all_features.append(f)
        all_features.sort(key=lambda x: (self.merged_refdict.index(x.chromosome), x))
        self.anno = {f: {} for f in all_features}
        # build some auxiliary dicts
        self.gene = {f.gene_id: f for f in self.__iter__(feature_types=['gene'])}
        self.gene_name = {f.name: f for f in self.__iter__(feature_types=['gene'])}
        self.transcript = {f.transcript_id: f for f in self.__iter__(feature_types=['transcript'])}
        self.transcripts = list(self.__iter__(feature_types=['transcript']))
        # load sequences
        if get_config(self.config, 'load_sequences', default_value=False):
            self.load_sequences()
        # build itree
        for g in tqdm(self.genes, desc=f"Build interval tree", total=len(self.genes)):
            if g.chromosome not in self.chr2itree:
                self.chr2itree[g.chromosome] = IntervalTree()
            # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
            self.chr2itree[g.chromosome].addi(g.start, g.end + 1, g)

    def load_sequences(self):
        """Loads feature sequences from a genome FASTA file"""
        genome_offsets = get_config(self.config, 'genome_offsets', default_value={})
        with pysam.Fastafile(get_config(self.config, 'genome_fa', required=True)) as fasta:
            for g in tqdm(self.genes, desc='Load sequence', total=len(self.genes)):
                self.anno[g]['dna_seq'] = fasta.fetch(reference=g.chromosome,
                                                      start=g.start - genome_offsets.get(g.chromosome, 0),
                                                      end=g.end - genome_offsets.get(g.chromosome, 0) + 1)

    def find_attr_rec(self, f, attr):
        """ recursively finds attribute from parent(s) """
        if f is None:
            return None, None
        if attr in self.anno[f]:
            return f, self.anno[f][attr]
        return self.find_attr_rec(f.parent, attr)

    def get_sequence(self, f, mode='dna'):
        """
            Returns the 5'-3' DNA sequence (as shown in a genome browser) of this feature.
            If mode is rna then the reverse complement of negative-strand features will be returned.
        """
        p, pseq = self.find_attr_rec(f, 'dna_seq')
        if p is None:
            return None
        if p == f:
            seq = pseq
        else:
            idx = f.start - p.start
            seq = pseq[idx:idx + len(f)]  # slice from parent sequence
        if (seq is not None) and (mode == 'rna') and (f.strand == '-'):  # revcomp if rna mode and - strand
            seq = reverse_complement(seq)
        return seq

    def get_spliced_seq(self, f, show_exon_boundaries=False):
        """The fully spliced sequence, always in 'rna' mode and valid only for containers of exons."""
        assert 'exon' in f.subfeature_types, "Can only splice features that have annotated exons"
        sep='*' if show_exon_boundaries else ''
        fseq = self.get_sequence(f, mode='dna')
        if f.strand == '-':
            fseq_spliced = reverse_complement(
                sep.join([fseq[(ex.start - f.start):(ex.start - f.start) + len(ex)] for ex in reversed(f.exon)]))
        else:
            fseq_spliced = sep.join([fseq[(ex.start - f.start):(ex.start - f.start) + len(ex)] for ex in f.exon])
        return fseq_spliced

    def gene_triples(self, max_dist=None):
        """
            Convenience method that yields genes and their neighbouring (up-/downstream) genes.
            If max_dist is set and the neighbours are further away (or on other chromosomes),
            None is returned.

            To iterate over all neighbouring genes within a given genomic window, consider query()
            or implement a custom iterator.
        """
        for (x, y, z) in triplewise(chain([None], self.genes, [None])):
            if max_dist is not None:
                dx = None if x is None else x.distance(y)
                if (dx is None) or (abs(dx) > max_dist):
                    x = None
                dz = None if z is None else z.distance(y)
                if (dz is None) or (abs(dz) > max_dist):
                    z = None
            yield x, y, z

    def query(self, query, feature_types=None, envelop=False, sorted=True):
        """
            Query features of the passed class at the passed query location.
            If the respective interval trees are not existing yet, it is built and can directly
            be accessed via <transcriptome>.itrees[feature_class][chromosome].

            if 'envelop' is set, then only features fully contained in the query
            interval are returned.
        """
        if query.chromosome not in self.chr2itree:
            return []
        if isinstance(feature_types, str):
            feature_types=(feature_types,)
        # add 1 to end coordinate, see itree conventions @ https://github.com/chaimleib/intervaltree
        overlapping_genes = [x.data for x in self.chr2itree[query.chromosome].overlap(query.start, query.end + 1)]
        overlapping_features = overlapping_genes if (feature_types is None) or ('gene' in feature_types) else []
        if envelop:
            overlapping_features = [g for g in overlapping_features if query.envelop(g)]
        for g in overlapping_genes:
            if envelop:
                overlapping_features += [f for f in g.features(feature_types) if query.envelop(f)]
            else:
                overlapping_features += [f for f in g.features(feature_types) if query.overlaps(f)]
        if sorted:
            overlapping_features.sort(key=lambda x: (self.merged_refdict.index(x.chromosome), x))
        return overlapping_features

    def save(self, out_file):
        """
            Stores this transcriptome as dill (pickle) object.
            Note that this can be slow for large-scaled transcriptomes,
            e.g., 25min for gencode.v39 with a 3GB output file.
            Loading is considerably faster.
        """
        print(f"Storing {self} to {out_file}")
        with open(out_file, 'wb') as out:
            dill.dump(self, out, recurse=True)
            #, byref=True,  ) byref is broken as dynamically created dataclasses not supported

    @classmethod
    def load(cls, in_file):
        """Load transcriptome from pickled file"""
        print(f"Loading transcriptome model from {in_file}")
        import gc
        gc.disable()  # disable garbage collector
        with open(in_file, 'rb') as infile:
            obj = dill.load(infile)
        gc.enable()
        obj.cached = True
        print(f"Loaded {obj}")
        return obj

    def to_gff3(self, out_file, bgzip=True, feature_types=['gene', 'transcript', 'exon', 'three_prime_UTR', 'five_prime_UTR']):
        """
            Writes a GFF3 file with all features of the configured type (default: Gene, Transcript, Exon, Utr3, Utr5).
            The output file will be bgzipped and tabixed if bgzip is True.
            @see https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md

            Example:
                t.to_gff3('introns.gff3', feature_types=['intron']) # creates a file introns.gff3.gz containing all intron annotations
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
            for f in self.__iter__(feature_types):
                if f.feature_type == 'gene':
                    info = {'ID': f.gene_id, 'gene_id': f.gene_id, 'gene_name': f.name}
                elif f.feature_type == 'transcript':
                        info = {'ID': f.transcript_id, 'transcript_id': f.transcript_id,
                                'gene_id': f.parent.gene_id, 'Parent': f.parent.gene_id}
                else:
                    info={'gene_id': f.parent.parent.gene_id, 'transcript_id': f.parent.transcript_id}
                info.update({k: getattr(f, k) for k in copied_fields})  # add copied fields
                write_line(f, f.feature_type, info, out)
        if bgzip:
            bgzip_and_tabix(out_file)
            return out_file + '.gz'
        return out_file

    def __len__(self):
        return len(self.anno)

    def __repr__(self):
        return f"Transcriptome with {len(self.gene)} genes and {len(self.transcript)} tx" + (
            " (+seq)" if self.has_seq else "") + (" (cached)" if self.cached else "")

    def __iter__(self, feature_types=None):
        for f in self.anno.keys():
            if (not feature_types) or (f.feature_type in feature_types):
                yield f

# --------------------------------------------------------------
# utility functions
# --------------------------------------------------------------

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


        NOTE that gene objects that have no associated transcript left after filtering will be dropped .

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


def calc_3end(tx, width=200):
    """
        Returns a list of genomic intervals containing the last <width> bases
        of the passed transcript or None if not possible
    """
    ret = []
    for ex in tx.exon[::-1]:
        if len(ex) < width:
            ret.append(ex.location())
            width -= len(ex)
        else:
            s, e = (ex.start, ex.start + width - 1) if (ex.strand == '-') else (ex.end - width + 1, ex.end)
            ret.append(gi(ex.chromosome, s, e, ex.strand))
            width = 0
            break
    return ret if width == 0 else None
