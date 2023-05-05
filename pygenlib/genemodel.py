"""

A gene (annotation) model.

Usage examples:


"""

from pygenlib.utils import get_config, parse_gff_info, reverse_complement, kmer_search, split_list, find_all
from enum import Enum
from collections import Counter
from dataclasses import dataclass
import pandas as pd
import dill
import pysam
from tqdm import tqdm
import gc
import mygene
import itertools
import numpy as np
from multiprocessing import Pool, Process, Queue
import psutil
import os
import re
from more_itertools import interleave

@dataclass
class gene_symbol:
    """
        Class for representing a gene symbol.
    """
    symbol: str  #
    name: str  #
    taxid: int  #
    def __repr__(self):
        return f"{self.symbol} ({self.name}, tax: {self.taxid})"

@dataclass
class loc_obj:
    """
        Class for representing a genomic location with an optional strand.

        NOTE be careful with stranded data. Comparisons of features that are on different strands or comparisons of
            unstranded and stranded features always return None.
    """
    chromosome: str  #
    start: int  #
    end: int  #
    strand: str = None
    @classmethod
    def from_str(cls, loc_string):
        """ Parse from chr:start-end. start+end must be >=0 """
        chromosome, start, end = re.split(':|-', loc_string)
        return cls(chromosome, int(start), int(end))

    def __repr__(self):
        return f"{self.chromosome}:{self.start}-{self.end} ({'u' if self.strand is None else self.strand})"

    def to_file_str(self):
        """ returns a string "<chrom>_<start>_<end>_<strand>"        """
        f"{self.chromosome}_{self.start}_{self.end}_{'u' if self.strand is None else self.strand}"

    def __len__(self):
        return self.end-self.start+1

    def __hash__(self):
        return hash(self.__repr__())

    def is_stranded(self):
        return self.strand is not None

    def cs_match(self, other):
        """ True if this location is on the same chrom/strand as the passed one """
        return (self.chromosome == other.chromosome) and (self.strand == other.strand)

    def __eq__(self, other):
        """
            Test whether this interval is equal to the other.
            Includes a chromosome and strand check.
        """
        if not self.cs_match(other):
            return False
        return (self.start == other.start) and (self.end==other.end)

    def __cmp__(self, other, cmp_str):
        if not self.cs_match(other):
            return None
        if self.start != other.start:
            return getattr(self.start, cmp_str)(other.start)
        return getattr(self.end, cmp_str)(other.end)

    def __lt__(self, other):
        """
            Test whether this interval is smaller than the other.
            Defined only on same chromosome/strand. If those do not match, None is returned.
        """
        return self.__cmp__(other, '__lt__')

    def __le__(self, other):
        """
            Test whether this interval is smaller or equal than the other.
            Defined only on same chromosome/strand. If those do not match, None is returned.
        """
        return self.__cmp__(other, '__le__')

    def __gt__(self, other):
        """
            Test whether this interval is greater than the other.
            Defined only on same chromosome/strand. If those do not match, None is returned.
        """
        return self.__cmp__(other, '__gt__')

    def __ge__(self, other):
        """
            Test whether this interval is greater or equal than the other.
            Defined only on same chromosome/strand. If those do not match, None is returned.
        """
        return self.__cmp__(other, '__ge__')

    def left_match(self, other):
        if not self.cs_match(other):
            return False
        return self.start == other.start

    def right_match(self, other):
        if not self.cs_match(other):
            return False
        return self.end == other.end

    def left_pos(self):
        return loc_obj(self.chromosome, self.start, self.start, strand=self.strand)

    def right_pos(self):
        return loc_obj(self.chromosome, self.end, self.end, strand=self.strand)

    def overlaps(self, other, strand_specific=True):
        if (strand_specific) and (not self.cs_match(other)):
            return False
        return self.start <= other.end and other.start <= self.end

    def split_coordinates(self) -> (str,int,int):
        return self.chromosome, self.start, self.end

    @classmethod
    def merge(cls, l):
        """ Merges a list of intervals.
            If intervals are not on the same chromosome or if strand is not matching, None is returned
        """
        if l is None:
            return None
        if len(l) == 1:
            return l[0]
        merged = None
        for x in l:
            if merged is None:
                merged = x
            else:
                if not merged.cs_match(x):
                    return None
                merged.start = min(merged.start, x.start)
                merged.end = max(merged.end, x.end)
        return merged

    def is_adjacent(self, other):
        """ true if intervals are directly next to each other (not overlapping!) """
        if not self.cs_match(other):
            return False
        a, b = (self.end + 1, other.start) if self.end < other.end else (other.end + 1, self.start)
        return a == b

    def split_by_maxwidth(self, maxwidth):
        """ Splits this into n intervals of maximum width """
        k, m = divmod(self.end - self.start + 1, maxwidth)
        ret = [
            loc_obj(self.chromosome, self.start + i * maxwidth, self.start + (i + 1) * maxwidth - 1, strand=self.strand)
            for i in range(k)]
        if m > 0:
            ret += [loc_obj(self.chromosome, self.start + k * maxwidth, self.end, strand=self.strand)]
        return ret

    def copy(self):
        """ Deep copy """
        return loc_obj(self.chromosome, self.start, self.end, self.strand)


@dataclass
class gene_obj:
    """Class for representing a gene."""
    gid: str  #
    location: loc_obj #
    gene_symbol: str  #
    gene_type: str  #
    transcripts: dict
    def __repr__(self):
        return f"{self.gid} ({len(self.transcripts) if self.transcripts is not None else 0} tx)"
    def __len__(self):
        return len(self.location)
    def __hash__(self):
        return self.gid.__hash__()

@dataclass
class transcript_obj:
    """Class for representing a gene."""
    tid: str  #
    location: loc_obj #
    gene: gene_obj
    exons: list
    introns: list
    utr3: list
    utr5: list
    def __repr__(self):
        return f"{self.tid} ({len(self.exons) if self.exons is not None else 0} ex)"
    def __len__(self):
        return len(self.location)
    def get_features(self):
        return self.utr5+interleave(self.exons, self.introns)+self.utr3 # utr5, ex1, in1, ..., exn, utr3
    def __hash__(self):
        return self.tid.__hash__()

class feature_type(Enum):
    """Supported feature types"""
    exon=0
    intron=1
    utr3=3
    utr5=5

@dataclass
class transcript_feature:
    """Class for representing a exon/intron/etc."""
    type: feature_type # exon,intron,utr3,utr5
    location: loc_obj #
    transcript: transcript_obj  #
    rnk: int # exon/utr or intron number in tx
    def __len__(self):
        return f"{self.type} ({len(self.location)}bp)"

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
    current_symbols=set()
    if gene_name_alias_file:
        tab=pd.read_csv(gene_name_alias_file, sep='\t', dtype={'alias_symbol':str, 'prev_symbol':str, 'symbol':str}, low_memory=False, keep_default_na=False)
        for _,r in tqdm(tab.iterrows(), desc='load gene aliases', total=tab.shape[0]):
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
    g=g.strip() # remove whitespace
    if (current_symbols is not None) and (g in current_symbols):
        return g # there is a current symbol so don't look for aliases
    if aliases is None:
        return g
    return g if aliases is None else aliases.get(g,g)

class transcriptome:
    def __init__(self, config):
        self.config=config
        self.genes, self.transcripts={},{}
        self.log = Counter()
        self.cached=False # if true then transcriptome as loaded from a pickled file
        self.has_seq=False
        self.has_quantseq = False
        self.build()
    def __repr__(self):
        return f"Transcriptome with {len(self.genes)} genes and {len(self.transcripts)} tx"+(" (+seq)" if self.has_seq else "")+(" (cached)" if self.cached else "")
    def build(self):
        """
            Builds a transcriptome model

            mandatory config properties:
                genome_fa: FASTA of reference genome
                gene_gff: GFF/GTF file with gene model (currently supported: GENCODE)

            optional config properties:
                mandatory_tags: optional list of mandatory tag values. Use, e.g., ['Ensembl_canonical'] for canonical
                    tx or ['basic'] for GENCODE basic entries only. default:None
                basic_only: boolean, if set only tx with 'basic' tags are considered. default:false

            Transcript sequences can be added via load_sequences(). Sequences are stranded (i.e., genomic sequences is
            reverse-complemented for minus-strand transcripts) but DNA alphabet (ACTG) is used to enable direct
            alignment/comparison with genomic seqeunces.
        """
        # config properties
        mandatory_tags=get_config(self.config,'mandatory_tags',default_value=None)
        if mandatory_tags:
            mandatory_tags=set(mandatory_tags.split(',')) if isinstance(mandatory_tags, str) else set(mandatory_tags)
        # read gene aliases (optional)
        aliases, current_symbols = (None,None) if get_config(self.config, 'gene_name_alias_file', default_value=None) is None else read_alias_file(get_config(self.config, 'gene_name_alias_file', default_value=None))
        # load gff
        f = pysam.TabixFile(get_config(self.config,'gene_gff',required=True), mode="r")
        # get valid references from genome
        valid_chroms = pysam.Fastafile(get_config(self.config, 'genome_fa', required=True)).references
        for row in tqdm(f.fetch(parser=pysam.asTuple()), "Loading gene model"):
            reference, source,ftype,fstart,fend,score,strand,phase,info=row
            location = loc_obj(reference, int(fstart), int(fend), strand)
            if reference not in valid_chroms:
                self.log['gff_ref_not_in_genome']+=1
                continue # skip if not in ref seq
            pinfo=parse_gff_info( info )
            # check for mandatory tags or skip if not found
            if mandatory_tags is not None:
                if 'tag' not in pinfo:
                    self.log['non_tag_entries_skipped'] += 1
                    continue
                tags=set(pinfo['tag'].split(','))
                missing=mandatory_tags.difference(tags)
                if len(missing)>0:
                    self.log['entries_with_missing_tags_skipped'] += 1
                    continue
            # create gene entries
            gid = pinfo['gene_id']
            if ftype=='gene':
                # if gid == 'ENSG00000243485.5':
                #     break
                gname = norm_gn(pinfo['gene_name'], current_symbols, aliases) # normalize gene names
                self.genes[gid]=gene_obj(gid, location, gname, pinfo['gene_type'], {})
            # create new transcript entries
            elif ftype=='transcript':
                tid=pinfo['transcript_id']
                self.transcripts[tid]=transcript_obj(tid,location,self.genes[pinfo['gene_id']], [],  [], [], [])
                self.genes[gid].transcripts[tid]=self.transcripts[tid]
            elif ftype == 'exon':
                tid=pinfo['transcript_id']
                feature=transcript_feature(feature_type.exon,location,self.transcripts[tid],int(pinfo.get('exon_number',None)))
                if location.strand=='-':
                    self.transcripts[tid].exons.insert(0,feature)
                else:
                    self.transcripts[tid].exons.append(feature)
            elif ftype =='three_prime_UTR':
                tid = pinfo['transcript_id']
                feature=transcript_feature(feature_type.utr3, location, self.transcripts[tid], int(pinfo.get('exon_number', None)))
                if location.strand=='-':
                    self.transcripts[tid].utr3.insert(0,feature)
                else:
                    self.transcripts[tid].utr3.append(feature)
            elif ftype == 'five_prime_UTR':
                tid = pinfo['transcript_id']
                feature=transcript_feature(feature_type.utr5, location, self.transcripts[tid], int(pinfo.get('exon_number', None)))
                if location.strand=='-':
                    self.transcripts[tid].utr5.insert(0,feature)
                else:
                    self.transcripts[tid].utr5.append(feature)
        # add intron features
        for tx in self.transcripts.values():
            for i in range(0, len(tx.exons)-1):
                strand=tx.location.strand
                ex0,ex1=(tx.exons[i+1],tx.exons[i]) if strand=='-' else (tx.exons[i],tx.exons[i+1])
                location=loc_obj(tx.location.chromosome, ex0.location.end + 1, ex1.location.start - 1, strand)
                feature = transcript_feature(feature_type.intron, location, tx, ex0.rnk)
                if strand=='-':
                    tx.introns.insert(0,feature)
                else:
                    tx.introns.append(feature)
    def fix_strand(self, seq, strand):
        if isinstance(seq, list):
            return list(reversed(seq)) if strand=='-' else seq
        elif isinstance(seq, str):
            return reverse_complement(seq) if strand=='-' else seq
    def load_sequences(self):
        """Adds transcript sequences from a genome sequence"""
        with pysam.Fastafile(get_config(self.config, 'genome_fa', required=True)) as fasta:
            for tx in tqdm(self.transcripts.values(), "Loading tx sequences"):
                tx.sequence=""
                tx.sequence_pos = [] # genomic positions of tx bases (in 5'-3' order)
                tx.sequence_spl = [] # 1 if tx base at this position is 5' of SJ
                for ex in tx.exons:
                    ex_sequence=self.fix_strand(fasta.fetch(reference=ex.location.chromosome, start=ex.location.start - 1, end=ex.location.end), tx.location.strand)
                    tx.sequence+=ex_sequence
                    tx.sequence_pos+=self.fix_strand(list(range(ex.location.start, ex.location.end + 1)),tx.location.strand)
                    tx.sequence_spl+=[0]*(ex.location.end-ex.location.start)+[1]
                if get_config(self.config,'load_intron_sequences',default_value=False):
                    for intron in tx.introns:
                        intron.sequence=self.fix_strand(fasta.fetch(reference=intron.location.chromosome, start=intron.location.start - 1, end=intron.location.end), tx.location.strand)
                tx.sequence_pos=np.array(tx.sequence_pos)
                if tx.location.strand=='-':
                    assert np.all(tx.sequence_pos[:-1] >= tx.sequence_pos[1:]), "ERR in tx %s"+tx
                else:
                    assert np.all(tx.sequence_pos[:-1] <= tx.sequence_pos[1:]), "ERR in tx %s"+tx
                tx.sequence_spl=np.array(tx.sequence_spl)
                tx.sequence_spl[-1]=0 # remove marker for end of last exon!
                assert sum(tx.sequence_spl)==len(tx.introns) #
                tx.sequence_msk = np.array([0] * len(tx.sequence)) # for each position: 0=exon,3=3'utr, 5=5'utr
                for f in tx.get_features():
                    if f.type in [feature_type.utr3, feature_type.utr5]: # update mask
                        for i in range(len(tx.sequence_msk)):
                            if (tx.sequence_pos[i] >= f.location.start) and (tx.sequence_pos[i] <= f.location.end):
                                tx.sequence_msk[i] = f.type.value
        self.has_seq=True
    def save(self, out_file):
        """Pickle object"""
        print("Storing transcriptome model to %s" %out_file)
        with open(out_file, 'wb') as out:
            dill.dump(self, out, recurse=True, byref=True)
    @classmethod
    def load(cls, in_file):
        """Unpickle object"""
        print(f"Loading transcriptome model from {in_file}")
        gc.disable() # disable garbage collector
        with open(in_file, 'rb') as infile:
            obj=dill.load(infile)
        gc.enable()
        obj.cached=True
        print(f"Loaded {obj}")
        return(obj)
    @classmethod
    def load_or_build(cls, config, ensure_seq=False, ensure_qseq=False):
        """ Loads or builds a transcriptome """
        transcriptome_file = get_config(config, 'transcriptome_file', default_value=None)
        if (transcriptome_file) and (os.path.isfile(transcriptome_file)) and not (get_config(config,'rebuild_transcriptome',default_value=False)):
            t=transcriptome.load(transcriptome_file)
            if ensure_seq and (not t.has_seq):
                t.load_sequences()
            if ensure_qseq and (not t.has_quantseq):
                t.load_quantseq_data()  # load quantseq data
        else:
            t = transcriptome(config)
            if ensure_seq:
                t.load_sequences()
            if ensure_qseq:
                t.load_quantseq_data()  # load quantseq data
            if transcriptome_file:
                t.save(transcriptome_file)
        return t
    def find_kmers(self, kmer2tid, include_revcomp=False):
        """ Find kmer occurrences in the passed tx."""
        kmer2tx= {}
        for kmer,tids in kmer2tid.items():
            for tx in [self.transcripts[tid] for tid in tids]:
                for sp in find_all(tx.sequence, kmer):
                    features = set()
                    for p in range(sp, sp+len(kmer)):
                        features.add(tx.sequence_msk[p])
                    if kmer not in kmer2tx:
                        kmer2tx[kmer]=[]
                    kmer2tx[kmer].append((tx, sp, features))
                kmer_pos = kmer_search(tx.sequence, kmer, include_revcomp)
        return kmer2tx
    def load_quantseq_data(self):
            """ Expected columns: 'dataset', 'gene_symbol', 'gid', 'tid', 'readsCPM', 'g_readsCPM',
           'frac_readsCPM' """
            self.quantseq_datasets=set()
            df = pd.concat([chunk for chunk in tqdm(pd.read_csv(get_config(self.config, 'quantseq_tab', required=True), chunksize=1000, sep='\t'), desc='Loading quantseq data')])
            for r in df.to_dict(orient="records"):
                if (r['tid'] not in self.transcripts) or (r['gid'] not in self.genes):
                    self.log['qs_missing_rec']+=1
                    continue
                gene=self.genes[r['gid']]
                tx=self.transcripts[r['tid']]
                ds=r['dataset']
                self.quantseq_datasets.add(ds)
                if not hasattr(gene, 'g_readsCPM'):
                    gene.g_readsCPM=dict()
                if not hasattr(tx, 't_readsCPM'):
                    tx.t_readsCPM=dict()
                if not hasattr(tx, 'frac_readsCPM'):
                    tx.frac_readsCPM=dict()
                gene.g_readsCPM[ds]=r['g_readsCPM']
                tx.t_readsCPM[ds] = r['t_readsCPM']
                tx.frac_readsCPM[ds] = r['frac_readsCPM']
                self.log['qs_found_rec'] += 1
            self.has_quantseq = True

# t=transcriptome(config)
# t.save(config['transcriptome_file'])
# t.load_sequences() # load transcript (RNA) sequences.
# #t.transcripts['ENST00000456328.2'].exons[0].sequence
#
# f='/Volumes/groups/ameres/Niko/projects/Zuber/shRNA/round5/shrna_tables/tx.pkl'
# t.save(f) # store to a pickle file
# # t=transcriptome.load(f) # load from a pickle file
#
# # get rank of last exon
# t.genes['ENSG00000227232.5'].transcripts['ENST00000488147.1'].exons[-1].rnk
#
# # get seq of 1st intron
# t.transcripts['ENST00000488147.1'].introns[0].sequence
# get splice donor positions for tx. These are
# t.transcripts['ENST00000676189.1'].sequence_pos[t.transcripts['ENST00000676189.1'].sequence_spl==1]
# get 5'UTR sequence
# ''.join(np.array(list(tx.sequence))[tx.sequence_msk==5])
# get all SJ genomic pos in the 3'UTR
# tx.sequence_pos[(tx.sequence_spl==1) & (tx.sequence_msk==3)]

# t=transcriptome(config)
# t.load_sequences()
# t.load_quantseq_data()
#tx=t.transcripts['ENST00000676189.1'] # actb
#tx=t.transcripts['ENST00000325404.3'] # sox2

#lib=guide_library(config)
