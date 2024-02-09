"""
    Interfaces to other libraries:
    - Archs4Dataset: a class to access Archs4 datasets (https://maayanlab.cloud/archs4).
"""
import logging
import random

import h5py
import numpy as np
import pandas as pd
import s3fs
from tqdm.auto import tqdm
from datetime import datetime

import rnalib as rna


# pd.set_option('display.width', 400)
# pd.set_option('display.max_columns', 7)


class Archs4Dataset:
    """ A class to access the Archs4 dataset.



    Parameters
    ----------
    location: str
        The file path or an s3 URL (e.g.,  referencing the h5 file containing the data.
        NOTE that direct access via s3 bucket is slow and not recommended except for testing.

    Examples
    --------
    >>> location = 'data/human_gene_v2.2.h5' # or e.g., 'https://s3.dev.maayanlab.cloud/archs4/files/mouse_gene_v2.2.h5'
    >>> with Archs4Dataset(location) as a4: # load the dataset
    >>>     a4.describe() # prints the number of unique values for each metadata field
    >>>     df = a4.get_sample_metadata(filter_string = "readsaligned>5000000") # pandas filtering with query
    >>>     df.groupby('series_id').size().reset_index(name='counts') # a df with GEO series ids and counts
    >>>     df.query("series_id=='GSE124076,GSE222593'") # query from one series (byte strings!)
    >>>     df_sample = df.query("instrument_model.str.contains('HiSeq')").sample(10).index # 10 random HiSat samples
    >>>     df_cnt = a4.get_counts(samples = df_sample) # get counts for 10 random samples
    """

    def __init__(self, location):
        self.location = location
        if location.startswith('https://'):
            endpoint, d, f = location.rsplit('/', 2)
            s3_url = f's3://{d}/{f}'
            self.s3 = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': endpoint})
            self.file = h5py.File(self.s3.open(s3_url, 'rb'), 'r', lib_version='latest')
            self.is_local = False
        else:
            self.file = h5py.File(self.location, 'r')
            self.is_local = True
        self.meta_keys = self.get_meta_keys()
        self.all_samples = self.get_sample_dict(remove_sc=False)
        self.nosc_samples = self.get_sample_dict(remove_sc=True)
        self.genes = [x.decode("UTF-8") for x in np.array(self.file["meta/genes/symbol"])]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info(f"Closing Archs4Dataset at {self.location}.")
        self.file.close()

    def __repr__(self):
        return (f"{'Local' if self.is_local else 'Remote'} Archs4Dataset with {len(self.all_samples)} "
                f"({len(self.nosc_samples)} w/o SC) samples.")

    def describe(self):
        """ Gets metadata for 1k random samples and prints the number of unique values for each metadata field.
        """
        print(self)
        rand_samples = random.sample(list(self.nosc_samples), 1000)
        df_rand = self.get_sample_metadata(samples=rand_samples, cols=None, disable_progressbar=True)  # data for all
        # columns
        print("Results from 1000 random samples:")
        for k in self.meta_keys:
            val_set = set(df_rand[k].values)
            print(f"{k}:\t{len(val_set)} unique values (example: {next(iter(val_set))})")

    def get_meta_keys(self):
        """ Returns a list of all archs4 sample metadata keys.
        """
        return list(self.file['meta/samples'].keys())

    def get_sample_dict(self, remove_sc=True):
        """ Returns a dict of GSM ids and sample indices.
            If remove_sc is True (default), then single cell samples are removed.
        """
        gsm_ids = [x.decode("UTF-8") for x in np.array(self.file["meta/samples/geo_accession"])]
        if remove_sc and "singlecellprobability" in self.meta_keys:
            singleprob = np.array(self.file["meta/samples/singlecellprobability"])
            idx = sorted(list(np.where(singleprob < 0.5)[0]))
        else:
            idx = sorted(range(len(gsm_ids)))
        return {gsm_ids[i]: i for i in idx}

    def get_sample_metadata(self, filter_string=None,
                            samples=None,
                            cols=('characteristics_ch1', 'data_processing', 'extract_protocol_ch1', 'instrument_model',
                                  'last_update_date', 'library_selection', 'library_source', 'molecule_ch1',
                                  'platform_id', 'readsaligned', 'relation', 'sample', 'series_id',
                                  'singlecellprobability', 'source_name_ch1', 'status', 'submission_date', 'title'),
                            disable_progressbar=False):
        """
        Creates a pandas DataFrame with sample metadata for all samples matching the passing filter query.
        To group the resturned data by series_id, use df.groupby('series_id').size().reset_index(name='counts')

        Parameters
        ----------
        filter_string: str
            A query string to filter the samples by. See pandas.DataFrame.query for details (if None, a sample list must
            be set).
        samples: list
            A list of sample ids to retrieve metadata for (if None, all samples will be considered).
        cols: list
            A list of metadata fields to retrieve. If None, all fields will be retrieved.
        disable_progressbar: bool
            Whether to disable the progress bar.
        """
        assert not (filter is None and samples is None), "Either filter or sample_dict must be specified."
        sample_dict = self.nosc_samples  # all samples (non sc)
        if samples is not None:
            sample_dict = {k: v for k, v in sample_dict.items() if k in samples}
        if cols is None:
            cols = self.meta_keys  # all cols
        cols = cols and self.meta_keys  # get rid of invalid column names
        dts = []
        batches = rna.split_list(sample_dict.keys(), n=1000, is_chunksize=True)
        iterated, found = 0, 0
        for batch in (pbar := tqdm(batches, disable=disable_progressbar)):
            sample_idx = sorted([sample_dict[s] for s in batch])
            res = {}
            for k in cols:
                if k in ['last_update_date', 'submission_date']:  # date conversion
                    res[k] = np.array([datetime.strptime(d.decode("utf-8"), "%b %d %Y")
                                       for d in self.file[f"meta/samples/{k}"][sample_idx]])
                else:  # autodetect dtype
                    res[k] = np.array(self.file[f"meta/samples/{k}"][sample_idx])
            res = pd.DataFrame(res, index=batch)
            for col, dtype in res.dtypes.items():
                if dtype == object:  # Only process object columns.
                    res[col] = res[col].str.decode('utf-8').fillna(
                        res[col])  # decode or return original if decode returns Nan
            if filter_string is not None:
                res.query(filter_string, inplace=True)
            iterated += len(batch)
            found += len(res.index)
            pbar.set_description(f"Querying dataset, found {found}/{iterated} ({found / iterated * 100:.2f}%)")
            dts.append(res)
        df = pd.concat(dts)
        return df

    def get_counts(self, samples, gene_symbols=None, disable_progressbar=False):
        """
        Retrieve gene expression data from a specified file for the given sample and gene indices.

        Parameters
        ----------
        samples: list
            A list of sample ids to retrieve gene expression data for.
        gene_symbols: list
            A list of gene symbols to retrieve gene expression data for (if None, all genes will be considered).
        disable_progressbar: bool
            Whether to disable the progress bar.

        Returns
        -------
        pd.DataFrame:
            A pandas DataFrame containing the gene expression data.

        """
        sample_dict = self.nosc_samples  # all samples (non sc)
        if samples is not None:
            sample_dict = {k: v for k, v in sample_dict.items() if k in samples}
        row_encoding = "meta/genes/symbol"  # a4.data.get_encoding(file) # h5 path to expression data
        # get gene indices
        genes = np.array([x.decode("UTF-8") for x in np.array(self.file[row_encoding])])
        if gene_symbols is None:  # all genes
            gene_idx = list(range(len(genes)))
        else:
            gene_idx = [i for i, g in enumerate(genes) if g in gene_symbols]
        # get batches
        dat = []
        dts = []
        for sample_name, idx in tqdm(sample_dict.items(), disable=disable_progressbar):
            # NOTE: we don't do batch access as it's slow
            # see https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets
            dat.append(self.file["data/expression"][:, idx][gene_idx].T)
            if len(dat) == 10000:  # convert 10k rows to df
                dts.append(pd.DataFrame(dat, columns=genes[gene_idx]))
                dat = []
        if len(dat) > 0:  # convert remaining rows to df
            dts.append(pd.DataFrame(dat, columns=genes[gene_idx]))
        df = pd.concat(dts)
        df.index = sample_dict.keys()
        return df
#
# # TODO: add tests
# location='/Users/niko/projects/rnalib/notebooks/rnalib_testdata/bigfiles/human_gene_v2.2.h5'
# samples=['GSM3024561', 'GSM3112192', 'GSM3139594', 'GSM3147220']
# gene_symbols=['TSPAN6', 'TNMD']
# with Archs4Dataset(location) as a4:
#     print(a4.get_counts(samples, gene_symbols))
#     sample_dict = {k: v for k, v in a4.nosc_samples.items() if k in samples}
#     gene_idx = [i for i, g in enumerate(a4.genes) if g in gene_symbols]
#     print('run ', len(gene_idx), len(sample_dict))
#     dat = []
#     for sample_name, idx in tqdm(sample_dict.items(), disable=False):
#        print(idx, sample_name, gene_idx)
#        dat.append(a4.file["data/expression"][:, idx][gene_idx].T)
