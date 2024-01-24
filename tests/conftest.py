""" Pytest configuration file for the test suite. """
import os
from pathlib import Path

import pytest

from pygenlib import TranscriptFilter
from pygenlib.testdata import get_resource


@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir


@pytest.fixture(autouse=True)
def default_testdata() -> dict:
    config = {
        'genome_fa': get_resource('ACTB+SOX2_genome'),  # get_resource('ACTB+SOX2_genome'),
        'genome_offsets': {'chr3': 181711825, 'chr7': 5526309},
        'annotation_gff': get_resource('gencode_gff'),  # get_resource('gencode_gff'),,
        'annotation_flavour': 'gencode',
        'feature_filter': {'gene': {'included': {'gene_type': ['protein_coding']}}}
    }
    return config
