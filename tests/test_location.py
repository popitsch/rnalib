from pygenlib.genemodel import *
import pytest
from pathlib import Path
from itertools import product

@pytest.fixture(autouse=True)
def base_path() -> Path:
    """Go to testdata dir"""
    testdir = Path(__file__).parent.parent / "testdata/"
    print("Setting working dir to %s" % testdir)
    os.chdir(testdir)
    return testdir

def merge_result_lists(l):
    """ helper function """
    l1, l2 = zip(*l)
    m=loc_obj.merge(list(l1))
    seq=''.join([str(x) for x in l2])
    return m, seq

def test_loc_simple():
    # simple tests on single chrom
    d = {
        'a': loc_obj.from_str('chr1:1-10'),
        'b': loc_obj.from_str('chr1:5-20'),
        'c': loc_obj.from_str('chr1:1-20'),
        'd': loc_obj.from_str('chr1:15-20')
    }
    assert d['a']<d['b']
    assert d['a']!=d['b']
    assert d['b']!=d['a']
    assert d['a']==d['a']
    assert d['a'].overlaps(d['b'])
    assert not d['a'].overlaps(d['d'])
    assert list(sorted(d.values()))==[d['a'], d['c'], d['b'], d['d']]
    assert list(reversed(sorted(d.values())))==list(reversed([d['a'], d['c'], d['b'], d['d']]))
    # stranded
    d['a'].strand='+'
    assert ((d['a'] < d['b']) is None) and ((d['a'] > d['b']) is None) and ((d['a'] == d['b']) is False) and (d['a'] != d['b'])

def test_loc_overlaps():
    # Some overlap tests
    # .........1........  ....2......
    # |-a-|
    #     |-b-|
    #   |-c-|
    #           |-d-|
    #                         |-e-|
    #                     |-f--------|
    d = {
        'a': loc_obj('1', 1, 10),
        'b': loc_obj('1', 11, 20),
        'c': loc_obj('1', 5, 15),
        'd': loc_obj('1', 30, 40),
        'e': loc_obj('2', 21, 30),
        'f': loc_obj('2', 1, 50)
    }
    d_plus = {x: d[x].copy() for x in d}
    d_minus = {x: d[x].copy() for x in d}
    for x,y in zip(d_plus.values(), d_minus.values()):
        x.strand='+'
        y.strand='-'
    assert (not d['a'].overlaps(d['b']))
    assert (d['a'].overlaps(d['c']))
    assert (d['c'].overlaps(d['a']))
    assert (d['b'].overlaps(d['c']))
    assert (d['e'].overlaps(d['f']))
    assert (d['f'].overlaps(d['e']))
    # is_adjacent
    assert (d['a'].is_adjacent(d['b']))
    # wrong chrom: no overlap
    for x, y in product(['a', 'b', 'c', 'd'], ['e', 'f']):
        assert (not d[x].overlaps(d[y]))
        assert (not d[x].is_adjacent(d[y]))
    # wrong strand: no overlap
    for x in d.keys():
        assert (not d[x].overlaps(d_plus[x]))
        assert (not d_minus[x].overlaps(d_plus[x]))
        assert (not d[x].is_adjacent(d_plus[x]))
        assert (not d_minus[x].is_adjacent(d_plus[x]))

def test_loc_merge():
    # Some merging tests
    # .........1........  ....2......
    # |-a-|
    #     |-b-|
    #   |-c-|
    #           |-d-|
    #                         |-e-|
    #                     |-f--------|
    d = {
        'a': loc_obj('1', 1, 10),
        'b': loc_obj('1', 11, 20),
        'c': loc_obj('1', 5, 15),
        'd': loc_obj('1', 30, 40),
        'e': loc_obj('2', 21, 30),
        'f': loc_obj('2', 1, 50)
    }
    d_plus = {x: d[x].copy() for x in d}
    d_minus = {x: d[x].copy() for x in d}
    for x, y in zip(d_plus.values(), d_minus.values()):
        x.strand = '+'
        y.strand = '-'
    assert loc_obj.merge([d_plus['a'], d_plus['a']]) == d_plus['a']
    assert loc_obj.merge([d['a'], d['a']]) == d['a']
    assert loc_obj.merge([d['a'], d['b']]) == loc_obj('1', 1, 20)
    assert loc_obj.merge([d['a'], d['e']]) is None  # chrom mismatch
    assert loc_obj.merge([d['e'], d['f']]) == loc_obj('2', 1, 50) # containment
    assert loc_obj.merge([d['a'], d_plus['a']]) is None  # strand mismatch
    assert loc_obj.merge([d_plus['a'], d_plus['a']]) == d_plus['a']
