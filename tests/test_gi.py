from pygenlib.genemodel import gi
import pytest
from pathlib import Path
from itertools import product, pairwise
import heapq
import os
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
    m=gi.merge(list(l1))
    seq=''.join([str(x) for x in l2])
    return m, seq

def from_str(s):
    return [gi.from_str(x) for x in s.split(',')]

def test_loc_simple():
    # simple tests on single chrom
    d = {
        'a': gi.from_str('chr1:1-10'),
        'b': gi.from_str('chr1:5-20'),
        'c': gi.from_str('chr1:1-20'),
        'd': gi.from_str('chr1:15-20')
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
    assert (d['a'] < d['b']) and (d['d'] > d['a']) and (not d['a'] == d['b']) and (d['a'] != d['b'])

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
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'f': gi('2', 1, 50)
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
        assert (not d[x].overlaps(d_plus[x], strand_specific=True))
        assert (not d_minus[x].overlaps(d_plus[x], strand_specific=True))
        assert (not d[x].is_adjacent(d_plus[x], strand_specific=True))
        assert (not d_minus[x].is_adjacent(d_plus[x], strand_specific=True))

def test_loc_overlaps_unrestricted():
    d = {
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'f': gi('2', 1, 50)
    }
    # unrestricted intervals
    assert d['a'].overlaps(gi('1',None,None))
    assert d['a'].overlaps(gi('1', 8, None))
    assert not d['a'].overlaps(gi('1', 11, None))
    assert d['a'].overlaps(gi(None, 1, None))

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
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'f': gi('2', 1, 50)
    }
    d_plus = {x: d[x].copy() for x in d}
    d_minus = {x: d[x].copy() for x in d}
    for x, y in zip(d_plus.values(), d_minus.values()):
        x.strand = '+'
        y.strand = '-'
    assert gi.merge([d_plus['a'], d_plus['a']]) == d_plus['a']
    assert gi.merge([d['a'], d['a']]) == d['a']
    assert gi.merge([d['a'], d['b']]) == gi('1', 1, 20)
    assert gi.merge([d['a'], d['e']]) is None  # chrom mismatch
    assert gi.merge([d['e'], d['f']]) == gi('2', 1, 50) # containment
    assert gi.merge([d['a'], d_plus['a']]) is None  # strand mismatch
    assert gi.merge([d_plus['a'], d_plus['a']]) == d_plus['a']

def test_loc_merge_unrestricted():
    d = {
        'a': gi('1', 1, 10),
        'b': gi(None, 11, 20)
    }
    print(gi.merge([d['a'], d['b']]))
    print(gi.merge([d['b'], d['a']]))

def test_loc_merge_sorted():
    a = {
        'a': gi('1', 1, 10),
        'c': gi('1', 5, 15),
        'f': gi('2', 1, 50),
        'e': gi('2', 21, 30),
    }
    b = {
        'a': gi('1', 1, 10),
        'c': gi('3', 5, 15),
        'f': gi('4', 1, 50),
        'e': gi('4', 21, 30)
    }
    for n,l in a.items():
        l.data=n
    for n,l in b.items():
        l.data=n
    for i in heapq.merge(a.values(), b.values()):
        print(i, i.data)

def test_distance():
    a=from_str('1:1-10,1:1-10,1:10-20,1:25-30,1:1-10,2:1-10,2:11-12')
    dist=[a.distance(b) for a,b in pairwise(a)]
    assert dist==[0, 0, 5, -15, None, 1]


def test_sort():
    a=from_str('1:1-10,1:1-10,1:10-20,1:25-30,1:1-10,2:1-10,2:11-12')
    for i,x in enumerate(a):
        x.strand='+' if i%2==0 else '-'
    list(sorted(a))


def test_eq():
    a = from_str('1:1-10')
    assert a[0] in a
    assert a[0].copy() in a
    b=a[0].copy()
    b.strand='-'
    assert b not in a