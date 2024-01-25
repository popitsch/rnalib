"""
Tests for GenomicInterval class
"""

from itertools import product, pairwise

import rnalib
from rnalib import gi, MAX_INT, ReferenceDict

assert rnalib.__RNALIB_TESTDATA__ is not None, ("Please set rnalib.__RNALIB_TESTDATA__ variable to the testdata "
                                                "directory path")
def merge_result_lists(lst):
    """ helper function """
    l1, l2 = zip(*lst)
    m = gi.merge(list(l1))
    seq = ''.join([str(x) for x in l2])
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
    assert d['a'] < d['b']
    assert d['a'] != d['b']
    assert d['b'] != d['a']
    assert d['a'] == d['a']
    assert d['a'].overlaps(d['b'])
    assert not d['a'].overlaps(d['d'])
    assert list(sorted(d.values())) == [d['a'], d['c'], d['b'], d['d']]
    assert list(reversed(sorted(d.values()))) == list(reversed([d['a'], d['c'], d['b'], d['d']]))
    # stranded
    d['a+'] = d['a'].get_stranded('+')
    assert (d['a+'] < d['b']) and (d['d'] > d['a+']) and (not d['a+'] == d['b']) and (d['a+'] != d['b'])


def test_loc_overlaps():
    # Some overlap tests
    # .........1........  ....2......
    # |-a-|
    #     |-b-|
    #   |-c-|
    #           |-d-|
    #                         |-e-|
    #                     |-feature--------|
    d = {
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'feature': gi('2', 1, 50)
    }
    d_plus = {x: d[x].get_stranded('+') for x in d}
    d_minus = {x: d[x].get_stranded('-') for x in d}

    assert (not d['a'].overlaps(d['b']))
    assert (d['a'].overlaps(d['c']))
    assert (d['c'].overlaps(d['a']))
    assert (d['b'].overlaps(d['c']))
    assert (d['e'].overlaps(d['feature']))
    assert (d['feature'].overlaps(d['e']))
    # is_adjacent
    assert (d['a'].is_adjacent(d['b']))
    # wrong chrom: no overlap
    for x, y in product(['a', 'b', 'c', 'd'], ['e', 'feature']):
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
        'feature': gi('2', 1, 50)
    }
    # unrestricted intervals
    assert d['a'].overlaps(gi('1', None, None))
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
    #                     |-feature--------|
    d = {
        'a': gi('1', 1, 10),
        'b': gi('1', 11, 20),
        'c': gi('1', 5, 15),
        'd': gi('1', 30, 40),
        'e': gi('2', 21, 30),
        'feature': gi('2', 1, 50)
    }
    d_plus = {x: d[x].get_stranded('+') for x in d}
    # d_minus = {x: d[x].get_stranded('-') for x in d}
    assert gi.merge([d_plus['a'], d_plus['a']]) == d_plus['a']
    assert gi.merge([d['a'], d['a']]) == d['a']
    assert gi.merge([d['a'], d['b']]) == gi('1', 1, 20)
    assert gi.merge([d['a'], d['e']]) is None  # chrom mismatch
    assert gi.merge([d['e'], d['feature']]) == gi('2', 1, 50)  # containment
    assert gi.merge([d['a'], d_plus['a']]) is None  # strand mismatch
    assert gi.merge([d_plus['a'], d_plus['a']]) == d_plus['a']


def test_loc_merge_unrestricted():
    d = {
        'a': gi('1', 1, 10),
        'b': gi(None, 11, 20)
    }
    assert gi.merge([d['a'], d['b']]) is None
    assert gi.merge([d['b'], d['a']]) is None


# def test_loc_merge_sorted():
#     a = {
#         'a': gi('1', 1, 10),
#         'c': gi('1', 5, 15),
#         'feature': gi('2', 1, 50),
#         'e': gi('2', 21, 30),
#     }
#     b = {
#         'a': gi('1', 1, 10),
#         'c': gi('3', 5, 15),
#         'feature': gi('4', 1, 50),
#         'e': gi('4', 21, 30)
#     }
#     for n,l in a.items():
#         l.data=n
#     for n,l in b.items():
#         l.data=n
#     for i in heapq.merge(a.values(), b.values()):
#         print(i, i.data)

def test_distance():
    a = from_str('1:1-10,1:1-10,1:10-20,1:25-30,1:1-10,2:1-10,2:11-12')
    dist = [a.distance(b) for a, b in pairwise(a)]
    assert dist == [0, 0, 5, -15, None, 1]


def test_eq():
    a = from_str('1:1-10')
    assert a[0] in a
    assert a[0].copy() in a
    b = a[0].get_stranded('+')
    assert b not in a


def test_regexp():
    locs = "1:1-10(+),  1:1-10 (-), chr2:1-10, 1:30-40    (+),"
    assert str(from_str(locs)) == "[1:1-10 (+), 1:1-10 (-), chr2:1-10, 1:30-40 (+), None]"


def test_dict():
    f = from_str('1:1-10 (+),1:1-10 (-),1:10-20 (+),1:25-30 (-),1:1-10 (+),2:1-10,2:11-12')
    d = {k: str(k) for k in f}
    assert all(k in d for k in f) and all(k in f for k in f)


def test_overlap():
    assert gi.from_str('1:1-10 (+)').overlap(gi.from_str('1:1-10 (+)')) == 10
    assert gi.from_str('1:1-10 (+)').overlap(gi.from_str('1:1-10 (-)')) == 10
    assert gi.from_str('1:1-10 (+)').overlap(gi.from_str('1:1-10 (-)'), strand_specific=True) == 0
    assert gi.from_str('1:1-10').overlap(gi.from_str('1:10-15')) == 1
    assert gi.from_str('1:5-10 (+)').overlap(gi.from_str('1:1-5 (-)'), strand_specific=True) == 0


def test_sort():
    """Assert sort order including empty and unbounded intervals"""
    locs = from_str("1:1-10(+),3:5-100,1:1-10 (-), chr2:1-10,1:30-40(+)") + [gi(None, 200, 300), gi('1', end=10), gi('chr2', 2, 1)]  # empty
    # sorted(locs) # no chrom order!
    refdict = ReferenceDict({'1': None, 'chr2': None, '3': None}, name='test')
    assert sorted(locs, key=lambda x: (refdict.index(x.chromosome), x)), [locs[x] for x in [4, 5, 0, 2, 6, 3, 1]]


def test_len():
    assert len(gi('chr1', 1, 2)) == 2  # interval
    assert len(gi('chr1', 1, 1)) == 1  # point
    assert gi('chr1', 20, 10).is_empty() and len(gi('chr1', 20, 10)) == 0  # empty interval
    assert len(gi()) == MAX_INT  # None:-inf-inf # unbounded intervals
    assert len(gi('chr1', 1)) == MAX_INT  # chr1:1-inf
    assert len(gi('chr1', None, 1)) == MAX_INT  # chr1:-inf-1


def test_empty():
    assert gi('chr1', 1, 0).is_empty()
    assert not gi('chr1', 1, 0).overlaps(gi('chr1', 1, 0))
    assert gi('chr1', 1, 0).overlap(gi('chr1', 1, 0)) == 0
    assert gi('chr1', 1, 0) == gi('chr1', 1000, 999)  # empty intervals are equal if on same chrom
    assert gi('chr1', 1, 0) != gi('chr2', 1000, 999)


def test_unbounded():
    assert gi().is_unbounded()


def test_updownstream():
    assert gi('chr1', 10, 20).get_upstream(3) is None  # no strand: return None
    assert [gi('chr1', 10, 20, '+').get_upstream(3)] == from_str("chr1:7-9(+)")
    assert [gi('chr1', 10, 20, '+').get_downstream(3)] == from_str("chr1:21-23(+)")
    assert [gi('chr1', 10, 20, '-').get_downstream(3)] == from_str("chr1:7-9(-)")
    assert [gi('chr1', 10, 20, '-').get_upstream(3)] == from_str("chr1:21-23(-)")
