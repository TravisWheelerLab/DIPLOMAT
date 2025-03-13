from diplomat.predictors.sfpe.avl_tree import BufferTree, insert, nearest_pop, inorder_traversal, _DEPTH
import numpy as np


def test_tree_ordering():
    # Small, easy to check test...
    t = BufferTree(bytearray(BufferTree.get_buffer_size(20)))
    insert(t, 20, 200)
    insert(t, 14, 90)
    insert(t, 25, 10)
    insert(t, 20, 300)
    insert(t, 500, 20)
    insert(t, 30, 300)
    insert(t, 33, 10)
    insert(t, 50, 20)
    insert(t, 20, 200)
    assert nearest_pop(t, 18) == (20, 200)
    assert nearest_pop(t, 25) == (25, 10)
    assert nearest_pop(t, 16, left=True) == (14, 90)
    assert nearest_pop(t, 33, 10) == (33, 10)
    assert nearest_pop(t, 600) == (None, None)
    assert np.all(inorder_traversal(t) == [
        [20, 300],
        [30, 300],
        [50, 20],
        [500, 20]
    ])

    # Test 2: Make sure tree structure remains stable after a large number of insertions and then deletions,
    # and it's inorder traversal is still in order...
    np.random.seed(0)
    t2 = BufferTree(bytearray(BufferTree.get_buffer_size(10000)))
    for i in range(10000):
        insert(t2, np.random.randint(1, 1000), np.random.randint(1, 1000))

    assert t2.size == 9951

    for i in range(5000):
        res = nearest_pop(t2, np.random.randint(1, 1000), np.random.randint(1, 1000))

    assert t2.size == 4951
    assert t2.data[t2.root, _DEPTH] == 15
    t2_ordered = inorder_traversal(t2)
    assert np.all(t2_ordered[np.lexsort((t2_ordered[:, 1], t2_ordered[:, 0]))] == t2_ordered)

    # Test 3: Make sure a running nearest search from below returns same values as trivial algorithm...
    for i in range(5000):
        lookup_key = np.random.randint(1, 1000)
        lookup_val = np.random.randint(1, 1000)
        kb, vb = nearest_pop(t2, lookup_key, lookup_val, left=True)

        if(kb is not None):
            assert kb < lookup_key or (kb == lookup_key and vb <= lookup_val)

        below_vals = np.flatnonzero((t2_ordered[:, 0] < lookup_key) | ((t2_ordered[:, 0] == lookup_key) & (t2_ordered[:, 1] <= lookup_val)))

        if(kb is not None):
            assert np.all(t2_ordered[below_vals[-1]] == [kb, vb])
        else:
            assert below_vals.size == 0

        if(kb is not None):
            insert(t2, kb, vb)

    # Test 4: Make sure a running nearest search from above returns same values as trivial algorithm...
    for i in range(5000):
        lookup_key = np.random.randint(1, 1000)
        lookup_val = np.random.randint(1, 1000)
        ka, va = nearest_pop(t2, lookup_key, lookup_val, left=False)

        if(ka is not None):
            assert ka > lookup_key or (ka == lookup_key and va >= lookup_val)

        above_vals = np.flatnonzero((t2_ordered[:, 0] > lookup_key) | ((t2_ordered[:, 0] == lookup_key) & (t2_ordered[:, 1] >= lookup_val)))

        if(ka is not None):
            assert np.all(t2_ordered[above_vals[0]] == [ka, va])
        else:
            assert above_vals.size == 0

        if(ka is not None):
            insert(t2, ka, va)