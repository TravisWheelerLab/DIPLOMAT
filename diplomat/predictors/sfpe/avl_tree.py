from typing import Tuple, Optional
from typing_extensions import Protocol
import numpy as np
import numba
from numba.core.types.containers import Tuple as NumbaTuple


_KEY = 0
_VALUE = 1
_DEPTH = 2
_LESS_OR_EQ = 3
_MORE = 4
_PARENT = 5


class Tree(Protocol):
    root: int
    size: int
    data: np.ndarray


class NumpyTree:
    def __init__(self, max_size: int):
        self.root = 0
        self.size = 0
        self.data = np.zeros((max_size, 6), dtype=np.int64)


class BufferTree:
    def __init__(self, buffer):
        int_size = np.dtype(np.int64).itemsize
        max_size = (len(buffer) - (2 * int_size)) // (6 * int_size)
        if(max_size <= 0):
            raise ValueError("Buffer provided not big enough to store a tree...")

        self._root = np.ndarray((1,), np.int64, buffer, 0, order="C")
        self._size = np.ndarray((1,), np.int64, buffer, int_size, order="C")
        self.data = np.ndarray((max_size, 6), np.int64, buffer, int_size * 2, order="C")

    @property
    def size(self) -> int:
        return self._size[0]

    @size.setter
    def size(self, val: int):
        self._size[0] = val

    @property
    def root(self) -> int:
        return self._root[0]

    @root.setter
    def root(self, val: int):
        self._root[0] = val

    @classmethod
    def get_buffer_size(cls, tree_size: int) -> int:
        int_size = np.dtype(np.int64).itemsize
        return tree_size * (6 * int_size) + (2 * int_size)


def tree_to_string(tree: Tree) -> str:
    return f"{type(tree).__name__}(root={tree.root}, size={tree.size}, data=\n{tree.data[:tree.size]}\n)"


def insert(tree: Tree, key: int, val: int) -> bool:
    if(tree.size >= tree.data.shape[0]):
        raise ValueError("Tree is full.")

    tree.data[tree.size] = (key, val, 1, -1, -1, -1)
    tree.root, was_inserted = _insert(tree.data, tree.root, tree.size)
    assert tree.root >= 0
    tree.data[tree.root, _PARENT] = -1

    if(was_inserted):
        tree.size = tree.size + 1
    else:
        tree.data[tree.size, :] = 0

    return was_inserted


oint = Optional[int]


def nearest_pop(tree: Tree, key: int, val: Optional[int] = None, left: bool = False) -> Tuple[oint, oint]:
    if(tree.size <= 0):
        return (None, None)
    index = _nearest_search(tree.data, tree.root, key, val is not None, val if(val is not None) else 0, bool(left))

    if(index == -1):
        return (None, None)

    # Try removing the selected node...
    key, val = tree.data[index, _KEY], tree.data[index, _VALUE]
    _remove(tree, index, tree.size - 1)
    tree.size = tree.size - 1

    return key, val


def remove(tree: Tree, key: int, val: int) -> bool:
    if(tree.size <= 0):
        return False
    index = _nearest_search(tree.data, tree.root, key, True, val, False)

    if(index == -1 or (tree.data[index, _KEY] != key) or (tree.data[index, _VALUE] != val)):
        return False

    _remove(tree, index, tree.size - 1)
    tree.size = tree.size - 1
    return True


def inorder_traversal(tree: Tree) -> np.ndarray:
    inorder_lst = np.zeros((tree.size, 2), dtype=np.int64)
    if(tree.size == 0):
        return inorder_lst
    _inorder_traversal(tree.data, tree.root, 0, inorder_lst)
    return inorder_lst


@numba.njit("int64(int64, int64)")
def imax(a: int, b: int) -> int:
    return a if(a > b) else b


@numba.njit
def _depth(tree: np.ndarray, current_index: int, direction: int):
    child_idx = tree[current_index, direction]
    if(child_idx == -1):
        return 0
    else:
        return tree[child_idx, _DEPTH]


@numba.njit
def _recompute_depth(tree: np.ndarray, index: int):
    tree[index, _DEPTH] = imax(
        _depth(tree, index, _LESS_OR_EQ) + 1,
        _depth(tree, index, _MORE) + 1
    )


@numba.njit
def _rotate_left(tree: np.ndarray, current_index: int) -> int:
    right_child = tree[current_index, _MORE]
    right_left_child = tree[right_child, _LESS_OR_EQ]

    tree[current_index, _MORE] = right_left_child
    if(right_left_child >= 0):
        tree[right_left_child, _PARENT] = current_index

    tree[right_child, _LESS_OR_EQ] = current_index
    tree[current_index, _PARENT] = right_child

    _recompute_depth(tree, current_index)
    _recompute_depth(tree, right_child)

    return right_child


@numba.njit
def _rotate_right(tree: np.ndarray, current_index: int) -> int:
    left_child = tree[current_index, _LESS_OR_EQ]
    left_right_child = tree[left_child, _MORE]

    tree[current_index, _LESS_OR_EQ] = left_right_child
    if(left_right_child >= 0):
        tree[left_right_child, _PARENT] = current_index

    tree[left_child, _MORE] = current_index
    tree[current_index, _PARENT] = left_child

    _recompute_depth(tree, current_index)
    _recompute_depth(tree, left_child)

    return left_child


@numba.njit(numba.int64(numba.int64[:, :], numba.int64))
def _rebalance(tree: np.ndarray, current_idx: int) -> int:
    # Cleanup: If the balance is more than 2 we rotate left
    less_depth = _depth(tree, current_idx, _LESS_OR_EQ)
    more_depth = _depth(tree, current_idx, _MORE)

    if(more_depth > less_depth + 1):
        return _rotate_left(tree, current_idx)
    elif(less_depth > more_depth + 1):
        return _rotate_right(tree, current_idx)
    else:
        return current_idx


@numba.njit(numba.int64(numba.int64[:, :], numba.int64, numba.int64, numba.boolean, numba.int64, numba.boolean))
def _nearest_search(tree: np.ndarray, current_idx: int, key: int, use_val: bool, val: int, left: bool) -> int:
    if(current_idx < 0):
        return -1

    if(tree[current_idx, _KEY] < key):
        branch = _MORE
        current_selection = current_idx if(left) else -1
    elif(tree[current_idx, _KEY] == key):
        if(use_val):
            if(tree[current_idx, _VALUE] < val):
                branch = _MORE
                current_selection = current_idx if (left) else -1
            elif(tree[current_idx, _VALUE] > val):
                branch = _LESS_OR_EQ
                current_selection = -1 if (left) else current_idx
            else:
                branch = _MORE if (left) else _LESS_OR_EQ
                current_selection = current_idx
        else:
            branch = _MORE if(left) else _LESS_OR_EQ
            current_selection = current_idx
    else:
        branch = _LESS_OR_EQ
        current_selection = -1 if(left) else current_idx

    sub_selection = _nearest_search(tree, tree[current_idx, branch], key, use_val, val, left)

    return sub_selection if(sub_selection >= 0) else current_selection


@numba.njit(numba.int64(numba.int64[:, :], numba.int64))
def _rebalance_all_from(tree: np.ndarray, index: int) -> int:
    current = index
    parent = tree[current, _PARENT]

    while(True):
        direction = _LESS_OR_EQ if(parent >= 0 and tree[parent, _LESS_OR_EQ] == current) else _MORE
        new_subtree_parent = _rebalance(tree, current)

        if(parent < 0):
            tree[new_subtree_parent, _PARENT] = -1
            return new_subtree_parent

        tree[new_subtree_parent, _PARENT] = parent
        tree[parent, direction] = new_subtree_parent
        _recompute_depth(tree, parent)

        current = parent
        parent = tree[parent, _PARENT]


def _remove(full_tree: Tree, index: int, last_idx: int):
    tree = full_tree.data

    if(index < 0):
        return

    removed_item = _find_substitute_leaf(tree, index)

    if(removed_item < 0):
        # No children, we can just delete the current node...
        parent_val = tree[index, _PARENT]
        if(parent_val < 0):
            tree[index, :] = 0
            full_tree.root = 0
            return
        elif(tree[parent_val, _LESS_OR_EQ] == index):
            tree[parent_val, _LESS_OR_EQ] = -1
        else:
            tree[parent_val, _MORE] = -1
        _recompute_depth(tree, parent_val)
        full_tree.root = _rebalance_all_from(tree, parent_val)
        removed_item = index
    else:
        tree[index, _KEY] = tree[removed_item, _KEY]
        tree[index, _VALUE] = tree[removed_item, _VALUE]
        full_tree.root = _rebalance_all_from(tree, index)

    # Swap ending value into deleted space...
    tree[removed_item, :] = tree[last_idx, :]
    tree[last_idx, :] = 0

    if(removed_item == last_idx):
        # Removed item same as last index, avoid swap code...
        return

    swapped_parent = tree[removed_item, _PARENT]
    if(swapped_parent < 0):
        full_tree.root = removed_item
    elif(tree[swapped_parent, _LESS_OR_EQ] == last_idx):
        tree[swapped_parent, _LESS_OR_EQ] = removed_item
    else:
        tree[swapped_parent, _MORE] = removed_item

    if(tree[removed_item, _LESS_OR_EQ] >= 0):
        tree[tree[removed_item, _LESS_OR_EQ], _PARENT] = removed_item
    if(tree[removed_item, _MORE] >= 0):
        tree[tree[removed_item, _MORE], _PARENT] = removed_item


@numba.njit(NumbaTuple((numba.int64, numba.int64))(numba.int64[:, :], numba.int64, numba.int64))
def _remove_outward_leaf(tree: np.ndarray, current_index: int, direction: int) -> Tuple[int, int]:
    if(current_index < 0):
        return (-1, -1)

    if(tree[current_index, direction] < 0):
        # Perform the removal...
        opposite_direction = _MORE if(direction == _LESS_OR_EQ) else _LESS_OR_EQ
        tree[current_index, _PARENT] = -1
        if(tree[current_index, opposite_direction] >= 0):
            return (tree[current_index, opposite_direction], current_index)
        return (-1, current_index)

    new_root, removed_item = _remove_outward_leaf(tree, tree[current_index, direction], direction)
    tree[current_index, direction] = new_root
    if(new_root >= 0):
        tree[new_root, _PARENT] = current_index
    _recompute_depth(tree, current_index)

    # Re-balance the tree...
    new_root = _rebalance(tree, current_index)

    return new_root, removed_item


@numba.njit(numba.int64(numba.int64[:, :], numba.int64, numba.int64, numba.int64[:, :]))
def _inorder_traversal(tree: np.ndarray, index: int, list_index: int, lst: np.ndarray) -> int:
    if(index < 0):
        return list_index

    list_index = _inorder_traversal(tree, tree[index, _LESS_OR_EQ], list_index, lst)

    lst[list_index, 0] = tree[index, _KEY]
    lst[list_index, 1] = tree[index, _VALUE]
    list_index += 1

    list_index = _inorder_traversal(tree, tree[index, _MORE], list_index, lst)
    return list_index


@numba.njit(numba.int64(numba.int64[:, :], numba.int64))
def _find_substitute_leaf(tree: np.ndarray, index: int):
    new_root, removed_item = _remove_outward_leaf(tree, tree[index, _LESS_OR_EQ], _MORE)
    tree[index, _LESS_OR_EQ] = new_root
    if(new_root >= 0):
        tree[new_root, _PARENT] = index

    if(removed_item >= 0):
        _recompute_depth(tree, index)
        return removed_item

    new_root, removed_item = _remove_outward_leaf(tree, tree[index, _MORE], _LESS_OR_EQ)
    tree[index, _MORE] = new_root
    if(new_root >= 0):
        tree[new_root, _PARENT] = index

    _recompute_depth(tree, index)
    return removed_item


@numba.njit(NumbaTuple((numba.int64, numba.boolean))(numba.int64[:, :], numba.int64, numba.int64))
def _insert(tree: np.ndarray, current_idx: int, ins_index: int) -> Tuple[int, bool]:
    if(current_idx < 0 or current_idx == ins_index):
        return (ins_index, True)

    # Compute the next branch to go down, and update the balance of this node...
    if(tree[current_idx, _KEY] < tree[ins_index, _KEY]):
        branch = _MORE
    elif(tree[current_idx, _KEY] == tree[ins_index, _KEY] and tree[current_idx, _VALUE] < tree[ins_index, _VALUE]):
        branch = _MORE
    elif(tree[current_idx, _KEY] == tree[ins_index, _KEY] and tree[current_idx, _VALUE] == tree[ins_index, _VALUE]):
        return current_idx, False
    else:
        branch = _LESS_OR_EQ

    next_idx = tree[current_idx, branch]

    root_idx, did_insert = _insert(tree, next_idx, ins_index)
    assert root_idx >= 0
    tree[current_idx, branch] = root_idx
    tree[root_idx, _PARENT] = current_idx
    _recompute_depth(tree, current_idx)

    new_root = _rebalance(tree, current_idx)

    return new_root, did_insert


def _tree_test():
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


if(__name__ == "__main__"):
    _tree_test()