import numpy as np


def l_child_idx(i):
    return 2 * i + 1


def r_child_idx(i):
    return 2 * i + 2


def split_to_node_descendants_matrix(max_depth: int, signed: bool = True):
    """
    A matrix representing the 'descendant of' relationship, not necessarily direct
    If signed represent the left/right (-1/+1) descendants
    """
    n_nodes: int = 2 ** max_depth - 1
    n_splits: int = 2 ** (max_depth - 1) - 1

    split_to_node_matrix = np.zeros((n_splits, n_nodes))

    def upd(s, j, i, val=None):
        if val is not None:
            s[j, i] = val

        l_child = l_child_idx(i)
        r_child = r_child_idx(i)

        if l_child < s.shape[1]:
            upd(s, j, l_child, val=-1 if val is None else val)
        if r_child < s.shape[1]:
            upd(s, j, r_child, val=+1 if val is None else val)
        pass

    for i in range(n_splits):
        upd(split_to_node_matrix, i, i)

    if not signed:
        split_to_node_matrix = np.abs(split_to_node_matrix)

    return split_to_node_matrix
