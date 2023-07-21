import numpy as np


def l_child_idx(i):
    return 2 * i + 1


def r_child_idx(i):
    return 2 * i + 2


def split_to_node_descendants_matrix(max_depth: int, signed: bool = True, leaves_only: bool = False):
    n_nodes: int = (max_depth - 1) + 2 ** (max_depth - 1)
    n_splits: int = max_depth - 1

    split_to_node_matrix = np.zeros((n_splits, n_nodes))

    for j in range(n_splits):
        for i in range(n_splits, n_nodes):
            v = 1 if int((1 + j + i) / (1 + j)) % 2 == 0 else -1
            split_to_node_matrix[j, i] = v

    if not signed:
        split_to_node_matrix = np.abs(split_to_node_matrix)

    if leaves_only:
        split_to_node_matrix = split_to_node_matrix[:, -(n_nodes - n_splits):]

    return split_to_node_matrix


split_to_node_descendants_matrix(4, leaves_only=True)
