import tensorflow as tf
from tensorflow import keras
import numpy as np

from .tree_struct import binary_trees_struct


def g(a, q, reg_lambda: float):
    reg_part = keras.backend.sum(a ** 2, axis=0)
    sqr_part = (a - (q + 0.5)) ** 2 * keras.backend.cast_to_floatx(a <= q + 0.5)
    sqr_part = keras.backend.sum(sqr_part, axis=0)
    return 0.5 * reg_lambda * reg_part + 0.5 * sqr_part


class LatentTree(keras.layers.Layer):
    """
    Latent trees as described in 'Learning Binary Decision Trees by Argmin Differentiation'
    The inner optimization loop is a little different, using softmax and set of candidate solutions instead of exact solving
    https://arxiv.org/pdf/2010.04627.pdf
    """

    def __init__(self, max_depth: int, reg_lambda: float = 0.0):
        super(LatentTree, self).__init__()

        self.reg_lambda: float = reg_lambda
        self.max_depth: int = max_depth

        split_to_node = binary_trees_struct.split_to_node_descendants_matrix(max_depth, signed=True)
        self.n_splits, self.n_nodes = split_to_node.shape
        split_to_node = split_to_node.reshape(-1, self.n_splits, self.n_nodes)
        self.split_to_node = keras.backend.constant(split_to_node)

        self.q_layer = keras.layers.Dense(
            units=self.n_splits,
            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode="fan_in"),
            bias_initializer=keras.initializers.RandomNormal()
        )

        node_idx = keras.backend.arange(self.n_nodes)
        parent_idx = (node_idx - 1) // 2

        self.node_to_parent = np.zeros((self.n_nodes, self.n_nodes))
        self.node_to_parent[node_idx[1:], parent_idx[1:]] = 1.0
        self.idx = keras.backend.eye(self.n_nodes)

        self.q_by_node = keras.backend.ones((1, self.n_nodes))

    def _q_compute(self, x):
        method = 0
        if method == 0:
            q_by_split = self.q_layer(x)
            q_by_split = keras.backend.reshape(q_by_split, (-1, self.n_splits, 1))
            q_by_split_and_node = q_by_split * self.split_to_node + 1.0 - keras.backend.abs(self.split_to_node)
            q_by_node = keras.backend.min(q_by_split_and_node, axis=1)
            return q_by_node
        if method == 1:
            q_by_split = self.q_layer(x)
            q_by_node = [1.0 for _ in range(self.n_nodes)]
            for i in range(self.n_splits):
                j = 2 * i + 1
                q_by_node[j] = keras.backend.minimum(q_by_node[j], -q_by_split[:, i])
                q_by_node[j] = keras.backend.minimum(q_by_node[j], q_by_node[i])
                j = 2 * i + 2
                q_by_node[j] = keras.backend.minimum(q_by_node[j], q_by_split[:, i])
                q_by_node[j] = keras.backend.minimum(q_by_node[j], q_by_node[i])
                pass
            q_by_node[0] = 1.0 + 0.0 * q_by_node[-1]
            q_by_node = keras.backend.stack(q_by_node, axis=-1)
            return q_by_node
        if method == 2:
            q_by_split = self.q_layer(x)

            q = 1.0 + q_by_split[:, :1] * 0.0 + keras.backend.cast_to_floatx(keras.backend.arange(self.n_nodes)) * 0.0

            l_children = keras.backend.arange(1, self.n_nodes, 2)
            r_children = keras.backend.arange(2, self.n_nodes, 2)

            l_mask = keras.backend.one_hot(l_children, self.n_nodes)
            l_mask_sum = keras.backend.sum(l_mask, 0)

            r_mask = keras.backend.one_hot(r_children, self.n_nodes)
            r_mask_sum = keras.backend.sum(r_mask, 0)

            vl = keras.backend.minimum(q[:, :self.n_splits], -q_by_split) @ l_mask
            vr = keras.backend.minimum(q[:, :self.n_splits], q_by_split) @ r_mask
            q = (1 - l_mask_sum - r_mask_sum) * q + vl + vr

            for _ in range(self.max_depth):
                vl = keras.backend.minimum(q[:, :self.n_splits], q @ keras.backend.transpose(l_mask)) @ l_mask
                vr = keras.backend.minimum(q[:, :self.n_splits], q @ keras.backend.transpose(r_mask)) @ r_mask
                q = (1 - l_mask_sum - r_mask_sum) * q + vl + vr
                pass
            return q
        pass

    @tf.function
    def call(self, x, *args, **kwargs):
        q_by_node = self._q_compute(x)

        a_candidates = tf.linspace(0.0, 1.0, 21)
        a_candidates = keras.backend.reshape(a_candidates, (1, 1, -1))
        a_candidates_by_node = keras.backend.tile(a_candidates, (1, self.n_nodes, 1))

        # FOR LOOP
        g_a = g(a_candidates_by_node, keras.backend.reshape(q_by_node, (-1, self.n_nodes, 1)), 1.0)
        g_a = keras.backend.reshape(g_a, (self.n_nodes, 1, -1))  # shape: (node, group, candidates)
        node2group = self.idx
        # for
        for _ in range(self.n_nodes + 1):
            n2g = keras.backend.reshape(node2group, (self.n_nodes, self.n_nodes, 1))
            g_a_by_group = keras.backend.sum(n2g * g_a, 0)
            a_by_group = keras.backend.sum(a_candidates_by_node[0] * keras.backend.softmax(-100. * g_a_by_group, axis=1), axis=1)
            a_by_node = keras.backend.reshape(a_by_group, (1, -1)) @ keras.backend.transpose(node2group)
            a_by_node = a_by_node[0]

            a_parent = keras.backend.sum(a_by_node * self.node_to_parent, 1)[1:]
            a_parent = keras.backend.concatenate([a_parent[:1] * 0.0 + 1.0, a_parent], 0)
            violations = a_by_node - a_parent
            if keras.backend.max(violations) <= 1e-8:

                t_max = keras.backend.argmax(violations)
                if t_max > 0:
                    t_max = keras.backend.one_hot(t_max, self.n_nodes)
                    t_max = keras.backend.expand_dims(t_max, 0)

                    p_t_max = t_max @ self.node_to_parent

                    node2group = node2group + keras.backend.transpose(t_max) * (p_t_max - t_max)

        trajectory = keras.backend.clip(q_by_node, 0.0, keras.backend.reshape(a_by_node, (1, self.n_nodes)))

        return trajectory
