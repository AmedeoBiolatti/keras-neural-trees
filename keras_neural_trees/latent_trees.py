import keras_core as keras
import numpy as np

from .tree_struct import binary_trees_struct


def g(a, q, reg_lambda: float):
    reg_part = keras.ops.sum(a ** 2, axis=0)
    sqr_part = (a - (q + 0.5)) ** 2 * keras.ops.cast(a <= q + 0.5, dtype=a.dtype)
    sqr_part = keras.ops.sum(sqr_part, axis=0)
    return 0.5 * (reg_lambda * reg_part + sqr_part)


class LatentTree(keras.layers.Layer):
    """
    Latent trees as described in 'Learning Binary Decision Trees by Argmin Differentiation'
    The inner optimization loop is a little different, using softmax and set of candidate solutions instead of exact solving
    https://arxiv.org/pdf/2010.04627.pdf
    """

    def __init__(self,
                 max_depth: int,
                 reg_lambda: float = 0.0,
                 n_activation_candidates: int = 21,
                 method: int = 0
                 ):
        super(LatentTree, self).__init__()

        self.reg_lambda: float = reg_lambda
        self.max_depth: int = max_depth

        split_to_node = binary_trees_struct.split_to_node_descendants_matrix(max_depth, signed=True)
        self.n_splits, self.n_nodes = split_to_node.shape
        split_to_node = split_to_node.reshape(-1, self.n_splits, self.n_nodes)
        self.split_to_node = keras.ops.convert_to_tensor(split_to_node, dtype=self.compute_dtype)

        self.q_layer = keras.layers.Dense(
            units=self.n_splits,
            kernel_initializer=keras.initializers.VarianceScaling(scale=2.0, mode="fan_in"),
            bias_initializer=keras.initializers.RandomNormal()
        )

        node_idx = keras.ops.arange(self.n_nodes, dtype="int32")
        parent_idx = (node_idx - 1) // 2

        self.node_to_parent = np.zeros((self.n_nodes, self.n_nodes))
        self.node_to_parent[node_idx[1:], parent_idx[1:]] = 1.0
        self.idx = keras.ops.eye(self.n_nodes)

        self.q_by_node = keras.ops.ones((1, self.n_nodes))

        self.tol: float = 1e-8
        self.inv_temp: float = 100.0
        self.n_activation_candidates: int = n_activation_candidates

        self.method: int = method

    def _q_compute(self, x):
        method = self.method
        if method == 0:
            q_by_split = self.q_layer(x)
            q_by_split = keras.ops.reshape(q_by_split, (-1, self.n_splits, 1))
            q_by_split_and_node = q_by_split * self.split_to_node + 1.0 - keras.ops.abs(self.split_to_node)
            q_by_node = keras.ops.min(q_by_split_and_node, axis=1)
            return q_by_node
        if method == 1:
            q_by_split = self.q_layer(x)
            q_by_node = [1.0 for _ in range(self.n_nodes)]
            for i in range(self.n_splits):
                j = 2 * i + 1
                q_by_node[j] = keras.ops.minimum(q_by_node[j], -q_by_split[:, i])
                q_by_node[j] = keras.ops.minimum(q_by_node[j], q_by_node[i])
                j = 2 * i + 2
                q_by_node[j] = keras.ops.minimum(q_by_node[j], q_by_split[:, i])
                q_by_node[j] = keras.ops.minimum(q_by_node[j], q_by_node[i])
                pass
            q_by_node[0] = 1.0 + 0.0 * q_by_node[-1]
            q_by_node = keras.ops.stack(q_by_node, axis=-1)
            return q_by_node
        if method == 2:
            q_by_split = self.q_layer(x)

            q = 1.0 + q_by_split[:, :1] * 0.0 + keras.ops.cast(keras.ops.arange(self.n_nodes), self.compute_dtype) * 0.0

            l_children = keras.ops.arange(1, self.n_nodes, 2)
            r_children = keras.ops.arange(2, self.n_nodes, 2)

            l_mask = keras.ops.one_hot(l_children, self.n_nodes)
            l_mask_sum = keras.ops.sum(l_mask, 0)

            r_mask = keras.ops.one_hot(r_children, self.n_nodes)
            r_mask_sum = keras.ops.sum(r_mask, 0)

            vl = keras.ops.minimum(q[:, :self.n_splits], -q_by_split) @ l_mask
            vr = keras.ops.minimum(q[:, :self.n_splits], q_by_split) @ r_mask
            q = (1 - l_mask_sum - r_mask_sum) * q + vl + vr

            for _ in range(self.max_depth):
                vl = keras.ops.minimum(q[:, :self.n_splits], q @ keras.ops.transpose(l_mask)) @ l_mask
                vr = keras.ops.minimum(q[:, :self.n_splits], q @ keras.ops.transpose(r_mask)) @ r_mask
                q = (1 - l_mask_sum - r_mask_sum) * q + vl + vr
                pass
            return q
        pass

    def call(self, x, *args, **kwargs):
        q_by_node = self._q_compute(x)

        a_candidates = keras.ops.linspace(0.0, 1.0, self.n_activation_candidates, dtype=self.compute_dtype)
        a_candidates = keras.ops.reshape(a_candidates, (1, 1, -1))
        a_candidates_by_node = keras.ops.tile(a_candidates, (1, self.n_nodes, 1))

        # FOR LOOP
        g_a = g(a_candidates_by_node, keras.ops.reshape(q_by_node, (-1, self.n_nodes, 1)), 1.0)
        g_a = keras.ops.reshape(g_a, (self.n_nodes, 1, -1))  # shape: (node, group, candidates)
        node2group = self.idx
        # for
        for _ in range(self.n_nodes + 1):
            n2g = keras.ops.reshape(node2group, (self.n_nodes, self.n_nodes, 1))
            g_a_by_group = keras.ops.sum(n2g * g_a, 0)
            a_by_group = keras.ops.sum(
                a_candidates_by_node[0] * keras.ops.softmax(-self.inv_temp * g_a_by_group, axis=1), axis=1)
            a_by_node = keras.ops.reshape(a_by_group, (1, -1)) @ keras.ops.transpose(node2group)
            a_by_node = a_by_node[0]

            a_parent = keras.ops.sum(a_by_node * self.node_to_parent, 1)[1:]
            a_parent = keras.ops.concatenate([a_parent[:1] * 0.0 + 1.0, a_parent], 0)
            violations = a_by_node - a_parent

            def update_node2group():
                t_max = keras.ops.argmax(violations)
                t_max = keras.ops.one_hot(t_max, self.n_nodes)
                t_max = keras.ops.expand_dims(t_max, 0)

                p_t_max = t_max @ self.node_to_parent

                n2g = node2group + keras.ops.transpose(t_max) * (p_t_max - t_max)
                return n2g

            node2group = keras.ops.cond(
                keras.ops.logical_and(
                    keras.ops.max(violations) <= self.tol,
                    keras.ops.argmax(violations) > 0),
                update_node2group,
                lambda: node2group
            )

        trajectory = keras.ops.clip(q_by_node, 0.0, keras.ops.reshape(a_by_node, (1, self.n_nodes)))

        return trajectory
