import keras_core as keras

from keras_neural_trees.base_tree import BaseTrees
from keras_neural_trees.tree_struct import binary_trees_struct


class QTrees(keras.layers.Layer):
    def __init__(
            self,
            n_estimators: int = 1,
            max_depth: int = 1,
            units: int = 1,
            base_score: float = 0.0,
            activation=None,
            grouping=None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.units = units
        self.base_score = base_score

        split_to_node = binary_trees_struct.split_to_node_descendants_matrix(self.max_depth + 1, signed=True)
        self.n_splits, self.n_leaves = split_to_node.shape
        split_to_node = split_to_node.reshape(-1, self.n_splits, self.n_leaves)
        self.split_to_node = keras.ops.convert_to_tensor(split_to_node, dtype=self.compute_dtype)

        self.pre_activation = keras.activations.hard_sigmoid
        self.activation = keras.activations.get(activation)
        self.grouping = grouping

        self.t = 100.0

    def build(self, input_shape):
        super().build(input_shape)
        self.dense = keras.layers.Dense(self.n_estimators * self.n_splits)
        self.dense.build(input_shape)

        self.leaf_values = self.add_weight(
            shape=(1, self.n_estimators, self.n_leaves, self.units),
            initializer=keras.initializers.HeNormal()
        )

    def q(self, x):
        q_by_split = self.dense(x)
        q_by_split = keras.ops.reshape(q_by_split, (-1, self.n_estimators, self.n_splits, 1))
        q_by_split_and_node = q_by_split * self.split_to_node + 1.0 - keras.ops.abs(self.split_to_node)
        q_by_node = keras.ops.min(q_by_split_and_node, axis=-2)
        return q_by_node

    def call(self, x, *, by_tree=False):
        q = self.t * self.q(x) + 0.5
        leaf_assignement = self.pre_activation(q)

        # pred by-tree
        out = keras.ops.sum(
            keras.ops.reshape(leaf_assignement, (-1, self.n_estimators, self.n_leaves, 1)) *
            self.leaf_values,
            axis=-2
        )

        if self.grouping:
            out = keras.ops.reshape(out, (-1, self.n_trees // self.grouping, self.grouping * self.units))

        if not by_tree:
            out = keras.ops.sum(out, axis=-2)
        out = out + self.base_score

        if self.activation is not None:
            out = self.activation(out)

        return out

    @classmethod
    def from_base_trees(cls, base_trees: BaseTrees, input_shape) -> 'QTrees':
        model = cls(n_estimators=base_trees.n_estimators, max_depth=base_trees.max_depth)
        model.build(input_shape)
        model.dense.kernel.assign(0.0 * model.dense.kernel)
        model.dense.bias.assign(0.0 * model.dense.bias)
        model.leaf_values.assign(0.0 * model.leaf_values)
        model.base_score = base_trees.base_score  # TODO assign

        for tree_index in range(model.n_estimators):
            cols = base_trees.col[tree_index, :model.n_splits]
            for split_idx, col in enumerate(cols):
                for node_idx in range(model.n_leaves):
                    v = model.split_to_node[0, split_idx, node_idx].numpy()
                    if v != 0.0:
                        model.dense.kernel[col, split_idx + tree_index * model.n_splits].assign(-1.0)

        model.dense.bias.assign(keras.ops.ravel(base_trees.thr[:, :model.n_splits]))
        model.leaf_values.assign(keras.ops.reshape(base_trees.leaf_values, model.leaf_values.shape))
        model.leaf_values.assign(
            model.leaf_values *
            keras.ops.cast(keras.ops.reshape(base_trees.is_leaf, model.leaf_values.shape), model.compute_dtype)
        )

        model.dense.kernel.trainable = False

        return model
