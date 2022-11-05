"""Neural oblivious decision trees"""

import tensorflow as tf
from tensorflow import keras

from keras_neural_trees.tree_struct import binary_trees


class NODELayer(keras.layers.Layer):
    def __init__(self,
                 units: int = 1,
                 n_trees: int = 1,
                 depth: int = 4,
                 *,
                 return_by_tree: bool = False,
                 return_by_leaf: bool = False
                 ):
        super().__init__()
        self.units: int = units
        self.n_trees = n_trees
        self.depth = depth
        # TODO : replace with oblivious trees
        self.s2l = binary_trees.split_to_node_descendants_matrix(self.depth, leaves_only=True).astype('float32')
        self.n_splits, self.n_leaves = self.s2l.shape
        self.s2l = keras.backend.reshape(self.s2l, (1, 1, self.n_splits, self.n_leaves))

        self.F = None
        self.b = None
        self.leaves_values = None

        self.return_by_tree: bool = return_by_tree
        self.return_by_leaf: bool = return_by_leaf
        pass

    def build(self, input_shape):
        assert len(input_shape) == 2
        n_cols = input_shape[-1]
        self.F = self.add_weight(
            name="F",
            shape=(1, n_cols, self.n_trees, self.n_splits),
            initializer=keras.initializers.RandomUniform(0.0, 1.0)
        )
        self.b = self.add_weight(
            name="bias",
            shape=(1, self.n_trees, self.n_splits),
            initializer=keras.initializers.RandomNormal(0.0, 0.1)
        )
        self.leaves_values = self.add_weight(
            name="leaves_values",
            shape=(1, self.n_trees, self.n_leaves, self.units),
            initializer=keras.initializers.VarianceScaling(scale=2.0, mode="fan_out")
        )

    def call(self, x, *args, **kwargs):
        x_ = keras.backend.expand_dims(keras.backend.expand_dims(x, axis=-1), axis=-1)

        e = keras.activations.softmax(self.F, axis=1)
        q_by_split = keras.backend.sum(e * x_, axis=1) - self.b

        abs_s2l = keras.backend.abs(self.s2l)
        q = keras.activations.sigmoid(keras.backend.expand_dims(q_by_split, -1) * self.s2l) * abs_s2l + 1 * (1 - abs_s2l)
        mask = keras.backend.expand_dims(keras.backend.prod(q, axis=2), axis=-1)

        axis = ()
        if not self.return_by_tree:
            axis = axis + (1,)
        if not self.return_by_leaf:
            axis = axis + (2,)

        values = mask * self.leaves_values
        if len(axis) > 0:
            values = keras.backend.sum(values, axis=axis)

        return values

    pass


class NODE(keras.layers.Layer):
    def __init__(
            self,
            units: int = 1,
            n_layers: int = 1,
            n_trees_per_layer: int = 1,
            depth: int = 4,
            name=None
    ):
        super().__init__(name=name)
        self.node_layers = [
            NODELayer(units=units, n_trees=n_trees_per_layer, depth=depth) for _ in range(n_layers)
        ]
        pass

    def call(self, x, *args, **kwargs):
        z = x
        out = 0.0
        for node_layer in self.node_layers:
            node_layer_out = node_layer(z, *args, **kwargs)
            out = out + node_layer_out
            z = keras.backend.concatenate([z, node_layer_out], axis=-1)

        return out
