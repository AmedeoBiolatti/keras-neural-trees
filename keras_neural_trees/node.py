"""Neural oblivious decision trees"""

import keras_core as keras

from keras_neural_trees.tree_struct import binary_trees_struct
from keras_neural_trees.tree_struct import oblivious_trees_struct


class NODELayer(keras.layers.Layer):
    def __init__(self,
                 units: int = 1,
                 n_trees: int = 1,
                 depth: int = 4,
                 *,
                 oblivious: bool = True,
                 return_by_tree: bool = False,
                 return_by_leaf: bool = False
                 ):
        super().__init__()
        self.units: int = units
        self.n_trees = n_trees
        self.depth = depth
        if oblivious:
            self.s2l = oblivious_trees_struct.split_to_node_descendants_matrix(self.depth, leaves_only=True)
        else:
            self.s2l = binary_trees_struct.split_to_node_descendants_matrix(self.depth, leaves_only=True)
        self.n_splits, self.n_leaves = self.s2l.shape
        self.s2l = keras.ops.reshape(self.s2l.astype('float32'), (1, 1, self.n_splits, self.n_leaves))

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
        x_ = keras.ops.expand_dims(keras.ops.expand_dims(x, axis=-1), axis=-1)

        e = keras.activations.softmax(self.F, axis=1)
        q_by_split = keras.ops.sum(e * x_, axis=1) - self.b

        abs_s2l = keras.ops.abs(self.s2l)
        q = keras.activations.sigmoid(keras.ops.expand_dims(q_by_split, -1) * self.s2l) * abs_s2l + (1 - abs_s2l)
        mask = keras.ops.expand_dims(keras.ops.prod(q, axis=2), axis=-1)

        axis = ()
        if not self.return_by_tree:
            axis = axis + (1,)
        if not self.return_by_leaf:
            axis = axis + (2,)

        values = mask * self.leaves_values
        if len(axis) > 0:
            values = keras.ops.sum(values, axis=axis)

        return values

    pass


class NODE(keras.layers.Layer):
    def __init__(
            self,
            units: int = 1,
            n_layers: int = 1,
            n_trees_per_layer: int = 1,
            depth: int = 4,
            oblivious: bool = True,
            name=None
    ):
        super().__init__(name=name)
        self.node_layers: list[NODELayer] = [
            NODELayer(units=units, n_trees=n_trees_per_layer, depth=depth, oblivious=oblivious) for _ in range(n_layers)
        ]
        pass

    def call(self, x, *args, **kwargs):
        z = x
        out = 0.0
        for node_layer in self.node_layers:
            node_layer_out = node_layer(z, *args, **kwargs)
            out = out + node_layer_out
            z = keras.ops.concatenate([z, node_layer_out], axis=-1)
        return out
