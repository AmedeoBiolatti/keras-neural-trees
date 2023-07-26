import lightgbm
import xgboost

import keras_core as keras
import numpy as np

from keras_core.backend.common.keras_tensor import KerasTensor


class TrueInitializer(keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return keras.ops.ones(shape=shape, dtype="float32") >= 0.0


def my_take(x, i, transpose=False):
    if transpose:
        i = keras.ops.transpose(i)
    out = keras.ops.take(x, i + keras.ops.expand_dims(keras.ops.arange(x.shape[0]), 1) * x.shape[1])
    if transpose:
        out = keras.ops.transpose(out)
    return out


class BaseTrees(keras.layers.Layer):
    def __init__(self,
                 n_estimators: int,
                 max_depth: int = 1,
                 units: int = 1,
                 base_score: float = 0.0,
                 activation=None,
                 grouping=None
                 ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_nodes = 2 ** (max_depth + 1) - 1
        self.units = units

        self.is_leaf = self.add_weight(
            shape=(self.n_estimators, self.n_nodes,),
            initializer=TrueInitializer(),
            dtype="bool",
            trainable=False
        )

        self.col = self.add_weight(
            shape=(self.n_estimators, self.n_nodes,),
            initializer=keras.initializers.Zeros(),
            dtype="int32",
            trainable=False
        )

        self.thr = self.add_weight(
            shape=(self.n_estimators, self.n_nodes,),
            initializer=keras.initializers.HeNormal()
        )

        self.leaf_values = self.add_weight(
            shape=(self.n_estimators, self.n_nodes, self.units),
            initializer=keras.initializers.HeNormal()
        )

        self.base_score = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(base_score)
        )

        self.activation = keras.activations.get(activation)
        self.grouping = grouping

    def _many(self, x):
        i = keras.ops.zeros((keras.ops.shape(x)[0], self.n_estimators), dtype="int32")
        for _ in range(self.max_depth):
            leaf = keras.ops.cast(my_take(self.is_leaf, i, transpose=True), dtype="int32")
            col = my_take(self.col, i, transpose=True)
            x_col = my_take(x, col)
            thr = my_take(self.thr, i, transpose=True)
            p = keras.ops.cast(keras.ops.greater_equal(x_col, thr), dtype="int32")
            i = leaf * i + (1 - leaf) * (2 * i + 2 - p)
        return keras.ops.one_hot(i, self.n_nodes)

    def call(self, x, *args, by_tree=False, **kwargs):
        leaf_assignement = self._many(x)
        out = keras.ops.sum(
            keras.ops.expand_dims(leaf_assignement, axis=-1) *
            keras.ops.expand_dims(self.leaf_values, axis=0),
            axis=-2
        )
        if self.grouping:
            out = keras.ops.reshape(out, (-1, self.n_estimators // self.grouping, self.grouping * self.units))

        if not by_tree:
            out = keras.ops.sum(out, axis=-2)
        out = out + self.base_score

        if self.activation is not None:
            out = self.activation(out)

        return out

    def compute_output_spec(self, x, *args, **kwargs):
        return KerasTensor(shape=(keras.ops.shape(x)[0], self.units), dtype=self.compute_dtype)

    @classmethod
    def from_xgboost(cls, model):
        model_df = model.get_booster().trees_to_dataframe()

        n_estimators = len(np.unique(model_df['Tree']))

        def get_max_depth(df, name) -> int:
            node = df[df['ID'] == name].iloc[0]
            if np.isnan(node['Split']):
                return 0
            return 1 + max([get_max_depth(df, node['Yes']), get_max_depth(df, node['No'])])

        max_depth = max([get_max_depth(model_df, f"{i}-0") for i in np.unique(model_df['Tree'])])

        tree = cls(n_estimators=n_estimators, max_depth=max_depth)
        if model.base_score is None:
            tree.base_score.assign(0.5)
        else:
            tree.base_score.assign(model.base_score)

        def upd(tree: BaseTrees, df):
            for tree_id in df['Tree'].unique():
                _upd(tree, df, tree_id, 0, 0)

        def _upd(tree: BaseTrees, df, tree_id: int, node_id: int, tree_node_id: int):
            node = df[(df['Tree'] == tree_id) & (df['Node'] == node_id)].iloc[0]

            if np.isnan(node['Split']):
                tree.is_leaf[tree_id, tree_node_id].assign(True)
                tree.leaf_values[tree_id, tree_node_id].assign(node['Gain'])
            else:
                col = int(node['Feature'].split('f')[-1])
                tree.is_leaf[tree_id, tree_node_id].assign(False)
                tree.col[tree_id, tree_node_id].assign(col)
                tree.thr[tree_id, tree_node_id].assign(node['Split'])

                _upd(tree, df, tree_id, int(node['No'].split('-')[-1]), tree_node_id * 2 + 1)
                _upd(tree, df, tree_id, int(node['Yes'].split('-')[-1]), tree_node_id * 2 + 2)

        upd(tree, model_df)

        # if binary classification
        if isinstance(model, xgboost.XGBClassifier) and model.n_classes_ == 2:
            tree.base_score.assign(keras.ops.log(tree.base_score / (1 - tree.base_score)))
            tree.activation = keras.activations.sigmoid

        # if multi-classification
        if isinstance(model, xgboost.XGBClassifier) and model.n_classes_ > 2:
            tree.activation = keras.activations.softmax
            tree.grouping = model.n_classes_

        return tree

    @classmethod
    def from_lightgbm(cls, model):
        model_df = model.booster_.trees_to_dataframe()

        n_estimators = len(np.unique(model_df['tree_index']))
        max_depth = model_df['node_depth'].max() - 1

        tree = cls(n_estimators=n_estimators, max_depth=max_depth)

        def upd_all(tree: BaseTrees, df):
            for tree_index in df['tree_index'].unique():
                upd_tree(tree, df, tree_index, f"{tree_index}-S0", 0)

        def upd_tree(tree: BaseTrees, df, tree_index, node_index, i):
            node = df[(df['tree_index'] == tree_index) & (df['node_index'] == node_index)].iloc[0]

            if node['split_feature'] is None:
                tree.is_leaf[tree_index, i].assign(True)
                tree.leaf_values[tree_index, i].assign(node['value'])
            else:
                col = int(node['split_feature'].split('_')[-1])
                tree.is_leaf[tree_index, i].assign(False)
                tree.col[tree_index, i].assign(col)
                tree.thr[tree_index, i].assign(node['threshold'])

                upd_tree(tree, df, tree_index, node['right_child'], i * 2 + 1)
                upd_tree(tree, df, tree_index, node['left_child'], i * 2 + 2)

        upd_all(tree, model_df)

        # if binary classification
        if isinstance(model, lightgbm.LGBMClassifier) and model.n_classes_ == 2:
            # tree.base_score.assign(keras.ops.log(tree.base_score / (1 - tree.base_score)))
            tree.activation = keras.activations.sigmoid

        # if multi-classification
        if isinstance(model, lightgbm.LGBMClassifier) and model.n_classes_ > 2:
            tree.activation = keras.activations.softmax
            tree.grouping = model.n_classes_

        return tree
