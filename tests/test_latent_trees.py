import numpy as np
from tensorflow import keras

import unittest

from keras_neural_trees import latent_trees


class LatentTreeTestSuit(unittest.TestCase):
    depth: int = 4

    def setUp(self) -> None:
        keras.backend.clear_session()
        self.tree = latent_trees.LatentTree(self.depth)
        self.x = np.zeros((100, 10))

    def test_output_shape(self):
        z = self.tree(self.x).numpy()
        self.assertEqual(z.ndim, 2)
        self.assertEqual(
            z.shape[0],
            self.x.shape[0]
        )
        self.assertEqual(
            z.shape[1],
            2 ** self.depth - 1
        )

    def test_output_value_range(self):
        z = self.tree(self.x).numpy()
        self.assertGreaterEqual(z.min(), 0.0)
        self.assertLessEqual(z.max(), 1.0)

    def test_output_row_sum(self):
        z = self.tree(self.x).numpy()
        z_row_sum = z.sum(1)
        self.assertLessEqual(
            z_row_sum.max(),
            self.depth,
            "The sum per level should be no more than one"
        )
        self.assertGreaterEqual(
            z_row_sum.max(),
            1.0,
            "At least root node"
        )
