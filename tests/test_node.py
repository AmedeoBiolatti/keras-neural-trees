import numpy as np
import unittest

from keras_neural_trees import node


class LatentTreeTestSuit(unittest.TestCase):
    units: int = 3

    def setUp(self) -> None:
        self.tree = node.NODE(units=self.units)
        self.x = np.zeros((100, 10))

    def test_output_shape(self):
        z = self.tree(self.x).numpy()
        self.assertEqual(z.ndim, 2)
        self.assertEqual(
            z.shape,
            (self.x.shape[0], self.units)
        )
