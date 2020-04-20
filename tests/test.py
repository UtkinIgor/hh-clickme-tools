import unittest
from utils import minmax_scale, get_stats, softmax, get_best_banner
import numpy as np

class test_get_banner(unittest.TestCase):

    def test_minmax_scaler(self):
        self.assertEqual(minmax_scale([1, 1, 1]), [0, 0, 0], 'not zero list')
        self.assertEqual(minmax_scale([0, 0, 0]), [0, 0, 0], 'not zero list')
        self.assertEqual(minmax_scale([1, 2, 3]).tolist(), [0., 0.5, 1.], 'wrong calculate')
        self.assertEqual(minmax_scale([0, 2, 2, 10]).tolist(), [0., 0.2, 0.2, 1.], 'wrong calculate')
        self.assertIsInstance(minmax_scale([1, 2, 3]), np.ndarray, 'result is not np.ndarray')
        self.assertGreaterEqual(np.min(minmax_scale([1, 20, 300])), 0, 'min value less than zero')
        self.assertLessEqual(np.min(minmax_scale([1, 20, 300])), 1, 'max value more than 1')

    def test_get_stats(self):
        self.assertIsInstance(get_stats({'a': [10, 2, 0, 0]}), list, 'result is not list')
        self.assertEqual(get_stats({'a': [10, 2, 0, 0]})[0], 0.2, 'wrong compute')
        self.assertEqual(get_stats({'a': [20, 5, 0, 0]})[0], 0.25, 'wrong compute')
        self.assertEqual(get_stats({'a': [0, 0, 0, 0]})[0], 0, 'wrong compute')


    def test_softmax(self):
        self.assertIsInstance(softmax([3, 3, 3, 3]), np.ndarray, 'result is not np.ndarray')
        self.assertEqual(softmax([3, 3, 3, 3]).tolist(), [.25, .25, .25, .25], 'wrong compute')
        self.assertEqual(softmax([3000, 3000]).tolist(), [.5, .5], 'wrong compute')


    def test_get_banner(self):
        self.assertIsInstance(get_best_banner({'a': [10, 5, 0, 0], 'b': [6, 2, 0, 0], 'c': [20, 2, 0, 0]}), list, 'result is not list')
        self.assertEqual(len(get_best_banner({'a': [10, 5, 0, 0], 'b': [6, 2, 0, 0], 'c': [20, 2, 0, 0]})), 3, 'wrong default arg top')
        self.assertEqual(len(get_best_banner({'a': [10, 5, 0, 0], 'b': [6, 2, 0, 0], 'c': [20, 2, 0, 0]}, top=1)), 1, 'len result is not 1')

if __name__ == '__main__':
    unittest.main()