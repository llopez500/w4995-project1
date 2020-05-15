import unittest
import numpy as np
from random_neighbor import RandomNeighbor

class TestRandomNeighbor(unittest.TestCase):

    def test_constructor(self):
        data = np.array([[1,2]])
        random_neighbor = RandomNeighbor(data)
        
    def test_add_to_data(self):
        data = np.array([[1,2]])
        random_neighbor = RandomNeighbor(data)
        random_neighbor.add_to_data(np.array([3,4]))
        np.testing.assert_array_equal(random_neighbor.data, np.array([[1,2],[3,4]]), "Should be [[1,2],[3,4]]")

    def test_find_nn(self):
        data = np.array([[1,2]])
        random_neighbor = RandomNeighbor(data)
        idx, point = random_neighbor.find_nn(np.array([0,0]))
        self.assertEqual(idx, 0, "Should be 0")
        np.testing.assert_array_equal(point, np.array([1,2]), "Should be [1,2]")
                         
if __name__ == '__main__':
    unittest.main()