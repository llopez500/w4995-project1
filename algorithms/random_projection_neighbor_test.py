import unittest
import numpy as np
from random_projection_neighbor import RandomProjectionNeighbor

class TestRandomProjectionNeighbor(unittest.TestCase):

    def test_constructor(self):
        data = np.array([[1,2]])
        random_projection_neighbor = RandomProjectionNeighbor(data)
        
    def test_add_to_data(self):
        data = np.array([[1,2]])
        random_projection_neighbor = RandomProjectionNeighbor(data)
        random_projection_neighbor.add_to_data(np.array([3,4]))
        np.testing.assert_array_equal(random_projection_neighbor.data, np.array([[1,2],[3,4]]), "Should be [[1,2],[3,4]]")

    def test_find_nn(self):
        data = np.array([[1,2]])
        random_projection_neighbor = RandomProjectionNeighbor(data)
        idx, point = random_projection_neighbor.find_nn(np.array([0,0]))
        self.assertEqual(idx, 0, "Should be 0")
        np.testing.assert_array_equal(point, np.array([1,2]), "Should be [1,2]")
                         
if __name__ == '__main__':
    unittest.main()