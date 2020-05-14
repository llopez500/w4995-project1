import unittest
import numpy as np
from linear_scan_nn import LinearScanNN

class TestLinearScan(unittest.TestCase):

    def test_constructor(self):
        data = np.array([[1,2]])
        linear_scan = LinearScan(data)
        
    def test_add_to_data(self):
        data = np.array([[1,2]])
        linear_scan = LinearScan(data)
        linear_scan.add_to_data(np.array([3,4]))
        np.testing.assert_array_equal(linear_scan.data, np.array([[1,2],[3,4]]), "Should be [[1,2],[3,4]]")

    def test_find_nn(self):
        data = np.array([[1,2],[3,4]])
        linear_scan = LinearScan(data)
        idx, point = linear_scan.find_nn(np.array([3,5]))
        self.assertEqual(idx, 1, "Should be 1")
        np.testing.assert_array_equal(point, np.array([3,4]), "Should be [3,4]")
                         
if __name__ == '__main__':
    unittest.main()