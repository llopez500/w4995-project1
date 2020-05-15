import unittest
import numpy as np
from nsg import NSG

class TestNSG(unittest.TestCase):

    def test_constructor(self):
        data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
        k=3
        l=3
        m=3
        nsg = NSG(data, k ,l, m)
        
    def test_add_to_data(self):
        data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
        k=1
        l=1
        m=1
        nsg = NSG(data, k ,l, m)
        nsg.add_to_data(np.array([13,14]))
        np.testing.assert_array_equal(nsg.data, np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]), "Should be [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]")

    def test_find_nn(self):
        data = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
        k=4
        l=5
        m=5
        nsg = NSG(data, k ,l, m)
        idx, point = nsg.find_nn(np.array([0,0]))
        self.assertEqual(idx, 0, "Should be 0")
        np.testing.assert_array_equal(point, np.array([1,2]), "Should be [1,2]")
                         
if __name__ == '__main__':
    unittest.main()