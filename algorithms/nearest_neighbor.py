import numpy as np

# Base class
class NearestNeighbor:
    def __init__(self, data):
        assert(type(data) == np.ndarray)
        self.data = data
        
    def add_to_data(self, point):
        """Return None

        Add a new point to the dataset
        """
        raise ValueError('Method[add_to_data] not implemented.')
      
    
    def find_nn(self, query):
        """Return (index, np.array)

        Query closest neighbor in the dataset
        """
        raise ValueError('Method[find_nn] not implemented.')