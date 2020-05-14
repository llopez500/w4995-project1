from nearest_neighbor import NearestNeighbor
import numpy as np

class RandomNeighbor(NearestNeighbor):
    def add_to_data(self, point):
        self.data = np.vstack([self.data, point])
    
    def find_nn(self, query):
        idx = np.random.randint(self.data.shape[0])
        return (idx,self.data[idx])