from nearest_neighbor import NearestNeighbor
import numpy as np
from scipy.spatial import distance

class LinearScan(NearestNeighbor):
    def add_to_data(self, point):
        self.data = np.vstack([self.data, point])
    
    def find_nn(self, query):
        idx = distance.cdist([query], self.data).argmin()
        return (idx,self.data[idx])