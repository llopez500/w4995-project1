from nearest_neighbor import NearestNeighbor
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from scipy.spatial import distance

class RandomProjectionNeighbor(NearestNeighbor):
    def __init__(self, data, output_dim=1):
        super().__init__(data)
        self.output_dim = output_dim
        self.randproj = GaussianRandomProjection(n_components=output_dim)
        self.new_data = self.randproj.fit_transform(self.data)
    
    def add_to_data(self, point):
        self.data = np.vstack([self.data, point])
        self.randproj = GaussianRandomProjection(n_components=self.output_dim)
        self.new_data = self.randproj.fit_transform(self.data)
    
    def find_nn(self, query):
        new_query = self.randproj.fit_transform(query.reshape(1,-1))
        idx = distance.cdist(new_query, self.new_data).argmin()
        return (idx, self.data[idx])