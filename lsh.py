
from __future__ import print_function
from nearest_neighbor import NearestNeighbor
from sklearn.datasets import load_iris
import numpy as np
import falconn
import timeit
import math

#MAKE THIS A FUNCTION!!!
class LSH(NearestNeighbor):
    def __init__(self, dataset):

        number_of_queries = 10
        # we build only 50 tables, increasing this quantity will improve the query time
        # at a cost of slower preprocessing and larger memory footprint, feel free to
        # play with this number
        number_of_tables = 50

        params_cp = falconn.LSHConstructionParameters()

        params_cp.dimension = len(dataset[0])
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        # we set one rotation, since the data is dense enough,
        # for sparse data set it to 2
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        # we want to use all the available threads to set up
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        self.params_cp = params_cp

        # we build 18-bit hashes so that each table has
        # 2^18 bins; this is a good choise since 2^18 is of the same
        # order of magnitude as the number of data points
        falconn.compute_number_of_hash_functions(18, params_cp)

        print('Constructing the LSH table')
        self.table = falconn.LSHIndex(params_cp)
        self.table.setup(dataset)
        self.data = dataset
        self.query_object = self.table.construct_query_object()



        #super().__init__(data)

        #self.output_dim = output_dim
        #self.randproj = GaussianRandomProjection(n_components=output_dim)
        #self.new_data = self.randproj.fit_transform(self.data)


    def add_to_data(self, point):
        """Return None

        Add a new point to the dataset
        """
        falconn.compute_number_of_hash_functions(18, self.params_cp)

        print('Constructing the LSH table')
        self.table = falconn.LSHIndex(self.params_cp )
        self.data = np.vstack([self.data, point])
        self.table.setup(self.data)


    def find_nn(self, query):
        """Return (index, np.array)

        Query closest neighbor in the dataset
        """
        return self.query_object.find_nearest_neighbor(query)

if __name__ == '__main__':
    data = load_iris()['data']
    lsh = LSH(data)
    point = np.array([4.7,3.2,1.3,0.3])
    lsh.add_to_data(point)
    neighbors = lsh.find_nn(point)
    print(neighbors)


    """number_of_queries = 10
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 50

    print('Reading the dataset')
    #CHANGE THE DATASET!!!
    dataset = load_iris()['data']
    print('Done')

    print(type(dataset[0][0]))

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    dataset = np.float32(dataset)
    assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.
    print('Normalizing the dataset')
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    print('Done')

    # MAKE CHANGES HERE!!! Choose random data points to be queries.
    print('Generating queries')
    np.random.seed(4057218)
    np.random.shuffle(dataset)
    queries = dataset[len(dataset) - number_of_queries:]
    print(dataset)
    dataset = dataset[:len(dataset) - number_of_queries]
    print(dataset)
    print('Done')

    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for query in queries:
        answers.append(np.dot(dataset, query).argmax())
    t2 = timeit.default_timer()
    print('Done')
    print('Linear scan time: {} per query'.format((t2 - t1) / float(
        len(queries))))

    # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    print('Centering the dataset and queries')
    center = np.mean(dataset, axis=0)
    dataset -= center
    queries -= center
    print('Done')

    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(dataset[0])
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = number_of_tables
    # we set one rotation, since the data is dense enough,
    # for sparse data set it to 2
    params_cp.num_rotations = 1
    params_cp.seed = 5721840
    # we want to use all the available threads to set up
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # we build 18-bit hashes so that each table has
    # 2^18 bins; this is a good choise since 2^18 is of the same
    # order of magnitude as the number of data points
    falconn.compute_number_of_hash_functions(18, params_cp)

    print('Constructing the LSH table')
    t1 = timeit.default_timer()
    table = falconn.LSHIndex(params_cp)
    table.setup(dataset)
    t2 = timeit.default_timer()
    print('Done')
    print('Construction time: {}'.format(t2 - t1))

    query_object = table.construct_query_object()

    # find the smallest number of probes to achieve accuracy 0.9
    # using the binary search
    print('Choosing number of probes')
    number_of_probes = number_of_tables

    def evaluate_number_of_probes(number_of_probes):
        query_object.set_num_probes(number_of_probes)
        score = 0
        for (i, query) in enumerate(queries):
            if answers[i] in query_object.get_candidates_with_duplicates(
                    query):
                score += 1
        return float(score) / len(queries)

    while True:
        accuracy = evaluate_number_of_probes(number_of_probes)
        print('{} -> {}'.format(number_of_probes, accuracy))
        if accuracy >= 0.9:
            break
        number_of_probes = number_of_probes * 2
    if number_of_probes > number_of_tables:
        left = number_of_probes // 2
        right = number_of_probes
        while right - left > 1:
            number_of_probes = (left + right) // 2
            accuracy = evaluate_number_of_probes(number_of_probes)
            print('{} -> {}'.format(number_of_probes, accuracy))
            if accuracy >= 0.9:
                right = number_of_probes
            else:
                left = number_of_probes
        number_of_probes = right
    print('Done')
    print('{} probes'.format(number_of_probes))

    # final evaluation
    t1 = timeit.default_timer()
    score = 0
    for (i, query) in enumerate(queries):
        if query_object.find_nearest_neighbor(query) == answers[i]:
            score += 1
    t2 = timeit.default_timer()

    print('Query time: {}'.format((t2 - t1) / len(queries)))
    print('Precision: {}'.format(float(score) / len(queries)))"""
