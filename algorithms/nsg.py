from nearest_neighbor import NearestNeighbor
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.spatial import distance
import random
# from collections import deque
import networkx as nx


##### NOT DONE YET
#### NEED TO FINISH INMPLEMENTING INTERFACE

class NSG(NearestNeighbor):
    def __init__(self, data, k=5, l=10, m=10):
        super().__init__(data)
        # build KNN Graph
        self.k = k
        self.l = l
        self.m = m
        self.buildKNNGraph(self.data, self.k)
        self.NSGBuild(self.l, self.m)
        
    def buildKNNGraph(self, data, k):
        """build an kNN graph using existing package in scikit-learn
        Inputs:
            - data: the dataset
            - k: the number of nearest neighbor you want to represent in KNN graph
        """
        A = kneighbors_graph(data, k, mode='distance', include_self=False)
        self.G = nx.from_scipy_sparse_matrix(A)
    
    def getNeighbors(self, i):
        """
        Get neighbors of node i
        inputs:
            - i: index of the node
        Outputs:
            - neighbors: index of its neighbor
        """
        return list(self.G.neighbors(i))


    def sortOnDst(self, lst, query):
        """
        Sort the data in list based on its distance to query node q

        Inputs:
            - lst: list of data point
            - q: query node

        Outputs:
            - sorted_lst: sorted S
        """
        dst = [distance.euclidean(self.data[n], query) for n in lst]
        sorted_lst = [lst for _,lst in sorted(zip(dst, lst))]

        return sorted_lst
    
    def searchOnGraph(self, p_index, query, l):
        """Algorithm Search-on-Graph(G,p,q,l)

        Inputs:
            - graph G
            - start nodes p
            - query point q
            - candidate pool size l
        """
#         print("SEARCH ON GRAPH", p_index, query, l)
        i = 0
        S = [p_index]
        checked = set()
        while i < l:
            # i = the index of the first unchecked node in S
            if set(S) == checked:
                return S
            for idx in S:
                if idx not in checked:
                    i = idx
                    break
                    
            # Mark pi as checked
            checked.add(i)

            # Foll all neighbors n of pi in G, add to S
            neighbors = self.getNeighbors(i)

            for neighbor in neighbors:
                if neighbor not in S:
                    S.append(neighbor)

            # sort S in ascending order based on distance to q
            S = self.sortOnDst(S, query)

            # Only preseve the l nearest neighbors
            S = S[:l]
# 
        return S
    
    def findEdge(self, intree, outtree):
        """ Find an edge between in and out."""
        e1 = random.choice(list(intree))
        e2 = self.searchOnGraph(random.choice(list(outtree)), self.data[e1], 1)[0]
        return e1, e2

    def NSGBuild(self, l, m):
        # calculate the centroid c of the dataset.
        k_means = KMeans(n_clusters=1)
        k_means.fit(self.data)
        center = k_means.cluster_centers_

        min_dst = float("Inf")
        c_idx = None
        idx = 0
        for row in self.data:
            dst = distance.euclidean(row, center)
            if dst < min_dst:
                dst = min_dst
                c_idx = idx
            idx += 1

#         print("Find the row number of centroid in dataset {}".format(c_idx))


        # r: generate random node
        r = random.choice(range(len(self.data)))
        # Search-on-Graph(G,r,c,l) %navigating node: searh nearest node of c starting from r
        n = self.searchOnGraph(r, self.data[c_idx], l)

        edges = []

        for v in range(len(self.data)):
            #if v % 10 == 0:
                #print("The {}th node".format(v))
#             print("The {}th node".format(v))



            E = list() # all the nodes checked along the search
            for node in n:
                node_knn = self.searchOnGraph(node, self.data[v], l)
#                 print("node_knn stage")
                E.extend(node_knn) # l nearest neighbors for v
            E = list(set(E))
#             print("E:", E)


            # Sort E in the ascending order of the distance to v

            E = self.sortOnDst(E, self.data[v])
#             print("E", E)

            # result set R=∅,p0 = the closest node to v in E
            R = list()
            p0 = E.pop(0)
            R.append(p0)

            while len(E) > 0 and len(R) < m:
                p = E.pop(0)
                if p == v:
                    continue
                CONFLICT = False
                for r in R:
                    # check if pv conflicts with edge pr
                    dst_pv = distance.euclidean(self.data[p], self.data[v])
                    dst_pr = distance.euclidean(self.data[p], self.data[r])

                    if dst_pr < dst_pv:
                        CONFLICT = True
                        break

                if not CONFLICT:
                    R.append(p)

#             print("R", R)
            for n_r in R:
                edges.append((v, n_r))
#             print("\n")


        # build a tree with edges in NSG from root n with DFS
        while not nx.is_connected(self.G):
#             print("Graph is not connected")
            components = list(nx.connected_components(self.G))
            in_tree = components[0]
            out_tree = components[1:]
            for t in out_tree:
                # establish a link between in_tree and out_tree
                u, v = self.findEdge(in_tree, t)
                self.G.add_edge(u, v)
            
        self.AKNN = self.G
        
#     def retrieveNN(self, node, k):
#         """ Retrieve k NN. """
#         neighbors = [n for n in self.AKNN.neighbors(node)]
#         print(neighbors)
#         knn = neighbors[:k]
#         return knn
    
    def add_to_data(self, point):
        self.data = np.vstack([self.data, point])
        self.buildKNNGraph(self.data, self.k)
        self.NSGBuild(self.l, self.m)
    
    def find_nn(self, query):
        start_idx = np.random.randint(self.data.shape[0])
        neighbor_idx = self.searchOnGraph(start_idx, query, self.l)[0]
#         print("neighbor_idx:", neighbor_idx)
#         print(self.data[neighbor_idx])
        neighbor = self.data[neighbor_idx]
        return (neighbor_idx, neighbor)
        