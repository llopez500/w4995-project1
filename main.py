import random
from scipy.spatial import distance
import numpy as np
from collections import deque
import networkx as nx

from sklearn.neighbors import kneighbors_graph
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

class NSG:
    def __init__(self, data = None, k = 10):
        """
        Initialize NSG Algorithm

        Inputs:
            - data: the dataset
            - k: the number of nearest neighbor you want to represent in KNN graph
        """

        print("Initalize NSG Algorithm")

        if data == None:
            # If no data is provided, using default dataset IRIS
            print("Using default dataset Iris")
            self.data = load_iris()['data']
              
        else:
            self.data = data

        # build KNN Graph
        self.G = self.buildKNNGraph(self.data, k)


    def buildKNNGraph(self, data, k):
        """build an kNN graph using existing package in scikit-learn
        Inputs:
            - data: the dataset
            - k: the number of nearest neighbor you want to represent in KNN graph
        """
        G = kneighbors_graph(data, k, mode='distance', include_self=False)

        return G

    def getNeighbors(self, i):
        """
        Get neighbors of node i
        inputs:
            - i: index of the node
        Outputs:
            - neighbors: index of its neighbor
        """
        return np.nonzero(self.G[i].todense()[0])[1]


    def sortOnDst(self, lst, q):
        """
        Sort the data in list based on its distance to query node q

        Inputs:
            - lst: list of data point
            - q: query node

        Outputs:
            - sorted_lst: sorted S
        """
        q_node = self.data[q]
        dst = [distance.euclidean(self.data[n], q_node) for n in lst]
        sorted_lst = [lst for _,lst in sorted(zip(dst, lst))]

        return sorted_lst




    def searchOnGraph(self, p, q, l):
        """Algorithm Search-on-Graph(G,p,q,l)

        Inputs:
            - graph G
            - start nodes p
            - query point q
            - candidate pool size l
        """
        i = 0
        S = list([p])
        checked = set()

        while i < l:
            for idx in range(len(S)):
                if idx in checked:
                    continue

            #print("Check neighbor of this node", node)
            neighbors = self.getNeighbors(idx)
            #print(neighbors)

            checked.add(idx)

            for neighbor in neighbors:
                if neighbor not in checked:
                    S.append(neighbor)

            # sort S in ascending order based on distance to
            S = list(set(S))
            S = self.sortOnDst(S, q)


            # Only preseve the l nearest neighbors
            if len(S) > l:
                S = S[:l]

            S = deque(S)
            i += 1


        return S

    def findEdge(self, intree, outtree):
        """ Find an edge between in and out."""
        e1 = random.choice(list(intree))
        e2 = self.searchOnGraph(outtree, random.choice(list(intree)), 1)[0]
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

        print("Find the row number of centroid in dataset {}".format(c_idx))


        # r: generate random node
        r = random.choice(range(len(self.data)))

        # Search-on-Graph(G,r,c,l) %navigating node: searh nearest node of c starting from r
        n = self.searchOnGraph(r, c_idx, l)
        print(n)

        edges = []

        for v in range(len(self.data)):
            #if v % 10 == 0:
                #print("The {}th node".format(v))
            print("The {}th node".format(v))



            E = list() # all the nodes checked along the search
            for node in n:
                node_knn = self.searchOnGraph(node, v, l)
                E.extend(node_knn) # l nearest neighbors for v
            E = list(set(E))


            # Sort E in the ascending order of the distance to v

            E = self.sortOnDst(E, v)
            E = deque(E)
            print("E", E)

            # result set R=∅,p0 = the closest node to v in E
            R = list()
            p0 = E.popleft()
            R.append(p0)

            while len(E) > 0 and len(R) < m:
                p = E.popleft()
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

            print("R", R)
            for n_r in R:
                edges.append((v, n_r))
            print("\n")


        # build a tree with edges in NSG from root n with DFS
        G=nx.Graph()
        G.add_nodes_from(range(len(self.data)))
        G.add_edges_from(edges)
        while not nx.is_connected(G):
            print("Graph is not connected")
            components = list(nx.connected_components(G))
            in_tree = components[0]
            out_tree = components[1:]
            for t in out_tree:
                # establish a link between in_tree and out_tree
                u, v = self.findEdge(in_tree, t)
                G.add_edge(u, v)

        self.AKNN = G

    def retrieveNN(self, node, k):
        """ Retrieve k NN. """
        neighbors = [n for n in self.AKNN.neighbors(node)]
        print(neighbors)
        knn = neighbors[:k]
        return knn



if __name__ == "__main__":
    n = NSG(k=5)
    n.NSGBuild(l=10, m=10)
    knn = n.retrieveNN(node=1, k=3)
    print(knn)
