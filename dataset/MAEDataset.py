
import torch
import numpy as np
import pandas as pd
import dgl
import os


class MAEDataset():
    def __init__(
            self,
            input: str,
            n_neighbors: int = 30):
        if os.path.exists(input):
            matrix = pd.read_csv(input, index_col=0, header=0)
        elif os.path.exists('./data/inputs/' + input):
            input = './data/inputs/' + input
            matrix = pd.read_csv(input, index_col=0, header=0)

        self.graph, self.node2id = self.matrix_to_graph(matrix, n_neighbors)
        self.num_features = self.graph.ndata["feat"].shape[1]

    def matrix_to_graph(self, matrix, n_neighbors):
        matrix.index = matrix.index.str.upper()
        node2id = dict(zip(matrix.index, range(0, len(matrix.index))))
        matrix = matrix.values
        features = torch.tensor(matrix)

        def dist(matrix):
            square_sum = np.sum(matrix**2, axis=1, keepdims=True)
            inner_product = np.dot(matrix, matrix.T)
            dist_matrix = np.sqrt(np.maximum(square_sum + square_sum.T - 2 * inner_product, 0))
            return dist_matrix
        #TODO 可优化
        dist_matrix = dist(matrix)
        threshold = np.percentile(dist_matrix, 100)
        nearest_indices = np.argsort(dist_matrix)[:, :(n_neighbors + 1)]
        index_0 = []
        index_1 = []
        for i in range(dist_matrix.shape[0]):
            index1 = nearest_indices[i][dist_matrix[i, nearest_indices[i,]] < threshold]
            if len(index1) == 0:
                index1 = np.append(index1, i)
            index_0 += ([i] * len(index1))
            index_1 += (index1.tolist())

        edge_index = torch.tensor([index_0, index_1])
        graph = dgl.graph((edge_index[0], edge_index[1]))

        graph.ndata['feat'] = features.to(dtype=torch.float32)
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        return graph, node2id

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
