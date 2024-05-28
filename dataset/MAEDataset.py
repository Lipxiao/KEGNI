from posixpath import join
from sys import path
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
import pandas as pd
from torch_cluster import knn
import dgl
import random
import scanpy as sc
from utils import parser_args
import os 
import math
from scipy.stats import spearmanr

class MAEDataset():

    def __init__(
        self,
        input:str,
        n_neighbors:int = 30):
        
        if os.path.exists(input):
            matrix = pd.read_csv(input, index_col = 0, header = 0)
        elif os.path.exists('./data/inputs/' + input):
            input = './data/inputs/' +input
            matrix = pd.read_csv(input, index_col = 0, header = 0)
            
        self.graph,self.node2id = self.matrix_to_graph(matrix,n_neighbors)
        self.num_features = self.graph.ndata["feat"].shape[1]

    @staticmethod
    def matrix_to_graph(matrix,n_neighbors):    
        matrix.index = matrix.index.str.upper()
        genes  = list(matrix.index)
        node2id = dict(zip(genes, range(0,len(genes))))
        matrix = matrix.values
        features = matrix
        features = torch.tensor(features)

        def dist(matrix):
            # 计算每个点的平方和
            square_sum = np.sum(matrix**2, axis=1, keepdims=True)
            # 计算两个点的内积
            inner_product = np.dot(matrix, matrix.T)
            # 计算距离矩阵
            # dist_matrix = np.sqrt(square_sum + square_sum.T - 2 * inner_product)
            dist_matrix = np.sqrt(np.maximum(square_sum + square_sum.T - 2 * inner_product, 0))
            return dist_matrix

        dist_matrix = dist(matrix)
        threshold = np.percentile(dist_matrix, 100)    
        nearest_indices = np.argsort(dist_matrix)[:, :(n_neighbors + 1 )]
        index_0 = []
        index_1 = []
        for i in range(dist_matrix.shape[0]): 
            index1 = nearest_indices[i][dist_matrix[i,nearest_indices[i,]] < threshold]
            if len(index1) ==0:
                index1 = np.append(index1,i)
            index_0 += ([i]* len(index1))
            index_1 += (index1.tolist())
        
        edge_index = torch.tensor([index_0,index_1])
        graph = dgl.graph((edge_index[0],edge_index[1]))

        graph.ndata['feat'] = features.to(dtype=torch.float32)
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        return(graph,node2id)

        
    def __getitem__(self, idx):
        return self.graph
        
    def __len__(self):
        return 1

