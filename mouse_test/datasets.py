import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd

def get_mouselymph_edge_index(pos, distance_thres):
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres # true is dist<50 otherwise false
    np.fill_diagonal(dists_mask, 0) # fill diagonal as false
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist() # indices to the matrix where the value is true
    return edge_list

def load_mouselymph_data(filename, distance_thres, sample_rate):
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(filename)
    X = df.iloc[:, 2:-1].values
    y = df['cluster']
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)
    train_X_df = pd.DataFrame(train_X)
    train_X_df = train_X_df.sample(n=round(sample_rate*len(train_X_df)), random_state=1)
    train_y_df = y[train_X_df.index]
    train_X = train_X_df.to_numpy()
    train_y = train_y_df.to_numpy()
    labeled_pos = train_X[:, -2:]
    unlabeled_pos = test_X[:, -2:]
    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    test_y = np.array([cell_type_dict[x] for x in test_y])
    labeled_edges = get_mouselymph_edge_index(labeled_pos, distance_thres)
    unlabeled_edges = get_mouselymph_edge_index(unlabeled_pos, distance_thres)
    return train_X.astype('float64'), train_y, test_X.astype('float64'), test_y, labeled_edges, unlabeled_edges, inverse_dict

class GraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, labeled_edges, unlabeled_edges, transform=None,):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T, y=torch.LongTensor(labeled_y))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data