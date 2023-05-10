import torch
import numpy as np
from builtins import range
from torch_geometric.data import InMemoryDataset, Data
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.model_selection import train_test_split



def get_tonsilbe_edge_index(pos, distance_thres):
    # construct edge indexes in one region
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres
    np.fill_diagonal(dists_mask, 0)
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
    return edge_list


def load_tonsilbe_data(filename, distance_thres, sample_rate):
    df = pd.read_csv(filename)
    train_df = df.loc[df['sample_name'] == 'tonsil']
    train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
    test_df = df.loc[df['sample_name'] == 'Barretts Esophagus']
    train_X = train_df.iloc[:, 1:-4].values
    test_X = test_df.iloc[:, 1:-4].values
    train_y = train_df['cell_type'].str.lower()
    labeled_pos = train_df.iloc[:, -4:-2].values
    unlabeled_pos = test_df.iloc[:, -4:-2].values
    cell_types = np.sort(list(set(train_y))).tolist()
    cell_type_dict = {}
    inverse_dict = {}    
    for i, cell_type in enumerate(cell_types):
        cell_type_dict[cell_type] = i
        inverse_dict[i] = cell_type
    train_y = np.array([cell_type_dict[x] for x in train_y])
    labeled_edges = get_tonsilbe_edge_index(labeled_pos, distance_thres)
    unlabeled_edges = get_tonsilbe_edge_index(unlabeled_pos, distance_thres)
    return train_X, train_y, test_X, labeled_edges, unlabeled_edges, inverse_dict


def get_mouselymph_edge_index(pos, distance_thres):
    edge_list = []
    dists = pairwise_distances(pos)
    dists_mask = dists < distance_thres # true is dist<50 otherwise false
    np.fill_diagonal(dists_mask, 0) # fill diagonal as false
    edge_list = np.transpose(np.nonzero(dists_mask)).tolist() # indices to the matrix where the value is true
    return edge_list

def load_mouselymph_data(filename, distance_thres, sample_rate, way = 'within_tissue'):
    if way == 'within_tissue':
        df = pd.read_csv(filename)
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['cluster'])

        train_df = pd.DataFrame(train)
    
        train_df_sampled = train_df.sample(
            n=round(sample_rate*len(train_df)), random_state=1)
    
        train_X = train_df_sampled.iloc[:, 1:-3].values
        train_y = train_df_sampled['cluster'].to_numpy()

        test_df = pd.DataFrame(test)
        test_X = test_df.iloc[:, 1:-3].values
        test_y = test_df['cluster'].to_numpy()

        labeled_pos = train_df_sampled.iloc[:, -3:-1].values 
        unlabeled_pos = test_df.iloc[:, -3:-1].values
    elif way == 'cross':
        df = pd.read_csv(filename)
        train_df = df.loc[df['region'] == 1]
        train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
        test_df = df.loc[df['region'] == 2]
        train_X = train_df.iloc[:, 1:-4].values
        test_X = test_df.iloc[:, 1:-4].values
        train_y = train_df['cluster'].str.lower()
        test_y = test_df['cluster'].str.lower()
        labeled_pos = train_df.iloc[:, -4:-2].values
        unlabeled_pos = test_df.iloc[:, -4:-2].values
        cell_types = np.sort(list(set(train_y))).tolist()
        cell_type_dict = {}
        inverse_dict = {}    
        for i, cell_type in enumerate(cell_types):
            cell_type_dict[cell_type] = i
            inverse_dict[i] = cell_type
        train_y = np.array([cell_type_dict[x] for x in train_y])
        labeled_edges = get_mouselymph_edge_index(labeled_pos, distance_thres)
        unlabeled_edges = get_mouselymph_edge_index(unlabeled_pos, distance_thres)

    elif way == 'cross_infection':
        df = pd.read_csv(filename)
        train_df = df.loc[df['region'] == 'healthy']
        train_df = train_df.sample(n=round(sample_rate*len(train_df)), random_state=1)
        test_df = df.loc[df['region'] == 'infected']
        train_X = train_df.iloc[:, 1:-4].values
        test_X = test_df.iloc[:, 1:-4].values
        test_y = test_df['cluster'].str.lower()
        train_y = train_df['cluster'].str.lower()
        labeled_pos = train_df.iloc[:, -4:-2].values
        unlabeled_pos = test_df.iloc[:, -4:-2].values
        cell_types = np.sort(list(set(train_y))).tolist()
        cell_type_dict = {}
        inverse_dict = {}    
        for i, cell_type in enumerate(cell_types):
            cell_type_dict[cell_type] = i
            inverse_dict[i] = cell_type
        train_y = np.array([cell_type_dict[x] for x in train_y])
        labeled_edges = get_mouselymph_edge_index(labeled_pos, distance_thres)
        unlabeled_edges = get_mouselymph_edge_index(unlabeled_pos, distance_thres)


    return train_X, train_y, test_X, test_y, labeled_edges, unlabeled_edges, inverse_dict




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