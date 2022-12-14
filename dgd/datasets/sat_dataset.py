import os

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

from dgd.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# TODO: Update
#SAT_GRAPH_FILE = "sat/allsat_v10_20_c20_50_n750_train.pt" 
#SAT_GRAPH_TEST_FILE = "sat/allsat_v10_20_c20_50_n250_test.pt" 

# 10 to 20 Variables random
#SAT_GRAPH_FILE = "sat/allsat_v10_20_c20_50_n75000_train.pt"
#SAT_GRAPH_TEST_FILE = "sat/allsat_v10_20_c20_50_n25000_test.pt"

# 100 Variables 3SAT
#SAT_GRAPH_FILE = "sat/3sat_v100_c428_n11000_train.pt"
#SAT_GRAPH_TEST_FILE = "sat/3sat_v100_c428_n1000_test.pt"

# 25 Variables 3SAT
#SAT_GRAPH_FILE = "sat/3sat_v25_c106_n11000_train.pt"
#SAT_GRAPH_TEST_FILE = "sat/3sat_v25_c106_n1000_test.pt"

# 10 Variables 3SAT
SAT_GRAPH_FILE = "sat/3sat_v10_c42_n11000_train.pt"
SAT_GRAPH_TEST_FILE = "sat/3sat_v10_c42_n1000_test.pt"



class SatDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.graphs = torch.load(filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        n_edge_classes = data.edge_attr.shape[-1]
        is_sat = data.y
        # Ignore is_sat for now, train on only sat problems
        y = torch.zeros([1, 0]).float()
        #y = (is_sat*torch.tensor([[0,1]]) + (1 - is_sat)*torch.tensor([[1,0]])).float()
        data.idx = idx
        n_nodes  = data.num_nodes * torch.ones(1, dtype=torch.long)

        # Symmetrize
        edge_index, edge_attr = torch_geometric.utils.to_undirected(data.edge_index, data.edge_attr,  data.num_nodes, reduce='max')
        n_edges = edge_index.shape[-1]

        # Add edge type for "no edge"
        new_edge_attr = torch.zeros(n_edges, n_edge_classes+1, dtype=torch.float)
        new_edge_attr[:, 1:] = edge_attr
        data_out = torch_geometric.data.Data(x=data.x.float(), edge_index=edge_index, edge_attr=new_edge_attr,
                                         y=y, idx=idx, n_nodes=n_nodes)
        return data_out


class SatDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.file_name = SAT_GRAPH_FILE
        self.test_file_name = SAT_GRAPH_TEST_FILE
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self):
        train_val_graphs = SatDataset(self.file_name)
        test_graphs = SatDataset(self.test_file_name)
        val_len = int(round(len(train_val_graphs) * 1./11.))
        train_len = len(train_val_graphs) - val_len
        train_val_splits = random_split(train_val_graphs, [train_len, val_len], generator=torch.Generator().manual_seed(1234))

        test_len = len(test_graphs)
        #test_len = 10
        #val_len = 100
        #train_len = 1000
        #discard_len = len(train_val_graphs) - train_len - val_len # Extra data we won't use
        #train_val_splits = random_split(train_val_graphs, [train_len, val_len, discard_len], generator=torch.Generator().manual_seed(1234))
        #discard_len = len(test_graphs) - test_len 
        #test_discard_splits = random_split(test_graphs, [test_len, discard_len], generator=torch.Generator().manual_seed(1234))

        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        datasets = {'train': train_val_splits[0], 'val': train_val_splits[1], 'test': test_graphs}
        super().prepare_data(datasets)


class SatDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name = 'nx_graphs'
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

