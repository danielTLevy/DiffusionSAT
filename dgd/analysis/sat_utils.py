from rdkit import Chem
from typing import Dict, List, Tuple
import rdkit
import torch
import torch_geometric
import pandas as pd
import wandb
import torch.nn as nn
import torch.nn.functional as f


class SatSamplingMetrics(nn.Module):
    '''
    Module for computing statistics between the generated graphs and test graphs
    '''
    def __init__(self, dataloaders, metrics_list=[]):
        super().__init__()
        self.metrics_list = metrics_list

    def forward(self, generated_graphs: list, name, current_epoch, val_counter, save_graphs=True, test=False):
        '''
        Compare generated_graphs list with test graphs
        '''
        if 'example_metric' in self.metrics_list:
            print("Computing example_metric stats..")
            # example_metric = compute_example_metric()
            # wandb.run.summary['example_metric'] = example_metric

    def reset(self):
        pass
