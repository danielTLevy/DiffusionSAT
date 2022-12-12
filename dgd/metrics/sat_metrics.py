import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import wandb
import torch.nn as nn
from collections import defaultdict
import numpy as np

def calc_frac_clause_sat(X_i, E_i):
    # Assume X and E are collapsed (argmax'd)
    clause_mask = X_i == 0.
    n_clauses = clause_mask.sum().item()
    pos_links = (E_i == 1).nonzero()
    neg_links = (E_i == 2).nonzero()
    pos_links = pos_links[pos_links[:, 0] < n_clauses]
    neg_links = neg_links[neg_links[:, 0] < n_clauses]
    neg_vars_dict = defaultdict(list)
    for clause, neg_var in zip(*neg_links.detach().cpu().numpy().T.tolist()):
        neg_vars_dict[clause].append(neg_var)
    pos_vars_dict = defaultdict(list)
    for clause, pos_var in zip(*pos_links.detach().cpu().numpy().T.tolist()):
        pos_vars_dict[clause].append(pos_var)
        
    clause_sat = []
    for clause_i in range(n_clauses):
        pos_vars = pos_vars_dict[clause_i]
        pos_sat = X_i[pos_vars] == 1
        neg_vars = neg_vars_dict[clause_i]
        neg_sat = X_i[neg_vars] == 2
        is_sat = torch.cat((pos_sat, neg_sat)).any().item()
        clause_sat.append(is_sat)
    return np.mean(clause_sat)



class SatSolvedMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_sat', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_clause_sat', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, X, E):
        """ Update state with percentage of clauses satisfied
            X: (bs, n, dx)
            E: (bs, n, n, de) """
        batch_size = X.shape[0]
        for batch_i in range(batch_size):
            X_i = X[batch_i]
            E_i = E[batch_i]
            frac_clause_sat = calc_frac_clause_sat(X_i, E_i)
            self.total_clause_sat += frac_clause_sat
            self.total_sat += 1 if frac_clause_sat == 1. else 0
        self.total_samples += batch_size


    def compute(self):
        return self.total_sat / self.total_samples, self.total_clause_sat / self.total_samples