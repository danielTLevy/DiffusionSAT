import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection
import time
import wandb
from dgd.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, true_X, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0


        if log:
            to_log = {"train_loss/batch_CE": (loss_X ).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1}
            wandb.log(to_log, commit=True)
        return loss_X# + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self, current_epoch, start_epoch_time):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss}
        wandb.log(to_log, commit=False)

        print(f"Epoch {current_epoch} finished: X: {epoch_node_loss :.2f} -- {time.time() - start_epoch_time:.1f}s ")

