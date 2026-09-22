import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_train_val_split(full_train_set, val_size, seed=42):
    n_total = len(full_train_set)
    n_train = n_total - val_size
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train_set, [n_train, val_size], generator=g)
    return train_set, val_set


class TemperatureScaler(nn.Module):
    """
    Kalibrierung nach dem training um T_dach zu finden. Lernt ihn auf dem val set und teilt die
    logits durch T_dach
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels, max_iter=100):
        #logits, labels: fp32 Tensoren auf dem gelichen device
        self.to(logits.device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def _closure():
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return float(self.temperature.detach().cpu().item())
