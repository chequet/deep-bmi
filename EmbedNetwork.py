import os
import torch
from torch import nn
from EffectEmbeddingDataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class EmbedNetwork(nn.Module):
    def __init__(self, n_inputs, reduction_factor, dropout):
        super(EmbedNetwork, self).__init__()
        n_features = n_inputs * 2
        n_out1 = math.ceil(n_features / reduction_factor)
        # (reduction_factor**2))
        n_out2 = math.ceil(n_out1 / reduction_factor)
        n_out3 = math.ceil(n_out2 / reduction_factor)
        self.model = nn.Sequential(
            torch.nn.Linear(n_features, n_out1),
            torch.nn.Dropout(dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_out1, n_out2),
            torch.nn.Dropout(dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_out2, n_out3),
            torch.nn.Dropout(dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_out3, 1)
        )
        self.downsample = torch.nn.Linear(n_features,1)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.model(x)
        out += residual
        return out