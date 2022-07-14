import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear

class BayesianNN(nn.Module):
    def __init__(self, layer_list, dropout, activation):
        super(BayesianNN, self).__init__()
        # initialise architecture using provided layers list
        self.layers = nn.ModuleList([])
        for i, input_size in enumerate(layer_list[:-1]):
            output_size = layer_list[i+1]
            # add regular linear layers for input and output
            if i == 0 or i == (len(layer_list)-1):
                self.layers.append(nn.Linear(input_size,output_size))
            else:
                self.layers.append(BayesianLinear(input_size,output_size))
            # add dropout after all but final layer
            if i < len(layer_list)-2:
                self.layers.append(nn.Dropout(dropout))
                # add activation function depending on what was chosen
                if activation == 'ELU':
                    self.layers.append(nn.ELU())
                elif activation == 'ReLU':
                    self.layers.append(nn.ReLU())
                elif activation == 'LeakyReLU':
                    self.layers.append(nn.LeakyReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

