import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleNet(nn.module):
    def __init__(self, layer_list, dropout):
        super(FlexibleNet, self).__init__()
        # initialise architecture using provided layers list
        self.layers = nn.ModuleList([])
        for i, input_size in enumerate(layer_list[:-1]):
            output_size = layer_list[i+1]
            self.layers.append(nn.Linear(input_size,output_size))
            # add dropout after all but final layer
            if i < len(layer_list)-2:
                self.layers.append(nn.Dropout(dropout))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

