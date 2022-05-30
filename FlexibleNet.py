import torch
import torch.nn as nn
class FlexibleNet(nn.module):

    def __init__(self, layer_list, dropout, type):
        super(FlexibleNet, self).__init__()
        # initialise architecture using provided layers list
        self.layers = nn.ModuleList([])
        for i, input_size in enumerate(layer_list[:-1]):
            output_size = layer_list[i+1]
            self.layers.append(nn.Linear(input_size,output_size))
            # add dropout after all but final layer
            if i < len(layer_list)-2:
                self.layers.appen(nn.Dropout(dropout))




    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x}
