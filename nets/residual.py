import torch.nn as nn

class block1(nn.Module):
    def __init__(self, n_inputs):
        super(block1, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_inputs, 1)
        self.conv2 = nn.Conv2d(n_inputs, n_inputs, 3, 1, 1)
        self.classifier = nn.Linear(n_inputs * 24 * 24, 751)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out += residual

        out = out.view(out.size(0), -1)
        return self.classifier(out)