import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear

class BayesianNN(nn.module):
