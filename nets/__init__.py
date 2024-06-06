device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from generators.EffectEmbeddingDataset import *
import torch.nn.functional as F

