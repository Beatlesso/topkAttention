import torch.utils.benchmark as benchmark
import torch
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import time
import scipy
from torch import Tensor

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        score = torch.matmul(q, k.transpose(-1, -2)) # 1.Matmul
        score = score / self.scale # 2.Scale

        if mask is not None:
            score = score.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(score) # 4.Softmax
        output = torch.matmul(attn, v) # 5.Output

        return attn, output