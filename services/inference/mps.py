"""Equivalent spatial pooling for PyTorch versions lacking MPS max_pool3d."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialMaxPool3d(nn.Module):
    """The reference frontend pools spatially with a temporal kernel/stride of 1."""
    def forward(self, x):
        b, c, t, h, w = x.shape
        spatial = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        pooled = F.max_pool2d(spatial, kernel_size=3, stride=2, padding=1)
        return pooled.reshape(b, t, c, *pooled.shape[-2:]).permute(0, 2, 1, 3, 4)
