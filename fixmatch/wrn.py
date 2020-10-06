"""Implementation of Wide Residual Networks.

See: https://arxiv.org/abs/1605.07146
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A basic (3, 3) residual block."""

    def __init__(self, in_channels, out_channels, stride, bn_momentum=0.001, lrelu_alpha=0.1, activate_before_residual=False):
        """Constructor creating the layers."""
        super(ResidualBlock, self).__init__()
        self.activate_before_residual = activate_before_residual
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_alpha, inplace=True)

        # First convolution in block
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)

        # Second convolution in block
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)

        # Mapping before add
        self.mapping = nn.Identity()
        if in_channels != out_channels:
            # Perform a convolution on the residual between blocks
            self.mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x0):
        """Performs a forward pass."""
        x = self.lrelu(self.bn1(x0))
        if self.activate_before_residual:
            x0 = x
        x = self.conv1(x)
        x = self.lrelu(self.bn2(x))
        x = self.conv2(x)
        return self.mapping(x0) + x


class WRN(nn.Module):
    """A Wide Residual Network."""

    def __init__(self, widening_factor=2):
        super(WRN, self).__init__()
        out_channels = {
            'conv1': 16,
            'conv2': widening_factor * 16,
            'conv3': widening_factor * 32,
            'conv4': widening_factor * 64
        }

        self.model = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=out_channels['conv1'],
                      kernel_size=3, stride=1, padding=1, bias=False),
            # conv2
            ResidualBlock(in_channels=out_channels['conv1'], out_channels=out_channels['conv2'], stride=1),
            ResidualBlock(in_channels=out_channels['conv2'], out_channels=out_channels['conv2'], stride=1),
            ResidualBlock(in_channels=out_channels['conv2'], out_channels=out_channels['conv2'], stride=1),
            ResidualBlock(in_channels=out_channels['conv2'], out_channels=out_channels['conv2'], stride=1),
            # conv3
            ResidualBlock(in_channels=out_channels['conv2'], out_channels=out_channels['conv3'], stride=1),
            ResidualBlock(in_channels=out_channels['conv3'], out_channels=out_channels['conv3'], stride=2),
            ResidualBlock(in_channels=out_channels['conv3'], out_channels=out_channels['conv3'], stride=2),
            ResidualBlock(in_channels=out_channels['conv3'], out_channels=out_channels['conv3'], stride=2),
            # conv4
            ResidualBlock(in_channels=out_channels['conv3'], out_channels=out_channels['conv4'], stride=1),
            ResidualBlock(in_channels=out_channels['conv4'], out_channels=out_channels['conv4'], stride=2),
            ResidualBlock(in_channels=out_channels['conv4'], out_channels=out_channels['conv4'], stride=2),
            ResidualBlock(in_channels=out_channels['conv4'], out_channels=out_channels['conv4'], stride=2)
        )
