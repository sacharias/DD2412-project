"""Implementation of Wide Residual Networks.

See: https://arxiv.org/abs/1605.07146
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A basic (3, 3) residual block."""

    def __init__(self, in_channels, out_channels, stride, activate_before_residual=False, lrelu_alpha=0.1, bn_momentum=0.001):
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
        x0 = self.mapping(x0)
        return x0 + x


class WRN(nn.Module):
    """A Wide Residual Network with basic (3, 3) residual blocks.

    The total number of convolutional layers (depth) is n and the widening factor is k.

    With N being the number of residual blocks per group, we get the total number of
    convolutional layers n = 4 + 6*N:
        conv1:      1 convolution
        mapping:    1 convolution
        conv2:      2*N convolutions
        mapping:    1 convolution
        conv3:      2*N convolutions
        mapping:    1 convolution
        conv4:      2*N convolutions
    """

    def __init__(self, num_classes, n=28, k=2, lrelu_alpha=0.1, bn_momentum=0.001):
        super(WRN, self).__init__()
        N = (n - 4) // 6
        out_channels = [16, 16*k, 32*k, 64*k]
        strides = [1, 1, 2, 2]

        # conv1
        layers = [nn.Conv2d(in_channels=3, out_channels=out_channels[0],
                            kernel_size=3, stride=strides[0], padding=1, bias=False)]

        # conv2-4
        for g in range(1, len(out_channels)):
            # The first residual block of conv2 does the activation first,
            # and the first layer has larger stride in conv3-4 to reduce output size
            layers.append(ResidualBlock(in_channels=out_channels[g-1], out_channels=out_channels[g],
                                        stride=strides[g], activate_before_residual=(g == 1)))
            for _ in range(N-1):
                layers.append(ResidualBlock(in_channels=out_channels[g], out_channels=out_channels[g], stride=1))

        # avg-pool and fully-connected layer
        layers.append(nn.BatchNorm2d(num_features=out_channels[-1], momentum=bn_momentum))
        layers.append(nn.LeakyReLU(negative_slope=lrelu_alpha, inplace=True))
        layers.append(nn.AdaptiveAvgPool2d(output_size=1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_features=out_channels[-1], out_features=num_classes))

        # Put it all together
        self.wrn = nn.Sequential(*layers)

    def forward(self, x):
        """Performs a forward pass."""
        return self.wrn(x)
