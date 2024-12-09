import torch
import torch.nn as nn
import numpy as np


class ScoreNet(nn.Module):
    """A score-based U-Net model.

    Args:
      source_channels: Number of input (and output) channels for the image.
      channels: A list specifying the number of channels for each level of the U-Net.
    """

    def __init__(self, source_channels=1, channels=[32, 64, 128, 256]):
        super().__init__()

   
        # Encoding layers
        self.conv1 = nn.Conv2d(
            source_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(
            channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(
            channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.conv4 = nn.Conv2d(
            channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, padding=1, output_padding=0, bias=False
        )
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] * 2, channels[1], 3, stride=2, padding=1, output_padding=0, bias=False
        )
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1]*2, channels[0], 3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(
            channels[0]*2, source_channels, 3, stride=1, padding=1
        )
        # Swish activation
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, rs):
        # Encoding
        h1 = self.act(self.gnorm1(self.conv1(x)))  # Resolution stays the same
        h2 = self.act(self.gnorm2(self.conv2(h1)))  # Downsampled by 2
        h3 = self.act(self.gnorm3(self.conv3(h2)))  # Downsampled by 2 again
        h4 = self.act(self.gnorm4(self.conv4(h3)))  # Downsampled by 2 again

        # Decoding
        h = self.act(self.tgnorm4(self.tconv4(h4)))  # Upsampled
        # Concatenate and upsample
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1))))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1))))
        # Concatenate and restore original resolution
        h = self.tconv1(torch.cat([h, h1], dim=1))

        return h / rs[:, None, None, None] ** 2
