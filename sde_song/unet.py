import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
    """A time-dependent score-based model for smaller inputs (e.g., 10x10)."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128], embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        # Time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        # Encoder
        self.conv1 = nn.Conv2d(4, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)  
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        # Decoder
        # Going from 3x3 back to 5x5
        self.tconv3 = nn.ConvTranspose2d(channels[2], channels[1], 3, stride=2, padding=1, output_padding=0, bias=False)
        self.dense4 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        # Going from 5x5 back to 10x10
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        # Final layer to get back to 10x10 from 10x10
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 4, 3, stride=1, padding=1)
        self.dense6 = Dense(embed_dim, 4)

        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        # Embedding for time
        embed = self.act(self.embed(t))

        # Encoding path
        # Stage 1
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        # Stage 2
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        # Stage 3
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        # Decoding path
        # From h3 (3x3) to h2 (5x5)
        h = self.tconv3(h3)
        h += self.dense4(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = torch.cat([h, h2], dim=1)

        # From 5x5 back to 10x10
        h = self.tconv2(h)
        h += self.dense5(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = torch.cat([h, h1], dim=1)

        # Final layer (10x10 -> 10x10)
        h = self.tconv1(h)
        # Add a final embed as well (optional)
        h += self.dense6(embed)
        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]

        return h
