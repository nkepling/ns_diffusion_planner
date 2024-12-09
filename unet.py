import torch
import torch.nn as nn

# UNet Model #
# I refrence this implementation from the following link:
# https://github.com/milesial/Pytorch-UNet/tree/master


# TODO: append time vector to input...
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization.
        # These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):  # Changed kernel size to 3
        super(SpatialAttention, self).__init__()
        # Padding to maintain the same spatial dimensions
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Shape: (batch_size, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # Shape: (batch_size, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Shape: (batch_size, 2, H, W)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)  # Shape: (batch_size, 1, H, W)
        # Apply sigmoid to normalize the attention map between 0 and 1
        out = self.sigmoid(out)
        return x * out  # Shape: (batch_size, channels, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, t=None):
        out_1 = self.conv1(x)
        out_1 += t if t is not None else 0
        act_1 = self.act(self.norm(out_1))
        out_2 = self.conv2(act_1)
        out_2 += t if t is not None else 0
        act_2 = self.act(self.norm(out_2))
        return act_2


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        return self.up(x)


class Down(nn.Module):
    def __init__(self) -> None:
        super(Down, self).__init__()
        # self.down = nn.MaxPool2d(kernel_size=3, stride=1,return_indices=True)
        self.down = nn.AvgPool2d(kernel_size=3, stride=1)
        # self.down = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        return self.down(x)


class UNet(torch.nn.Module):
    """UNet model for Score Function Approximation
    """

    def __init__(self, embed_dim=256) -> None:
        super(UNet, self).__init__()

        self.spatial_attention = SpatialAttention()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        # down sample and up sample layers
        self.down = Down()
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        # embedding layers

        self.layer_1_embed = Dense(embed_dim, 32)
        self.layer_2_embed = Dense(embed_dim, 64)
       
        self.layer_9_embed = Dense(embed_dim, 32)

        # Double Convolution Layers

        self.layer_1 = DoubleConv(4, 32)
        self.layer_2 = DoubleConv(32, 64)
     
        self.layer_9 = DoubleConv(64, 32)

        # Output Layer
        self.out = nn.Conv2d(32, 4, kernel_size=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):

        embed = self.act(self.embed(t))

        # Left Side of the UNet
        # 4, 10, 10 -> 32, 10, 10
        layer_1_out = self.layer_1(x, self.layer_1_embed(embed))
        layer_1_out = self.spatial_attention(
            layer_1_out)  # Apply spatial attention

        # 32, 10, 10 -> 64, 5, 5
        layer_2_out = self.layer_2(self.down(layer_1_out),
                                   self.layer_2_embed(embed))
        layer_2_out = self.spatial_attention(
            layer_2_out)  # Apply spatial attention


        up_4_out = self.up4(layer_2_out)  # 64, 5, 5 -> 32, 10, 10
        layer_9_out = self.layer_9(torch.cat([up_4_out, layer_1_out], 1),
                                   self.layer_9_embed(embed))
        layer_9_out = self.spatial_attention(
            layer_9_out)  # Apply spatial attention

        out = self.out(layer_9_out)  # Final output
        return out


if __name__ == "__main__":
    net = UNet()
    X = torch.randn(1, 4, 10, 10)
    t = torch.rand(1)
    net(X, t)
