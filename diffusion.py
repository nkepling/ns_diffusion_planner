import torch
from torch import nn


class DiffusionModel(nn.Module):
    def __init__(self, score_model) -> None:
        self.score_model = score_model

    def forward(self, XT, t):
        """Predict the score at time t for the a noisy input XT
        """
        t = self.time_embedding(t)
        return self.score_model(XT, t)

    def sliced_score_matching(self, X):
        """Sliced Score Matching
        """
        X = X.clone().detach().requires_grad_(True)
        V = torch.randn(X.shape)
        S = self(X)
        VS = torch.sum(V * S, axis=1)
        rhs = .5 * VS ** 2
        gVS = torch.autograd.grad(
            VS, self.X, grad_outputs=torch.ones_like(VS), create_graph=True)[0]
        lhs = torch.sum(V * gVS, axis=1)
        F_Divergence = torch.mean(lhs - rhs)
        return F_Divergence

    def time_embedding(self, t):
        """Maybe embed the time vector here? 
        """
        return t

    def forward_diffusion(self, x):
        """
        Forward diffusion: image -> noise

        Add noise to the image to get the noise sample
        """
        return 0

    def sample(self, shape, num_steps, device):
        """Reverse diffusion sampling: noise -> image
        """
        return 0
