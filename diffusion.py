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

    def sliced_score_matching(self, X, t):
        """Sliced Score Matching
        """
        # compute score
        X = X.clone().detach().requires_grad_(True)
        S = self(X, t).flatten(start_dim=1)

        # get unit vectors
        V = torch.randn(S.shape)
        V = V / torch.norm(V, dim=1)[:, None]

        VS = torch.sum(V * S, axis=1)
        rhs = .5 * VS ** 2

        gVS = torch.autograd.grad(
            VS, X, grad_outputs=torch.ones_like(VS), create_graph=True)[0]
        gVS = gVS.flatten(start_dim=1)

        lhs = torch.sum(V * gVS, axis=1)
        F_Divergence = torch.mean(lhs - rhs)
        return F_Divergence

    def time_embedding(self, t):
        """Maybe embed the time vector here?
           We can keep it linear for now, since it will be limited [0,1]
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
