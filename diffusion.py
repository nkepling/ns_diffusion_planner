import torch
from torch import nn
from numpy import sqrt


class DiffusionModel(nn.Module):
    def __init__(self, score_model, device, T=1000) -> None:
        super(DiffusionModel, self).__init__()
        self.score_model = score_model

        self.device = device
        self.score_model.to(device)

        self.ts = torch.linspace(0, 1, T).to(device)
        self.T = T

    def forward(self, XT, t):
        """Predict the score at time t for the a noisy input XT
        """
        t = self.time_embedding(t)
        t = t.to(self.device)
        return self.score_model(XT, t)

    def sliced_score_matching(self, X, t):
        """Sliced Score Matching
        """
        # compute score
        X = X.clone().detach().requires_grad_(True)
        S = self(X, t).flatten(start_dim=1)

        # get unit vectors
        V = torch.randn(S.shape).to(self.device)
        V = V / (torch.norm(V, dim=1)[:, None] + 1e-12)

        VS = torch.sum(V * S, axis=1)
        rhs = .5 * VS ** 2

        gVS = torch.autograd.grad(
            VS, X, grad_outputs=torch.ones_like(VS),
            create_graph=True)[0]
        gVS = gVS.flatten(start_dim=1)

        lhs = torch.sum(V * gVS, axis=1)
        F_Divergence = torch.mean(lhs - rhs)
        return F_Divergence

    def time_embedding(self, t):
        """Maybe embed the time vector here?
           We can keep it linear for now, since it will be limited [0,1]
           Time embedding is done in the unet architecture
           as a gaussian fourier projection
        """
        return t

    def preturb_data(self, X, t):
        """
        Forward diffusion: image -> noise
        t is a 1-dimensional scalar.
        Entire batch recieves same noise.
        applied to entire batch.

        Add noise to the image to get the noise sample
        """

        W = torch.randn_like(X, device=X.device)

        Xw = X * (1-t) + W * (t)

        return Xw

    def preturb_data_step(self, X):
        """
        Incremental Forward diffusion: image -> image+noise
        """

        dt = 1 / self.T

        W = torch.randn_like(X, device=X.device)

        Xw = X * (1-dt) + W * (dt)

        return Xw


class DiffusionSampler(nn.Module):
    def __init__(self, model, num_steps) -> None:
        super(DiffusionSampler, self).__init__()
        self.model = model
        self.ts = compute_sample_steps(num_steps)
        self.tau = 1 / num_steps

    def forward(self, shape, weights=None):
        """
        Reverse diffusion sampling: noise -> image
        Use Langevin Dynamics, start with noise and
        slowly change it with the score.
        """

        # sample X_T (gaussian noise) and step size
        X = torch.randn(shape)

        # iteratively update X with Euler-Maruyama method
        for t in reversed(self.ts):
            t = torch.tensor([t])
            noise = torch.randn(shape)
            score = self.model(X, t) if weights is None else self.model(X, t, weights)
            X = X + self.tau * score + sqrt(2 * self.tau) * noise

        return X


def compute_sample_steps(num_steps):
    """
    This function distributes the requested number of sample steps
    across our 1000 discrete ts.
    ### Example:
    if num_steps == 2000
    then sample steps takes two steps for each t
    [0,0, 1e-10, 1e-10,..., 1, 1]
    """
    T = max(num_steps, 1000)

    leftn = T % 1000
    leftend = leftn / 1000
    rep = T // 1000 + 1  # plus one to round up

    left = torch.linspace(0, leftend, leftn)
    left = left.repeat_interleave(rep)

    rightn = 1000 - leftn
    right = torch.linspace(leftend, 1, rightn)
    right = right.repeat_interleave(rep-1)

    # range is the
    return left.tolist() + right.tolist()
