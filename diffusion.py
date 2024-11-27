import torch
from torch import nn


class DiffusionModel(nn.Module):
    def __init__(self, score_model, device, T=1000) -> None:
        super(DiffusionModel, self).__init__()
        self.score_model = score_model
        self.sample_steps_cache = None

        self.device = device
        self.ts = torch.linspace(0, 1, T)
        self.T = T

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

    def sample(self, num_steps, shape):
        """
        Reverse diffusion sampling: noise -> image
        Use Langevin Dynamics, start with noise and
        slowly change it with the score.
        """

        # update self.sample_steps
        if self.sample_steps_cache is None or len(self.sample_steps_cache) != num_steps:
            self.compute_sample_steps(num_steps)

        # sample X_T (gaussian noise) and step size
        X = torch.randn(shape)
        tau = 1 / num_steps

        # iteratively update X with Euler-Maruyama method
        for t in self.sample_steps:
            noise = torch.randn(shape)
            score = self.score_model(X, t)
            X = X + tau * score + torch.sqrt(2 * tau) * noise

        return X

    def compute_sample_steps(self, num_steps):
        """
        This function distributes the requested number of sample steps
        across our T discrete ts.
        ### Example:
        if num_steps == 20000
        then sample steps takes two steps for each t
        [0,0, 1e-10, 1e-10,...]
        """
        T = max(num_steps, self.T)

        leftn = T % self.T
        leftend = leftn / self.T
        rep = T // self.T + 1  # plus one to round up

        left = torch.linspace(0, leftend, leftn)
        left = left.repeat_interleave(rep)

        rightn = self.T - leftn
        right = torch.linspace(leftend, 1, rightn)
        right = right.repeat_interleave(rep-1)

        # range is the
        self.sample_steps_cache = left.tolist() + right.tolist()
        return self.sample_steps_cache
