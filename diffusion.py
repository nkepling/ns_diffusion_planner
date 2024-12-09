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
        self.betas = torch.linspace(1e-4, 2e-2, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward(self, XT, t):
        """Predict the score at time t for the a noisy input XT
        """
        t = self.time_embedding(t)
        t = t.to(self.device)
        return self.score_model(XT, t/(self.T - 1))

    def sliced_score_matching(self, X, t, n_particles=10):
        # Prepare X for gradient
        X = X.clone().detach().requires_grad_(True)
        batch_size = X.size(0)

        # Expand X first
        expanded_X = X.unsqueeze(0).expand(
            n_particles, *X.shape).contiguous().view(-1, *X.shape[1:])
        expanded_X.requires_grad_(True)
        t_expanded = t.unsqueeze(0).expand(
            n_particles, batch_size).contiguous().view(-1)
        # Recompute the score on expanded_X so the graph depends on expanded_X
        expanded_S = self(expanded_X, t_expanded).flatten(
            start_dim=1)  # shape: [n_particles * batch_size, d]

        # Draw multiple random directions V
        V = torch.randn_like(expanded_S)
        V = V / (torch.norm(V, dim=1, keepdim=True) + 1e-12)

        # Compute V * S for each direction
        VS = torch.sum(V * expanded_S, dim=1)  # [n_particles * batch_size]
        rhs = 0.5 * VS**2

        # Now VS depends on expanded_X through expanded_S
        gVS = torch.autograd.grad(
            VS, expanded_X, grad_outputs=torch.ones_like(VS), create_graph=True
        )[0].flatten(start_dim=1)

        lhs = torch.sum(V * gVS, dim=1)  # [n_particles * batch_size]

        # Combine terms
        loss = lhs + rhs
        loss = loss.view(n_particles, batch_size)
        loss = loss.mean(dim=0)
        loss = loss.mean()

        return loss

    def time_embedding(self, t):
        """Maybe embed the time vector here?
           We can keep it linear for now, since it will be limited [0,1]
           Time embedding is done in the unet architecture
           as a gaussian fourier projection
        """
        return t

    def perturb_data(self, X, t_int):
        """
        Perturb data according to the forward noising process defined by the alpha_bars.
        t_int: integer time indices in [0, T-1].
        alpha_bars: precomputed cumulative product of alphas.
        """
        # t_int is a tensor of shape [batch] with values in [0, T-1]
        # Extract corresponding alpha_bar_t
        # assuming X is image-like [B, C, H, W]
        alpha_bar_t = self.alpha_bars[t_int].view(-1, 1, 1, 1)

        # Sample Gaussian noise
        eps = torch.randn_like(X)

        # Apply forward diffusion
        return torch.sqrt(alpha_bar_t) * X + torch.sqrt(1 - alpha_bar_t) * eps


class DiffusionSampler(nn.Module):
    def __init__(self, model, num_steps, betas) -> None:
        super(DiffusionSampler, self).__init__()
        self.model = model
        self.ts = compute_sample_steps(num_steps)
        self.tau = 1 / num_steps
        self.T = num_steps
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def langevin_sampler(self, shape, device):
        self.model.eval()
        with torch.no_grad():
            # Start from pure Gaussian noise
            x = torch.randn(shape, device=device)

            for i in reversed(range(self.T)):
                t_tensor = torch.tensor(
                    [i], device=device, dtype=torch.float32)
                # model should accept t as a tensor
                score = self.model(x, t_tensor)

                beta_t = self.betas[i]
                alpha_bar_t = self.alpha_bars[i]

                # Compute the mean of q(x_{t-1} | x_t)
                # Based on DDPM Eq. (4)
                mean = (1 / torch.sqrt(self.alphas[i])) * (x -
                                                           (beta_t / torch.sqrt(1 - alpha_bar_t)) * score)

                if i > 0:
                    # Add noise
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean

            return x
