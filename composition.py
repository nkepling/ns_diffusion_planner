import torch
from torch import nn


def hole_conditional_score(X, maps):
    """
        function that takes a qvmap and outputs the score.
        Following the gradient of this score makes the sample
        more likely to be originated from a particular map
        configuration

        we just use a standard normal distribution
        X: (batch, actions, row, col)
        map: (batch, row, col)
    """
    zeros = torch.zeros_like(X)
    for mapndx, map in enumerate(maps):
        for rowndx, row in enumerate(map):
            for colndx, state in enumerate(row):
                if state == 'H' or state == 'G':
                    zeros[mapndx, :, rowndx, colndx] = 1

    score = torch.neg(X * zeros)

    # output dim (batch, actions, row, col)
    return score


class Conditional(nn.Module):
    def __init__(self, base, conditional, maps):
        super(Conditional, self).__init__()
        self.base = base
        self.conditional = conditional
        self.maps = maps

    def forward(self, X, t):
        base_score = self.base(X, t)
        cond_score = self.conditional(X, self.maps)
        return base_score + cond_score

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

class Composition(nn.Module):
    def __init__(self, weights, *models):
        super(Composition, self).__init__()
        self.models = models

    def forward(self, X, ts, weights):
        score = torch.zeros_like(X)
        for model, weight in zip(self.models, weights):
            score += model(X, ts) * weight

        return score


def composition(model_1, a, model_2, b, num_steps, shape):
    """
    Reverse diffusion sampling: noise -> image
    Composition of two score models using
    a weighted product of experts scheme
    Use Langevin Dynamics, start with noise and
    slowly change it with the score.
    """
    # update self.sample_steps
    if model_1.sample_steps_cache is None or len(model_1.sample_steps_cache) != num_steps:
        model_1.compute_sample_steps(num_steps)
    # sample X_T (gaussian noise) and step size
    X = torch.randn(shape)
    tau = 1 / num_steps
    # iteratively update X with Euler-Maruyama method
    for t in model_1.sample_steps:
        noise = torch.randn(shape)
        score_1 = model_1.score_model(X, t)
        score_2 = model_2.score_model(X, t)
        composed_score = a * score_1 + b * score_2
        X = X + tau * composed_score + torch.sqrt(2 * tau) * noise
    return X


if __name__ == "__main__":

    import argparse
    from utils import parse_config
    parser = argparse.ArgumentParser(description='Generate Q-value map data.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file.')
    args = parser.parse_args()
    config = parse_config(args.config)

    model_1 = config['model_1']
    a = config['model_1_weight']
    model_2 = config['model_2']
    b = config['model_2_weight']

    composition(model_1, a, model_2, b)
