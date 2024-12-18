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
    return score, (1 - zeros)


class Conditional(nn.Module):
    def __init__(self, base, conditional, maps):
        super(Conditional, self).__init__()
        self.base = base
        self.conditional = conditional
        self.maps = maps

    def forward(self, X, t):
        base_score = self.base(X, t)
        cond_score, mask = self.conditional(X, self.maps)
        masked_base_score = base_score * mask
        return masked_base_score + cond_score


class Composition(nn.Module):
    def __init__(self, models, weights):
        super(Composition, self).__init__()
        self.models = models
        self.weights = weights

    def forward(self, X, ts):
        score = torch.zeros_like(X)
        for model, weight in zip(self.models, self.weights):
            score += model(X, ts) * weight

        return score
