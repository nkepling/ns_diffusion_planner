import torch
from utils import perturb_data


def denoise_score_matching(net, X, rs):
    X = X.clone().detach()
    pX, noise = perturb_data(X, rs)  
    target = noise / (rs[:, None, None, None] ** 2)
    S = net(pX, rs) 
    loss = torch.sum((S + target) ** 2, dim=(1, 2, 3))
    return .5 * torch.mean(loss/(rs ** 2), dim=0)
