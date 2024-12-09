import torch
import unet
import torch.autograd as autograd
from utils import perturb_data

# num_slices = 1
def sliced_score_matching(net: unet.ScoreNet, X: torch.Tensor, rs: torch.Tensor):
    X = perturb_data(X, rs)
    X = X.clone().detach().requires_grad_(True)  # shape: B,C,W,H
    S = net(X, rs).flatten(start_dim=1)  # shape: B,CWH
    V = torch.randn_like(S)  # shape: B,CWH
    VS = torch.sum(V * S, dim=1)  # shape: B,
    gVS = torch.autograd.grad(torch.sum(VS), X, create_graph=True)[
        0].flatten(start_dim=1)  # shape: B, CWH

    lhs = torch.sum(V * gVS, dim=1)  # shape: B
    rhs = torch.sum(S ** 2, dim=1) * 0.5  # shape: B
    return ((lhs + rhs) * rs**2).mean()


def denoise_score_matching(net, X, rs):
    X = X.clone().detach()
    pX = perturb_data(X, rs)  # pX = X + rs * epsilon
    # Correct target: (X - pX)/rs^2 = -epsilon/rs
    target = (X - pX) / (rs[:, None, None, None] ** 2)
    
    S = net(pX, rs) # Make sure net takes pX (noisy input), not X!
    loss = torch.sum((S - target) ** 2, dim=(1, 2, 3))
    
    return .5 * torch.mean(loss/(rs ** 2), dim=0)
