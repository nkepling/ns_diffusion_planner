import torch as tr
from torch import nn

class DiffusionModel(nn.Module):

  def __init__(self, design_matrix: tr.Tensor, unet) -> None:
    self.X = design_matrix
    (self.N, self.D) = design_matrix.shape
    
    self.unet = unet

  def forward(self, x):

    return 0
  
  def sliced_score_matching(self):
    X = self.X.clone().detach().requires_grad_(True)
    V = tr.randn((self.N, self.D))
    S = self.forward(self.X)
    VS = tr.sum(V * S, axis=1)
    rhs = .5 * VS ** 2
    gVS = tr.autograd.grad(VS, self.X, grad_outputs=tr.ones_like(VS),create_graph=True)[0]
    lhs = tr.sum(V * gVS, axis=1)    
    F_Divergence = tr.mean(lhs - rhs)
    return F_Divergence
