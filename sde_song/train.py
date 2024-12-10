import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import ValueMapData, get_marginal_prob_std_fn
from unet import ScoreNet
from loss import loss_fn
from tqdm import trange


device = torch.device('cuda')

sigma = 25.
prob_std = get_marginal_prob_std_fn(sigma, device)
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=prob_std))
score_model = score_model.to(device)

n_epochs =   10

batch_size =  4096

lr=1e-4 

dataset = ValueMapData('../data/p1')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = trange(n_epochs)
for epoch in tqdm_epoch:
  avg_loss = 0.
  num_items = 0
  for x in data_loader:
    x = x.to(torch.float32).to(device)
    loss = loss_fn(score_model, x, prob_std)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), 'modelp0.pth')