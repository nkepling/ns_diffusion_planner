import tqdm
import torch
from torch.optim import Adam
from unet import ScoreNet
from torchvision.datasets import MNIST
from loss import sliced_score_matching, denoise_score_matching
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
from utils import gen_geometric_progression, ValueMapData


def train():
    device = torch.device('cuda')
    score_model = torch.nn.DataParallel(ScoreNet(source_channels=4))
    if os.path.exists('sanity_check.pt'):
        print('loading checkpoint')
        score_model.load_state_dict(torch.load('sanity_check.pt'))
    score_model = score_model.to(device)

    n_epochs = 40
    batch_size = 1024
    lr = 1e-4

    dataset = ValueMapData('../ns_diffusion_planner/data/p0')

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    optimizer = Adam(score_model.parameters(), lr=lr)

    rs = gen_geometric_progression(.5, .01, 50)
    rs = rs.requires_grad_(False).to(device)

    tqdm_epoch = tqdm.trange(n_epochs)
    for _ in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x.to(device).to(torch.float32)
            batch_rs = rs[torch.multinomial(rs, x.shape[0], replacement=True)].requires_grad_(False)
            loss = denoise_score_matching(score_model, x, batch_rs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            tqdm_epoch.set_description(
                'Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), 'sanity_check.pt')


if __name__ == "__main__":
    train()
