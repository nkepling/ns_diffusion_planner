import tqdm
import torch
from torch.optim import Adam
from ..sde_song.unet import ScoreNet
from loss import denoise_score_matching
from torch.utils.data import DataLoader
import os
from utils import gen_geometric_progression, ValueMapData


def train():
    device = torch.device('cuda')
    score_model = torch.nn.DataParallel(ScoreNet(lambda x: x))
    if os.path.exists('sanity_check.pt'):
        print('loading checkpoint')
        score_model.load_state_dict(torch.load('sanity_check.pt', weights_only=True))
    score_model = score_model.to(device)

    n_epochs = 10
    batch_size = 1024
    lr = 1e-4

    dataset = ValueMapData('../data/p0')

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    optimizer = Adam(score_model.parameters(), lr=lr)

    rs = gen_geometric_progression(10, .1, 1000)
    rs = rs.requires_grad_(False).to(device)

    tqdm_epoch = tqdm.trange(n_epochs)
    for _ in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x.to(device).to(torch.float32)
            batch_rs = rs[torch.randint(low=0, high=len(rs), size=(x.shape[0], ))].requires_grad_(False)
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
