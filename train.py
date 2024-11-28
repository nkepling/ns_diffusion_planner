import torch
import os
from utils import ValueMapData, parse_config
from torch.utils.data import DataLoader
from diffusion import DiffusionModel
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def plot_loss(loss_log):
    os.makedirs('plots/', exist_ok=True)
    plt.plot(loss_log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plots/loss.png')


def train(model, optimizer, data, epochs):
    loss_log = []
    for ep in range(epochs):
        if ep % 100 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_epoch_{ep}.pt')
        i = 1
        epoch_loss = []
        for X in data:
            X = X.to(torch.float32).to(device)
            # assert X.device == model.device,f"got {X.device} expected {model.device}"
            # Sample random times t.
            # These times t are applied to each map in the batch
            t = model.ts[torch.randint(
                0, len(model.ts), (X.shape[0],))].to(device)
            # calculate divergence and take step
            F_divergence = model.sliced_score_matching(X, t)
            optimizer.zero_grad()
            F_divergence.backward()
            optimizer.step()

            print(f"epoch: {ep: 0{4}d}   ",
                  f"batch: {i: 0{4}d}    ",
                  f"t: {t[0].item(): .4f}    ",
                  f"loss: {F_divergence.item(): .4f}    ")
            i += 1
            epoch_loss.append(F_divergence.item())

        loss_log.append(np.mean(epoch_loss))

    return loss_log


def main(config):
    lr = config['lr']
    data_dir = config['data_dir']
    epochs = config['epochs']
    batch_size = config['batch_size']

    model = DiffusionModel(UNet(), device)
    model.to(device)

    # print(model.device)
    # model.double()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)

    data = ValueMapData(data_dir)
    data_loader = DataLoader(data, batch_size=batch_size, pin_memory=True)

    os.makedirs('checkpoints/', exist_ok=True)
    loss_log = train(model, optimizer, data_loader, epochs)

    plot_loss(loss_log)

    os.makedirs('models/', exist_ok=True)
    torch.save(model.state_dict(), 'models/model1.pt')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a diffusion model.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config file.')

    # Command line args to overwrite config

    # Training args
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, help='Learning rate for training.')

    # Data args
    parser.add_argument('--train_data_dir', type=str,
                        help='Path to the directory containing .npz files.')
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers for data loading.')

    # Model args

    # Misc args

    args = parser.parse_args()
    config = parse_config(args.config)

    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr

    print("Config:")
    print(config)
    print("device:", device)

    main(config)
