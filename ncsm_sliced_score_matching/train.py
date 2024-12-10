import torch
import os
from utils import ValueMapData, parse_config
from torch.utils.data import DataLoader
from diffusion import DiffusionModel
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import sys


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def train_val_test_split(val_ratio, test_ratio, data_dir, shuffle=True):
    """Splits the data into train, validation and test sets.

    Returns: List of indices for train, validation and test sets.
    """
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    if not file_list:
        raise ValueError(
            f"No .pt files found in the directory: {data_dir}")
    n = len(file_list)
    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)
    val_indices = indices[:int(n*val_ratio)]
    test_indices = indices[int(n*val_ratio):int(n*(val_ratio+test_ratio))]
    train_indices = indices[int(n*(val_ratio+test_ratio)):]

    return train_indices, val_indices, test_indices


def test(model, data):
    model.eval()

    loss = []

    for X in tqdm(data, desc="testing", ascii=" >=", leave=False):
        X = X.to(torch.float32).to(device)
        t_int = torch.randint(0, model.T, (X.shape[0],), device=device)
        # calculate divergence and take step
        Xt = model.perturb_data(X, t_int)
        F_divergence = model.sliced_score_matching(Xt, t_int, 10)
        loss.append(F_divergence.item())

    return np.mean(loss)


def plot_loss(loss_log, title='Loss'):
    os.makedirs('plots/', exist_ok=True)
    plt.plot(loss_log)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('plots/' + f"{title}.png")


def train(model: DiffusionModel, optimizer, train_data, val_data, epochs):
    loss_log = []
    val_loss_log = []
    model.train()

    for ep in trange(epochs, desc="epoch", ascii=" >=", leave=False):
        epoch_loss = []

        model.train()
        for X in tqdm(train_data, desc="epoch_batch", ascii=" >=", leave=False):
            X = X.to(torch.float32).to(device)
            X = torch.mean(X, dim=1, keepdim=True)
            # assert X.device == model.device,f"got {X.device} expected {model.device}"
            # Sample random times t.
            # These times t are applied to each map in the batch
            t_int = torch.randint(0, model.T, (X.shape[0],), device=device)
            # calculate divergence and take step
            Xt = model.perturb_data(X, t_int)
            F_divergence = model.sliced_score_matching(Xt, t_int, 10)
            optimizer.zero_grad()
            F_divergence.backward()
            optimizer.step()
            tqdm.write(f"train loss: {F_divergence: .4f}")

            epoch_loss.append(F_divergence.item())

        loss_log.append(np.mean(epoch_loss))
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{ep}.pt')

        # Validation loss
        model.eval()
        val_loss = []
        for X in tqdm(val_data, desc="epoch_val", ascii=" >=", leave=False):
            X = X.to(torch.float32).to(device)
            # t = model.ts[torch.randint(
            #     0, len(model.ts), (X.shape[0],))].to(device)
            t_int = torch.randint(0, model.T, (X.shape[0],), device=device)
            Xt = model.perturb_data(X, t_int)
            F_divergence = model.sliced_score_matching(X, t_int, 10)
            val_loss.append(F_divergence.item())

        val_loss_log.append(np.mean(val_loss))

        tqdm.write(f"epoch: {ep: 0{4}d}   \
                     train loss: {loss_log[-1]: .4f}    \
                     val loss: {val_loss_log[-1]: .4f}")

        if np.mean(epoch_loss) == 0:
            print('early termination')
            break

    return loss_log, val_loss_log


def main(config):
    lr = config['lr']
    data_dir = config['data_dir']
    epochs = config['epochs']
    batch_size = config['batch_size']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']

    unet = UNet()
    if config['checkpoint'] is not None:
        unet.load_state_dict(torch.load(
            config['checkpoint'], weights_only=True))
    model = DiffusionModel(unet, device)
    model.to(device)

    # print(model.device)
    # model.double()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr)

    # Split the data into train, validation and test sets
    train_ind, val_ind, test_ind = train_val_test_split(
        val_ratio, test_ratio, data_dir, shuffle=True)

    # Create data loaders
    train_data = ValueMapData(data_dir, train_ind)
    train_data_loader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=True, num_workers=12)

    val_data = ValueMapData(data_dir, val_ind)
    val_data_loader = DataLoader(
        val_data, batch_size=batch_size, pin_memory=True, num_workers=12)

    test_data = ValueMapData(data_dir, test_ind)
    test_data_loader = DataLoader(
        test_data, batch_size=batch_size, pin_memory=True, num_workers=12)

    os.makedirs('checkpoints/', exist_ok=True)
    loss_log, val_loss_log = train(
        model, optimizer, train_data_loader, val_data_loader, epochs)

    plot_loss(loss_log, 'train_loss')
    plot_loss(val_loss_log, 'val_loss')

    os.makedirs('models/', exist_ok=True)
    torch.save(model.state_dict(), config["model_save_path"])

    # Test loss

    test_loss = test(model, test_data_loader)
    print(f"Test loss: {test_loss}")


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
    parser.add_argument('--checkpoint', type=int,
                        help='checkpoint.')

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
    config['checkpoint'] = args.checkpoint

    print("Config:")
    print(config)
    print("device:", device)

    main(config)
