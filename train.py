import torch
from utils import ValueMapData, parse_config
from torch.utils.data import DataLoader
from diffusion import DiffusionModel

def train(config, model, optimizer, train_loader, val_loader, device):
    """Train Diffusion Model.
    TODO:
    """

    model.train()

    for epoch in range(config.num_epochs):


        for i,data in train_loader:
            x = x.to(device)

            t = torch.randint(0, config.num_timesteps, (config.batch_size,)).to(device)

            







def main(config):
    # Set device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Init model 
    model = DiffusionModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])


    # Load data
    train_indices = list(range(0, 1000)) # TODO change this to the actual number of files

    train_data = ValueMapData(config.data_dir,train_indices)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

    val_indices = list(range(1000, 1100)) # TODO change this to the actual number of files
    val_data = ValueMapData(config.data_dir, val_indices)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    # Train model
    train(config, model, optimizer, train_loader, val_loader, device)



if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a diffusion model.')
    parser.add_argument('--config', type=str, required=True,help='Path to the config file.')

    # Command line args to overwrite config

    # Training args
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, help='Learning rate for training.')

    # Data args
    parser.add_argument('--train_data_dir', type=str, help='Path to the directory containing .npz files.')
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading.')

    # Model args

    # Misc args

    args = parser.parse_args()
    config = parse_config(args.config)

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['lr'] = args.lr

    print("Config:")
    print(config)

    main(config)