import torch
from utils import ValueMapData, parse_config
from torch.utils.data import DataLoader
from diffusion import DiffusionModel

def train():
    """Train Diffusion Model.
    TODO:
    """


def main(config):


    # Set device:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    model = DiffusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    train()



























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