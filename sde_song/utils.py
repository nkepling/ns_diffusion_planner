import torch
import numpy as np

def get_marginal_prob_std_fn(sigma, device):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """

    def marginal_prob_std(t):
        nonlocal sigma, device
        t = t.clone().detach().to(device) 
        return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

    return marginal_prob_std

def get_diffusion_coeff_fn(sigma, device):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    def get_diffusion_coeff(t):
       nonlocal sigma, device 
       return torch.tensor(sigma**t, device=device)

    return get_diffusion_coeff

from torch.utils.data import Dataset
import os
import yaml


class ValueMapData(Dataset):
    def __init__(self, data_dir,indices=None) -> None:
        """
        Initializes the dataset by listing all .pt files in the data directory.

        Args:
            data_dir (str): Path to the directory containing .pt files.
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(
            data_dir) if f.endswith('.pt')]
        if not self.file_list:
            raise ValueError(
                f"No .pt files found in the directory: {data_dir}")
        
        if indices is not None:
            self.file_list = [self.file_list[i] for i in indices]
            
    def __getitem__(self, index):
        """
        Reads and decompresses an .pt file by index.

        Args:
            index (int): Index of the file to read.

        Returns:
            tuple: A tuple containing input data and
                   target value loaded from the .pt file.
        """
        file_path = os.path.join(self.data_dir, self.file_list[index])

        X = torch.load(file_path, weights_only=True)

        return X

    def __len__(self):
        """
        Returns the number of .pt files in the dataset.

        Returns:
            int: Number of files.
        """
        return len(self.file_list)
