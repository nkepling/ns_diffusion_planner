import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import yaml

class ValueMapData(Dataset):
    def __init__(self, data_dir) -> None:
        """
        Initializes the dataset by listing all .npz files in the data directory.
        
        Args:
            data_dir (str): Path to the directory containing .npz files.
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        if not self.file_list:
            raise ValueError(f"No .npz files found in the directory: {data_dir}")

    def __getitem__(self, index):
        """
        Reads and decompresses an .npz file by index.
        
        Args:
            index (int): Index of the file to read.
        
        Returns:
            tuple: A tuple containing input data and target value loaded from the .npz file.
        """
        file_path = os.path.join(self.data_dir, self.file_list[index])
        
        X = np.load(file_path)
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(X, dtype=torch.float32)
        
        return input_tensor

    def __len__(self):
        """
        Returns the number of .npz files in the dataset.
        
        Returns:
            int: Number of files.
        """
        return len(self.file_list)


def parse_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)