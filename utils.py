import torch
from torch.utils.data import Dataset
import os
import yaml


class ValueMapData(Dataset):
    def __init__(self, data_dir) -> None:
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

        X = torch.load(file_path,weights_only=True)

        return X

    def __len__(self):
        """
        Returns the number of .pt files in the dataset.

        Returns:
            int: Number of files.
        """
        return len(self.file_list)


def parse_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
