import torch
from torch.utils.data import Dataset
import os
import yaml


class ValueMapData(Dataset):
    def __init__(self, data_dir,ids) -> None:
        """
        Initializes the dataset by listing all .npz files in the data directory.

        Args:
            data_dir (str): Path to the directory containing .npz files.
            ids: List of indices to load for train test split
        """
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(
            data_dir) if f.endswith('.pt')]
        if not self.file_list:
            raise ValueError(
                f"No .tp files found in the directory: {data_dir}")

    def __getitem__(self, index):
        """
        Reads and decompresses an .npz file by index.

        Args:
            index (int or slice): Index of the file to read.

        Returns:
            tuple: A tuple containing input data and target value loaded from the .npz file.
        """
        if isinstance(index, slice):  # Handle slicing
            # Get the indices for the slice
            indices = range(*index.indices(len(self.file_list)))
            num_items = len(indices)  # Number of items in the slice

            # Load the first item to determine its shape
            first_file_path = os.path.join(
                self.data_dir, self.file_list[indices[0]])
            first_item = torch.load(first_file_path, weights_only=True)

            # Preallocate a tensor for all items
            items = torch.empty((num_items, *first_item.shape),
                                dtype=first_item.dtype, device=first_item.device)

            # Fill the tensor
            for i, idx in enumerate(indices):
                file_path = os.path.join(self.data_dir, self.file_list[idx])
                items[i] = torch.load(file_path, weights_only=True)

            return items

        else:  # Handle single index
            file_path = os.path.join(self.data_dir, self.file_list[index])
            X = torch.load(file_path, weights_only=True)
            return X

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
