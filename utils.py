from torch.utils.data import Dataset, DataLoader
import os


class ValueMapData(Dataset):

    def __init__(self,data_dir) -> None:
        self.data_dir = data_dir

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass




