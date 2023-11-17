import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SparseyDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to an array of integers
                matrix = [int(x) for x in line.strip().split(',')]
                # Ensure each matrix is 8x8
                matrix = np.array(matrix).reshape(8, 8)
                self.data.append(matrix)
        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])