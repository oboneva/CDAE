import torch
import numpy as np
import pandas as pd
from torch.optim.adagrad import Adagrad
from torch.utils.data import Dataset


class MovieLens10MDataset(Dataset):
    def __init__(self, path: str):
        self.matrix = torch.load(path)

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, index):
        return (self.matrix[index], index)

    def size(self):
        return self.matrix.size()


def main():
    dataset = MovieLens10MDataset("Data/tensor_train.pt")
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    main()
