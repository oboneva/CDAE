import torch
import numpy as np
import pandas as pd
from torch.optim.adagrad import Adagrad
from torch.utils.data import Dataset


class MovieLens10MDataset(Dataset):
    def __init__(self, csv_file: str):
        df = pd.read_csv(csv_file)

        list_of_tensors = [torch.tensor(np.array(row))
                           for row in df.itertuples(index=False)]
        self.matrix = torch.stack(list_of_tensors).float()

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, index):
        row = self.matrix[index]

        positive_indicies = torch.nonzero(row)
        positive_indicies = positive_indicies[1:]  # remove the useId index
        negative_indicies = (row == 0).nonzero()

        return (row, positive_indicies, negative_indicies, index)

    def size(self):
        return self.matrix.size()


def main():
    dataset = MovieLens10MDataset("./Data/matrix-small.csv")
    dataset.__getitem__(1)


if __name__ == "__main__":
    main()
