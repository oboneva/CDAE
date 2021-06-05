import numpy as np
import pandas as pd
import torch
from torch.tensor import Tensor

# UserID::MovieID::Rating::Timestamp


def data_to_user_item_matrix():
    df = pd.read_csv("./ml-10M100K/ratings.dat", names=["all"])

    df["rating"] = df["all"].apply(lambda x: x.split("::")[2])
    df["movieId"] = df["all"].apply(lambda x: x.split("::")[1])
    df["userId"] = df["all"].apply(lambda x: x.split("::")[0])

    df = df.drop(columns=['all'], index=1)

    df = df.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    df.to_csv("user-item-matrix.csv")


def preprocess_ratings():
    df = pd.read_csv("user-item-matrix.csv", index_col="userId")

    df = df.applymap(lambda x: 0 if x < 4 else 1)

    df.to_csv("user-item-matrix-preprocessed.csv")


def get_positive_indicies(tensor: Tensor):
    positive_indicies = torch.nonzero(tensor)
    positive_indicies = positive_indicies[1:]  # remove the userId index

    return positive_indicies


def split_ratings():
    df = pd.read_csv("Data/user-item-matrix-preprocessed.csv")

    list_of_tensors = [torch.tensor(np.array(row))
                       for row in df.itertuples(index=False)]

    positive_indicies = [torch.flatten(
        get_positive_indicies(tensor)) for tensor in list_of_tensors]

    indices_shuffled = [torch.randperm(len(item))
                        for item in positive_indicies]

    indicies_train_test = [(item[:int(len(item) * 0.8)], item[int(len(item) * 0.8):])
                           for item in indices_shuffled]

    positive_train = []
    positive_test = []
    for i in range(len(indicies_train_test)):
        indicies_train, indicies_test = indicies_train_test[i]

        train = positive_indicies[i][indicies_train]
        test = positive_indicies[i][indicies_test]

        positive_train.append(train)
        positive_test.append(test)

    train_matrix = torch.zeros(len(list_of_tensors), len(list_of_tensors[0]))
    test_matrix = torch.zeros(len(list_of_tensors), len(list_of_tensors[0]))

    for user in range(len(list_of_tensors)):
        train = torch.tensor(positive_train[user])
        test = torch.tensor(positive_test[user])

        train_matrix[user].index_fill_(0, train, 1)
        test_matrix[user].index_fill_(0, test, 1)

    torch.save(train_matrix, 'Data/tensor_train.pt')
    torch.save(test_matrix, 'Data/tensor_test.pt')


if __name__ == "__main__":
    split_ratings()
