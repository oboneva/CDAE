import pandas as pd

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


if __name__ == "__main__":
    preprocess_ratings()
