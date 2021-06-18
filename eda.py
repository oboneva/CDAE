import sys
from commandline import parse_args
from configs import data_config
import pandas as pd

# UserID::MovieID::Rating::Timestamp


def main():
    df = pd.read_csv(
        "{}/ratings.dat".format(data_config.data_dir), names=["all"])

    df["rating"] = df["all"].apply(lambda x: x.split("::")[2])
    df["movieId"] = df["all"].apply(lambda x: x.split("::")[1])
    df["userId"] = df["all"].apply(lambda x: x.split("::")[0])

    df = df.drop(columns=['all'], index=1)

    print(df.head())

    users = df.groupby(["userId"]).size().count()
    print("Number of users", users)

    movies = df.groupby(["movieId"]).size().count()
    print("Number of users", movies)

    ratings = len(df.index)
    print("Number of ratings", ratings)

    density = (ratings / (users * movies)) * 100
    print("Data density is {:.2f}% and data sparsity is {:.2f}%".format(
        density, 100 - density))


if __name__ == "__main__":
    parse_args(sys.argv[1:])
    main()
