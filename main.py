from evaluate import Evaluator
from train import Trainer
from cdae import CDAE
from torch.utils.data.dataset import random_split
from torch.utils.data.dataloader import DataLoader
from dataset import MovieLens10MDataset
import torch
import configs
from torch.nn import MSELoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 1. Prepare the Data.
    train = MovieLens10MDataset("./Data/tensor_train.pt")
    test = MovieLens10MDataset("./Data/tensor_test.pt")

    train_dl = DataLoader(
        train, batch_size=configs.data.train_batch_size, shuffle=True)
    test_dl = DataLoader(
        test, batch_size=configs.data.test_batch_size, shuffle=False)

    # 2. Define the Model.
    size = train.size()
    model = CDAE(model_conf=configs.model,
                 users_count=size[0], items_count=size[1], device=device)

    # 3. Train the Model.
    loss = MSELoss()
    Trainer.train(model=model, loss_func=loss, train_dataloader=train_dl,
                  config=configs.trainer)

    # 4. Evaluate the Model.
    for k in configs.evaluator.top_k_list:
        result = Evaluator().map_at_k(model, test_dataloader=test_dl, k=k)

        print(f"Avg Precision at {k} ", result)

    # 5. Make Predictions.


if __name__ == "__main__":
    main()
