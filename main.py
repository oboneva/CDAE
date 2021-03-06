import sys
from commandline import parse_args
from evaluate import Evaluator
from train import Trainer
from cdae import CDAE
from torch.utils.data.dataloader import DataLoader
from dataset import MovieLens10MDataset
import torch
from configs import data_config, model_config, trainer_config, evaluator_config
from torch.utils.tensorboard import SummaryWriter
from utils import model_metadata


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 1. Prepare the Data.
    train = MovieLens10MDataset(
        "{}/tensor_train_60.pt".format(data_config.data_dir))
    test = MovieLens10MDataset(
        "{}/tensor_test.pt".format(data_config.data_dir))
    validate = MovieLens10MDataset(
        "{}/tensor_validate.pt".format(data_config.data_dir))

    train_dl = DataLoader(
        train, batch_size=data_config.train_batch_size, shuffle=True)
    test_dl = DataLoader(
        test, batch_size=data_config.test_batch_size, shuffle=False)
    validation_dl = DataLoader(
        validate, batch_size=data_config.val_batch_size, shuffle=False)

    # 2. Define the Model.
    size = train.size()
    model = CDAE(
        model_config, users_count=size[0], items_count=size[1], device=device)

    # 3. Train the Model.
    writer = SummaryWriter(comment=model_metadata())

    trainer = Trainer(train_dl, validation_dl, trainer_config, writer)
    trainer.train(model, device)

    # 4. Evaluate the Model.
    best_model = CDAE(
        model_config, users_count=size[0], items_count=size[1], device=device)
    best_model.load_state_dict(torch.load(
        "state_dict_model{}.pt".format(model_metadata())))

    result = Evaluator().eval(model=best_model, dl=test_dl, verbose=True,
                              config=evaluator_config, writer=writer, writer_section="Test", device=device)
    print(result)

    writer.close()
    # 5. Make Predictions.


if __name__ == "__main__":
    parse_args(sys.argv[1:])
    main()
