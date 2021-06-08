import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from configs import trainer_config, evaluator_config
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader, train_config: trainer_config, eval_config: evaluator_config):
        self.train_dl = train_dataloader
        self.test_dl = test_dataloader

        self.train_config = train_config
        self.eval_config = eval_config

        self.train_loss = 0
        self.test_loss = 0

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader):
        loss_func = trainer_config.loss()
        loss = 0

        for step, (ratings, indices) in enumerate(dl):
            output = model(ratings, indices)
            loss = loss_func(output, ratings)

            loss += loss.item()

        loss /= step

        return loss

    def train(self, model: Module):
        loss_func = self.train_config.loss()
        optimizer = self.train_config.optimizer(model.parameters())

        writer = SummaryWriter()

        for epoch in range(self.train_config.epochs):
            print("--------------- Epoch --------------- ", epoch)

            train_loss = 0

            for step, (ratings, indices) in enumerate(self.train_dl):
                optimizer.zero_grad()
                output = model(ratings, indices)
                loss = loss_func(output, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= step
            writer.add_scalar("MLoss/train", train_loss, epoch)

            test_loss = self.eval_loss(model, self.test_dl).item()
            writer.add_scalar("MLoss/test", test_loss, epoch)

            print("MLoss/train", train_loss, epoch)
            print("MLoss/test", test_loss, epoch)

            writer.flush()

        writer.close()
