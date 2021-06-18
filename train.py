from evaluate import Evaluator
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from configs import evaluator_config, trainer_config
from utils import model_metadata


class Trainer:
    def __init__(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, train_config: trainer_config, writer):
        self.train_dl = train_dataloader
        self.val_dl = validation_dataloader

        self.train_config = train_config

        self.min_val_loss = 1
        self.patience = train_config.patience
        self.no_improvement_epochs = 0

        self.writer = writer

    @torch.no_grad()
    def eval_loss(self, model: Module, dl: DataLoader, device):
        loss_func = trainer_config.loss()
        loss = 0

        for step, (ratings, indices) in enumerate(dl):
            ratings = ratings.to(device)
            indices = indices.to(device)

            output = model(ratings, indices)
            loss = loss_func(output, ratings)

            loss += loss.item()

        loss /= step

        return loss

    def train(self, model: Module, device):
        loss_func = self.train_config.loss()
        optimizer = self.train_config.optimizer(model.parameters())

        for epoch in range(self.train_config.epochs):
            print("--------------- Epoch {} --------------- ".format(epoch))

            train_loss = 0

            for step, (ratings, indices) in enumerate(self.train_dl):
                optimizer.zero_grad()

                ratings = ratings.to(device)
                indices = indices.to(device)

                output = model(ratings, indices)
                loss = loss_func(output, ratings)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= step
            self.writer.add_scalar("MLoss/train", train_loss, epoch)

            val_loss = self.eval_loss(model, self.val_dl, device).item()
            self.writer.add_scalar("MLoss/validation", val_loss, epoch)

            print("MLoss/train", train_loss)
            print("MLoss/validation", val_loss)

            self.writer.flush()

            Evaluator().eval(model=model, dl=self.val_dl, verbose=False,
                             config=evaluator_config, writer=self.writer,
                             writer_section="Validation", device=device)

            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.no_improvement_epochs = 0

                print("New minimal validation loss", val_loss)

                torch.save(model.state_dict(),
                           "state_dict_model{}.pt".format(model_metadata()))
            elif self.no_improvement_epochs == self.patience:
                print("Early stoping on epoch {}".format(epoch))

                break
            else:
                self.no_improvement_epochs += 1
