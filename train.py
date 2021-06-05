from torch.optim import Optimizer, Adagrad
from torch.utils.data import DataLoader
from torch.nn import Module
from configs import trainer


class Trainer:
    @staticmethod
    def train(model: Module, loss_func, train_dataloader: DataLoader, config: trainer):
        optimizer = config.optimizer(
            model.parameters(), lr=config.learning_rate)
        for epoch in range(config.epochs):
            for step, (row, positive_indicies, negative_indicies, index) in enumerate(train_dataloader):

                optimizer.zero_grad()
                output = model(ratings, indices)
                loss = loss_func(output, ratings[:, 1:])
                loss.backward()
                optimizer.step()

                print("--------------- Step --- ", step)
            print("--------------- Epoch --- ", epoch)
