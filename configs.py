from torch.nn.modules.activation import ReLU, Sigmoid
from torch.optim.adagrad import Adagrad


class model:
    latent_representation_dim = 50
    corruption_ratio = 0.5
    hidden_mapping_function = Sigmoid()
    output_mapping_function = Sigmoid()


class data:
    train_batch_size = 32
    test_batch_size = 1024


class trainer:
    epochs = 10
    learning_rate = 0.01
    optimizer = Adagrad


class evaluator:
    top_k_list = [1, 5, 10]
