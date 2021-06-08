from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.loss import BCELoss
from torch.optim.adam import Adam


class model_config:
    latent_representation_dim = 50
    corruption_ratio = 0.5
    hidden_mapping_function = Sigmoid()
    output_mapping_function = Sigmoid()


class data_config:
    train_batch_size = 512
    test_batch_size = 1024
    val_batch_size = 1024


class trainer_config:
    epochs = 500
    optimizer = Adam
    loss = BCELoss


class evaluator_config:
    top_k_list = [1, 5, 10]
