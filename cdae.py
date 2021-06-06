from torch.tensor import Tensor
from configs import model
from torch import nn
import torch.nn.functional as func
from torch import device


class CDAE(nn.Module):
    def __init__(self, model_conf: model, users_count: int, items_count: int, device: device):
        super(CDAE, self).__init__()

        self.hidden_dim = model_conf.latent_representation_dim
        self.corruption_ratio = model_conf.corruption_ratio
        self.users_count = users_count
        self.items_count = items_count
        self.device = device

        self.user_embedding = nn.Embedding(
            num_embeddings=self.users_count, embedding_dim=self.hidden_dim)
        self.hidden_mapping_function = model_conf.hidden_mapping_function
        self.output_mapping_function = model_conf.output_mapping_function
        self.encoder = nn.Linear(
            in_features=self.items_count, out_features=self.hidden_dim)
        self.decoder = nn.Linear(
            in_features=self.hidden_dim, out_features=self.items_count)

        self.to(self.device)

    def forward(self, ratings: Tensor, indices):
        corrupted_ratings = func.dropout(ratings)

        hidden = self.encoder(corrupted_ratings) + self.user_embedding(indices)
        hidden = self.hidden_mapping_function(hidden)
        output = self.decoder(hidden)
        output = self.output_mapping_function(output)

        return output
