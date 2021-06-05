import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch import Tensor


class Evaluator:
    def _batch_precision_at_k(self, batch_rec_list, batch_target_list, k):
        batch_top_k_results = torch.gather(
            batch_target_list, 1, batch_rec_list)

        batch_top_k_results_good_ones = torch.sum(batch_top_k_results[:, ], 1)

        result = batch_top_k_results_good_ones / k
        result = result.unsqueeze(-1)

        return result

    def _recall_at_k(self, rec_indicies, target_list, k):
        asd = sum([1 for index in rec_indicies if index < len(target_list)])

        adopted = sum([1 for item in target_list if item.item() == 1])

        return asd / len(adopted)

    def _batch_avg_precision_at_k(self, batch_rec_list, batch_target_list, k):
        batch_len = len(batch_rec_list)
        batch_precisions = torch.zeros(batch_len, 1)

        adopted = torch.sum(batch_target_list[:, 1:], 1)

        for index in range(0, k):
            _, batch_top_k_indicies = torch.topk(
                input=batch_rec_list, k=index + 1)

            target_top_k_index = batch_top_k_indicies[:, index]
            # 1 if the item at rank k is adopted, otherwise zero
            target_top_k_index = target_top_k_index.unsqueeze(-1)

            batch_rel = torch.gather(batch_target_list, 1, target_top_k_index)

            batch_p_at_k = self._batch_precision_at_k(
                batch_top_k_indicies, batch_target_list, k=index)

            batch_precisions += batch_p_at_k * batch_rel

        return batch_precisions / min(k, len(adopted))

    def map_at_k(self, model: Module, test_dataloader: DataLoader, k: int):
        avg_precision = 0
        users = 0
        for i, (ratings, indicies) in enumerate(test_dataloader):
            output = model(ratings, indicies)

            batch_precisions = self._batch_avg_precision_at_k(
                output, ratings, k=k)
            avg_precision += torch.sum(batch_precisions)
            users += len(ratings)

        return avg_precision / users


def main():
    # evaluator = Evaluator()
    # batch_rec_list = torch.tensor([[0, 1],
    #                                [0, 2],
    #                                [1, 2]])
    # batch_rec_list.to(device="cpu")

    # batch_target_list = torch.tensor([[1., 0., 0.],
    #                                   [0., 1., 0.],
    #                                   [0., 1., 1.]])
    # batch_target_list.to(device="cpu")

    # k = 1  # -> tensor([1., 1., 1.])
    # k = 2  # -> tensor([0.5000, 0.0000, 1.0000])

    # print(evaluator._batch_precision_at_k(
    #     batch_rec_list, batch_target_list, k))

    # batch_p_at_k = torch.tensor([[0.5000],
    #                              [0.0000],
    #                              [1.0000]])

    # rel = torch.tensor([[0],
    #                     [1],
    #                     [1]])

    # asd = torch.mul(batch_p_at_k, rel)
    # print(asd)

    # a = torch.tensor([[1],
    #                   [2],
    #                   [3]])

    # b = torch.tensor([[1],
    #                   [2],
    #                   [3]])

    # a += b
    # print(a)

    # print(a / 2)
    pass


if __name__ == "__main__":
    main()
