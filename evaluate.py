import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from configs import evaluator_config


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

    def _batch_avg_precision_at_k(self, batch_rec_list, batch_target_list, k, device):
        batch_len = len(batch_rec_list)
        batch_precisions = torch.zeros(batch_len, 1).to(device)

        adopted = torch.sum(batch_target_list, 1)

        for index in range(0, k):
            _, batch_top_k_indicies = torch.topk(
                input=batch_rec_list, k=index + 1)

            target_top_k_index = batch_top_k_indicies[:, index]
            # 1 if the item at rank k is adopted, otherwise zero
            target_top_k_index = target_top_k_index.unsqueeze(-1)

            batch_rel = torch.gather(batch_target_list, 1, target_top_k_index)

            batch_p_at_k = self._batch_precision_at_k(
                batch_top_k_indicies, batch_target_list, k=index + 1)

            batch_precisions += batch_p_at_k * batch_rel

        return batch_precisions / min(k, len(adopted))

    @torch.no_grad()
    def map_at_k(self, model: Module, test_dataloader: DataLoader, k: int, device):
        avg_precision = 0
        users = 0
        for i, (ratings, indicies) in enumerate(test_dataloader):

            ratings = ratings.to(device)
            indicies = indicies.to(device)

            output = model(ratings, indicies)

            batch_precisions = self._batch_avg_precision_at_k(
                output, ratings, k=k, device=device)
            avg_precision += torch.sum(batch_precisions)
            users += len(ratings)

        return avg_precision / users

    def eval(self, model: Module, dl: DataLoader, config: evaluator_config, verbose: bool, writer, device):
        results = []

        for k in config.top_k_list:
            result = self.map_at_k(
                model, test_dataloader=dl, k=k, device=device)
            results.append(result)

            writer.add_scalar("Test/MAP@{}".format(k), result)
            writer.flush()

            if verbose:
                print(f"Mean Avg Precision at {k} ", result)

        return results


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
