import torch
from torch import Tensor
import torch.nn.functional as F


def kmeans(samples: Tensor, num_clusters: int, num_iters: int = 10, use_cosine_sim: bool = False):
    """
    Args:
        samples: A Tensor of shape [N, D]
        num_clusters: Number of clusters
        num_iters: Number of iterations kmeans
        use_cosine_sim: Use cosine similarity as the distance function

    Returns:
        A Tensor of shape [num_clusters, D] representing cluster centers

    """
    N, D = samples.shape
    assert N >= num_clusters, f'number of samples should >= number of clusters, get {N} and {num_clusters}'

    means = samples[torch.randperm(N, device=samples.device)[:num_clusters]]                    # [K, D]
    indices = torch.zeros((N, ), dtype=samples.dtype, device=samples.device)                    # [N]

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = -samples @ means.T                                                          # [N, K]
        else:
            dists = (torch.sum(samples ** 2, dim=-1, keepdim=True) +
                     torch.sum(means ** 2, dim=-1) -
                     2 * torch.mm(samples, means.T))                                            # [N, K]

        indices = torch.argmin(dists, dim=-1)                                                   # [N]
        onehot = F.one_hot(indices, num_classes=num_clusters)                                   # [N, K]
        zero_mask = torch.eq(torch.sum(onehot, dim=0), 0)                                       # [K]

        new_means = torch.zeros_like(means)                                                     # [K, D]
        new_means.scatter_add_(dim=0, index=indices[:, None].repeat(1, D), src=samples)         # [K, D]
        new_means = new_means / onehot.sum(dim=0)[:, None]                                      # [K, D]
        if use_cosine_sim:
            new_means = F.normalize(new_means, p=2, dim=-1)

        means = torch.where(zero_mask[:, None], means, new_means)

    return means, indices


def _test():
    import matplotlib.pyplot as plt
    data = torch.cat([
        torch.randn((100, 2)) * 0.5 + 0,
        torch.randn((40, 2)) * 1.5 + 6,
        torch.randn((10, 2)) * 1 - 4,
    ])
    means, indices = kmeans(data, num_clusters=3)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(data[indices == 0, 0], data[indices == 0, 1])
    ax.scatter(data[indices == 1, 0], data[indices == 1, 1])
    ax.scatter(data[indices == 2, 0], data[indices == 2, 1])
    plt.show()


if __name__ == '__main__':
    _test()
