import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch.nn.functional import l1_loss

def l1_loss_w(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# def aiap_loss(x_canonical, x_deformed, n_neighbors=5):
#     """
#     Computes the as-isometric-as-possible loss between two sets of points, which measures the discrepancy
#     between their pairwise distances.

#     Parameters
#     ----------
#     x_canonical : array-like, shape (n_points, n_dims)
#         The canonical (reference) point set, where `n_points` is the number of points
#         and `n_dims` is the number of dimensions.
#     x_deformed : array-like, shape (n_points, n_dims)
#         The deformed (transformed) point set, which should have the same shape as `x_canonical`.
#     n_neighbors : int, optional
#         The number of nearest neighbors to use for computing pairwise distances.
#         Default is 5.

#     Returns
#     -------
#     loss : float
#         The AIAP loss between `x_canonical` and `x_deformed`, computed as the L1 norm
#         of the difference between their pairwise distances. The loss is a scalar value.
#     Raises
#     ------
#     ValueError
#         If `x_canonical` and `x_deformed` have different shapes.
#     """

#     # if x_canonical.shape != x_deformed.shape:
#     #     raise ValueError("Input point sets must have the same shape.")

#     # _, nn_ix, _ = knn_points(x_canonical,
#     #                          x_canonical,
#     #                          K=n_neighbors,
#     #                          return_sorted=True)

#     # dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
#     # dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

#     # loss = l1_loss(dists_canonical, dists_deformed)

#     # return loss
#     if x_canonical.shape != x_deformed.shape:
#         raise ValueError("Input point sets must have the same shape.")

#     _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
#                              x_canonical.unsqueeze(0),
#                              K=n_neighbors,
#                              return_sorted=True)

#     dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
#     dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

#     loss = l1_loss(dists_canonical, dists_deformed)

#     return loss