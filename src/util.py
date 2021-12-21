import torch


def kullback_leibler_div(m1: torch.Tensor, s1: torch.Tensor, m2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """
     Computes the Kullback-Leibler Divergence of two multivariate gaussians D(P_1 || P_2)
     Implementation based on https://stanford.edu/~jduchi/projects/general_notes.pdf

     :param m1 mean of gaussian 1
     :param s1 covariance matrix of gaussian 1
     :param m2 mean of gaussian 2
     :param s2 covariance matrix of gaussian 2
     :return Kullback-leiber divergence beetween gaussian 1 and gaussian 2
    """
    # todo get dim
    n = m1.size()

    # todo: check if this implementation is correct
    kl = 0.5 * (torch.log(torch.det(s2) / torch.det(s1)) - n + torch.trace(torch.inverse(s2) * s1)
                + (m2 - m1).t() * torch.inverse(s2) * (m2 - m1))

    return kl
