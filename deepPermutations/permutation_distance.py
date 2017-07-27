import numpy as np
from scipy.stats import rankdata, kendalltau


def distance_from_name(distance_name: str):
    if distance_name == 'spearman':
        return spearman_rho
    elif distance_name == 'kendall':
        return kendall_tau


def spearman_rho(v1, v2, l=None):
    assert len(v1.shape) == 1
    assert len(v1) == len(v2)
    if l is None:
        l = len(v1)
        l = 128
        # l = 256

    # apply function
    f = lambda x: x
    # f = lambda x: np.power(x, 1/2)
    squash = np.vectorize(lambda rk: f(l) if rk > l else f(rk))

    # compute ranks
    r1 = squash(rankdata(-v1, method='average'))
    r2 = squash(rankdata(-v2, method='average'))

    # print number of zeros
    # print('Spearman:')
    # print(sum(v1 > 1), sum(v2 > 1))
    # print(sum(v1 > 0), sum(v2 > 0))
    # print(sum(v1 == 0), sum(v2 == 0))
    # print(sum(v1 < 0), sum(v2 < 0))

    return np.sqrt(np.sum(np.square(r1 - r2)))


def kendall_tau(v1, v2, l=None):
    assert len(v1.shape) == 1
    assert len(v1) == len(v2)
    if l is None:
        l = len(v1)
        l = 128
        # l = 256

    # apply function
    f = lambda x: x
    # f = lambda x: np.power(x, 1/2)
    squash = np.vectorize(lambda rk: f(l) if rk > l else f(rk))

    # compute ranks
    r1 = squash(rankdata(-v1, method='average'))
    r2 = squash(rankdata(-v2, method='average'))

    # print number of zeros
    print('Kendall:')
    print(sum(v1 > 1), sum(v2 > 1))
    print(sum(v1 > 0), sum(v2 > 0))
    print(sum(v1 == 0), sum(v2 == 0))
    print(sum(v1 < 0), sum(v2 < 0))

    tau, p_value = kendalltau(r1, r2)
    # tau = 1 means correlation
    return -tau
