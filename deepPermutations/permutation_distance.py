from scipy.stats import rankdata
import numpy as np

def spearman_rho(v1, v2):
    assert len(v1.shape) == 1
    assert len(v1) == len(v2)
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


def permutation_distance(input_1, input_2, distance_model):
    return spearman_rho(distance_model.predict(input_1, batch_size=1)[0],
                        distance_model.predict(input_2, batch_size=2)[0])

