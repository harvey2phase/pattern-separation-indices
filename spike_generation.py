from typing import Callable

import numpy as np


def single_neuron_spikes_bernoulli(
    p=.5, t=5, rng=np.random.default_rng(1),
):
    """ Generate an array representing the times at which a neuron spikes

    E.g. the array [0, 1, 0, 0, 0] represents discrete measurements at
    t = 0, ..., 4, and the neuron only fired at t = 1.

    Args:
        p: Probability that the neuron would fire at each time window.
        t: Number of time steps at which the neuron is observed
            (length of the array returned).
        rng: Random number generator.

    Return:
    """
    return rng.binomial(1, p, size=t)


def single_neuron_spikes_poisson(
    rate=.7, t=5, rng=np.random.default_rng(1),
):
    """ Generate an array representing the number of times a neuron spikes at
        each time bin

    Args:
        rng: Random number generator.

    Return:
    """
    return rng.poisson(rate, size=t)


def average(x: np.array):
    """
    Args:
        x: Space of all possible values of x.
    """
    return np.average(x)


def weibull(x, k=1, l=1):
    if x < 0: return 0
    return (k / l) * (x / l) * (k - 1) * np.exp(-(x / l) ** k * l)


def fisher_information(t, d2_log_p: Callable, expected: Callable):
    """
    Args:
        t: TODO
        d2_log_p: Second partial derivative wrt theta of log(P(x(t))), where
            P(X(t)) is a pdf that belongs to the exponential family.
        expected: Expectation value wrt to x(t).
    """
    return -expected(d2_log_p(t))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_gauss_sample(mu1, mu2, sigma1, sigma2, cov, rng):
    """ Samples once from a multi-variate Gaussian """
    mu = [mu1, mu2]
    #sigma = [
    #    [sigma1 ** 2, sigma1 * sigma2 * cov],
    #    [sigma1 * sigma2 * cov, sigma2 ** 2],
    #]
    sigma = [
        [sigma1, cov],
        [cov, sigma2],
    ]
    return rng.multivariate_normal(mu, sigma)


def logistic_norm_sample(mu1, mu2, sigma1, sigma2, cov, rng):
    return sigmoid(multi_gauss_sample(mu1, mu2, sigma1, sigma2, cov, rng))


def two_trains_logistic_normal(n, mu1, mu2, sigma1, sigma2, cov, rng):
    train1 = []
    train2 = []
    for _ in range(n):
        x1, x2 = logistic_norm_sample(mu1, mu2, sigma1, sigma2, cov, rng)
        train1.append(rng.binomial(1, x1))
        train2.append(rng.binomial(1, x2))
    return np.array(train1), np.array(train2)
