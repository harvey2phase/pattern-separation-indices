import numpy as np


def heaviside(alpha) -> int:
    return 1 if alpha > 0 else 0


def eps(delta_t, tau_m=10) -> float:
    return np.exp(-delta_t / tau_m) * heaviside(delta_t)


def input_potential(h_i=0):
    """ TODO """
    return h_i


def refactory(delta_t, ref_t=200):
    return delta_t / (ref_t + delta_t)


def sigmoid(u, gm=500, beta=8, uc=1):
    return gm * (1 + np.exp(-beta * (u - uc))) ** -1


def instantaneous_firing_pdf(t, t_i, input_potential=0):
    return sigmoid(input_potential) * refactory(t - t_i)
