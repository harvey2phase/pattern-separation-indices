import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

from spike_generation import single_neuron_spikes_poisson


def generate_single_r(seed, rates1, rates2, t=1000):
    rng = np.random.default_rng(seed)

    pattern1 = [
        single_neuron_spikes_poisson(rate, t=t, rng=rng) for rate in rates1
    ]
    pattern2 = [
        single_neuron_spikes_poisson(rate, t=t, rng=rng) for rate in rates2
    ]

    r, p = ss.pearsonr(
        np.array(pattern1).flatten(),
        np.array(pattern2).flatten(),
    )
    return r


def cal_avg_r_poisson(rates1, rates2, n_trials=1000):
    r_sum = 0
    for seed in range(n_trials):
        r_sum += generate_single_r(seed, rates1, rates2)

    return r_sum / n_trials


def cal_diff_fi_poisson(rates1, rates2):
    fi1 = 1 / rates1
    fi2 = 1 / rates2
    return sum(abs(fi1 - fi2))


def compare_fi_r(rates1, rates2):
    return (
        cal_avg_r_poisson(rates1, rates2),
        cal_diff_fi_poisson(rates1, rates2),
    )

factors = range(9)
for factor in factors:
    rates1 = np.array([1e-2, 1e-1, 1e0, 1e1]) * factor
    rates2 = np.array([1e-3, 1e-2, 1e-1, 1e0]) * factor
    avg_r, diff_fi = compare_fi_r(rates1, rates2)
    plt.scatter(diff_fi, avg_r)
    plt.xlabel("\"FI\"")
    plt.ylabel("r")
plt.show()


# rs = []
# seed = 1
# r2_ratios = np.linspace(1e-3, 1e3, 1000)
# for r2_ratio in r2_ratios:
#    rs.append(test_r_diff_patterns(seed, r2_ratio))
#
# plt.plot(r2_ratios, rs)
# plt.xscale('log')
# plt.show()

# average r ~= 0.045 for 1e4 seeds and time bins

def compare_bernoullis():
    rng = np.random.default_rng(1)
    n = 100
    flips1, flips2 = [], []
    avg1, avg2 = [], []
    for i in range(n):
        flips1.append(rng.binomial(1, .5))
        flips2.append(rng.binomial(1, .1))
        avg1.append(sum(flips1) / len(flips1))
        avg2.append(sum(flips2) / len(flips2))

    plt.plot(avg1)
    plt.plot(avg2)
    plt.show()


def compare_poisson():
    rng = np.random.default_rng(1)
    n = 100
    flips1, flips2 = [], []
    avg1, avg2 = [], []
    for i in range(n):
        flips1.append(rng.poisson(.2))
        flips2.append(rng.poisson(2))
        avg1.append(sum(flips1) / len(flips1))
        avg2.append(sum(flips2) / len(flips2))

    plt.plot(avg1)
    plt.plot(avg2)
    plt.show()
