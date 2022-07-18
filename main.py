import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
from scipy.integrate import dblquad
from scipy.misc import derivative

from spike_generation import single_neuron_spikes_bernoulli, \
    two_trains_logistic_normal


def generate_r_single_neuron(seed, rate1, rate2, t=200):
    # TODO: does representation (e.g. {-1, 1} vs. {0, 1}) affect PS measurement?
    rng = np.random.default_rng(seed)

    pattern1 = single_neuron_spikes_bernoulli(rate1, t=t, rng=rng)
    pattern2 = single_neuron_spikes_bernoulli(rate2, t=t, rng=rng)

    r, p = ss.pearsonr(pattern1, pattern2)
    return r


def cal_avg_r_single_poisson(rate1, rate2, n_trials=1000):
    rs = []
    for seed in range(n_trials):
        rs.append(generate_r_single_neuron(seed, rate1, rate2))

    rs = np.array(rs)

    return np.average(rs), np.std(rs, ddof=1) / np.sqrt(n_trials)


def fi_bernoulli(theta):
    return 1 / ((1 - theta) * theta)


def d_fi(theta1, theta2, fi=fi_bernoulli):
    return abs(fi(theta1) - fi(theta2))


def experiment1():
    d_theta0 = 2e-2
    thetas = [1e-1, 2e-1, 4e-1, 6e-1, 8e-1]
    for theta in thetas:
        factors = range(9)
        s2_list, r_list, d_theta_list, r_std_error_list = [], [], [], []

        for factor in factors:
            d_theta = d_theta0 * factor
            theta2 = theta + d_theta
            s2_list.append(d_fi(theta, theta2))
            d_theta_list.append(d_theta)
            r_avg, r_std_error = cal_avg_r_single_poisson(theta, theta2)
            r_list.append(r_avg)
            r_std_error_list.append(r_std_error)

        # plt.plot(s2_list, r_list, label="theta={}".format(theta))
        plt.errorbar(
            d_theta_list, r_list, yerr=r_std_error_list,
            label="theta={}".format(theta), capsize=5,
        )
    plt.legend()
    plt.xlabel("dl")
    plt.ylabel("r")
    plt.show()


def binarize_sign(z):
    """ Sign function for array `x`: 1 if `x_i` > 0, 0 otherwise. """
    return np.array([1 if z_i > 0 else 0 for z_i in z])


def multi_gauss_binarized_experiment(
    mu1=0, mu2=0, sigma1=1, sigma2=1, cov=.5,
    rng=np.random.default_rng(11),
):
    z = multi_gauss_sample(mu1, mu2, sigma1, sigma2, cov, rng)
    return binarize_sign(z)


def sim_eta_coordinate(mu1=0, mu2=0, sigma1=1, sigma2=1, cov=0.5, n=100000):
    eta1, eta2, eta12 = 0, 0, 0

    for i in range(n):
        x1, x2 = multi_gauss_binarized_experiment(
            mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, cov=cov,
        )

        eta1 += x1
        eta2 += x2
        eta12 += x1 * x2

    return np.array([eta1, eta2, eta12]) / n


def investigate_eta_coordinates():
    sigma = np.linspace(0.3, 1)
    eta1, eta2, eta12 = [], [], []
    for sigma_i in sigma:
        eta = sim_eta_coordinate(sigma1=sigma_i)
        eta1.append(eta[0])
        eta2.append(eta[1])
        eta12.append(eta[2])
    plt.plot(sigma, eta1, label="eta1")
    plt.plot(sigma, eta2, label="eta2")
    plt.plot(sigma, eta12, label="eta12")
    plt.legend()
    plt.show()


def visualize_multi_gauss():
    N = 500
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(N):
        z1, z2 = multi_gauss_sample(sigma1=1)
        c = 'k'
        if z1 > 0: c = 'green'
        if z2 > 0: c = 'blue'
        if z1 > 0 and z2 > 0: c = 'red'
        plt.scatter(z1, z2, c=c, s=.3)

    ax.set_aspect('equal', adjustable='box')
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def std_gauss_sig():
    rng = np.random.default_rng(11)

    N = 1000
    x = rng.normal(0, 1, N)
    y = 1 / (1 + np.exp(-x))
    plt.scatter(x, y)
    plt.show()


def multi_gauss_sample(
    mu1=0, mu2=0, sigma1=.5, sigma2=.5, cov=0, rng=np.random.default_rng(11),
):
    """ Samples once from a multi-variate Gaussian """
    mu = [mu1, mu2]
    sigma = [
        [sigma1 ** 2, sigma1 * sigma2 * cov],
        [sigma1 * sigma2 * cov, sigma2 ** 2],
    ]
    return rng.multivariate_normal(mu, sigma)


def logistic_norm_sample(
    mu1=0, mu2=0, sigma1=.5, sigma2=.5, cov=0, rng=np.random.default_rng(11),
):
    return sigmoid(
        multi_gauss_sample(
            mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2, cov=cov, rng=rng,
        )
    )


def experiment2(rng=np.random.default_rng(11)):
    N = 100000  # number of time steps
    probs = [[0, 0, 0, 0]]
    covs = [-1, -.8, -.3, 0, .3, .8, 1]

    for cov in covs:
        # neuron1, neuron2 = [], []
        for _ in range(N):
            x1, x2 = logistic_norm_sample(cov=cov, rng=rng)
            x1 = rng.binomial(1, x1)
            x2 = rng.binomial(1, x2)
            # neuron1.append(x1)
            # neuron2.append(x2)
            if x1 == 0 and x2 == 0:
                probs[-1][0] += 1
            elif x1 == 1 and x2 == 0:
                probs[-1][1] += 1
            elif x1 == 0 and x2 == 1:
                probs[-1][2] += 1
            elif x1 == 1 and x2 == 1:
                probs[-1][3] += 1
            else:
                print("ERROR!")

        plt.plot(np.array(probs[-1]) / N, label="cov={}".format(cov))
        probs.append([0, 0, 0, 0])

    plt.legend()
    plt.show()


def experiment3(rng=np.random.default_rng(12)):
    N = 1000000  # number of time steps
    covs = [0, .5, 1]
    sigma2s = [0, .5, 1]

    for i, cov in enumerate(covs):
        plt.figure(i)
        plt.title("cov = {}".format(cov))
        for sigma2 in sigma2s:
            p = [[0, 0], [0, 0]]
            for _ in range(N):
                x1, x2 = logistic_norm_sample(sigma2=sigma2, cov=cov, rng=rng)
                x1 = rng.binomial(1, x1)
                x2 = rng.binomial(1, x2)
                # neuron1.append(x1)
                # neuron2.append(x2)
                if x1 == 0 and x2 == 0:
                    p[0][0] += 1
                elif x1 == 1 and x2 == 0:
                    p[1][0] += 1
                elif x1 == 0 and x2 == 1:
                    p[0][1] += 1
                elif x1 == 1 and x2 == 1:
                    p[1][1] += 1
                else:
                    print("ERROR!")

            plt.plot(np.array(p).flatten() / N, label="s2={}".format(sigma2))
        plt.legend()

    plt.show()


def experiment4(rng=np.random.default_rng(1)):
    n = int(1e5)
    mu1, mu2 = 0, 0
    sigma1 = 1
    sigma2s = np.linspace(0.1, 1, 5)
    covs = np.linspace(0.1, 1, 5)

    for sigma2 in sigma2s:
        rs = []
        for cov in covs:
            print(cov, sigma2)
            train1, train2 = two_trains_logistic_normal(
                n, mu1, mu2, sigma1, sigma2, cov, rng
            )
            r, _ = ss.pearsonr(train1, train2)
            rs.append(r)
            plt.scatter(cov, r, c='black')
        plt.plot(covs, rs, label="sigma2={}".format(sigma2))
        plt.legend()
    plt.title("N = {}".format(n))
    plt.xlabel("Covariance")
    plt.ylabel("r")
    plt.show()


def multi_gauss_vis(rng=np.random.default_rng(11)):
    N = 100000  # number of time steps
    cov = 1
    mu1 = 0
    mu2 = 3

    x1_list, x2_list = [], []
    for _ in range(N):
        x1, x2 = multi_gauss_sample(mu1=mu1, mu2=mu2, cov=cov, rng=rng)
        x1_list.append(x1)
        x2_list.append(x2)
    plt.hist(x2_list, color='green', bins=50)
    plt.hist(x1_list, color='red', bins=50)
    plt.title("cov={}".format(cov))
    print("mu1", np.average(np.array(x1_list)))
    print("mu2", np.average(np.array(x2_list)))
    plt.show()


def logit(z):
    return np.log(np.divide(z, (1 - z)))


def logistic_pdf(l1, l2, mu1, mu2, sigma1, sigma2, cov):
    l = np.array([l1, l2])
    mu = np.array([mu1, mu2])
    diff = logit(l) - mu
    sigma = np.array([
        [sigma1 ** 2, cov * sigma1 * sigma2],
        [cov * sigma1 * sigma2, sigma2 ** 2],
    ])
    gamma = np.linalg.inv(sigma)

    jacobian = (l1 * (1 - l1) * l2 * (1 - l2)) ** -1
    det = np.sqrt(np.linalg.det(2 * np.pi * sigma)) ** -1
    pdf = np.exp(-1 * 0.5 * np.matmul(np.matmul(diff, gamma), diff))

    return jacobian * det * pdf


def exp_lambda(l1, l2, mu1, mu2, sigma1, sigma2, cov):
    return l1 * l2 * logistic_pdf(l1, l2, mu1, mu2, sigma1, sigma2, cov)


def integration():
    mu1 = 0
    mu2 = 0
    sigma1 = 0.5
    sigma2 = 0.5
    cov = 0

    I = dblquad(
        exp_lambda,
        0, 1,
        lambda l2: 0, lambda l2: 1,
        args=(mu1, mu2, sigma1, sigma2, cov),
    )
    print(I)


def main():
    rng = np.random.default_rng(11)
    experiment4(rng)


if __name__ == "__main__":
    main()
