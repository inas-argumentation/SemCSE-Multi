import numpy as np
import numba as nb

EPSILON_DBL = 1e-8
PERPLEXITY_TOLERANCE = 1e-5

@nb.njit(parallel=True)
def _binary_search_perplexity(sqdistances, desired_perplexity):
    n_steps = 100
    n_samples, n_neighbors = sqdistances.shape
    using_neighbors = n_neighbors < n_samples
    desired_entropy = np.log(desired_perplexity)

    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)
    betas = np.zeros(n_samples, dtype=np.float64)

    for i in nb.prange(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        for _ in range(n_steps):
            sum_Pi = 0.0
            for j in range(n_neighbors):
                if j != i or using_neighbors:
                    P[i, j] = np.exp(-sqdistances[i, j] * beta)
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = EPSILON_DBL

            sum_disti_Pi = 0.0
            for j in range(n_neighbors):
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            if np.abs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            if entropy_diff > 0.0:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2.0
                else:
                    beta = (beta + beta_max) / 2.0
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2.0
                else:
                    beta = (beta + beta_min) / 2.0

        betas[i] = beta

    return P, betas