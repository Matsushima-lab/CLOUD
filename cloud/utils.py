"""
This file is based on the implementation by K Budhathoki.
https://github.molgen.mpg.de/EDA/cisc
"""
import sys
import numpy as np
from math import ceil, log, sqrt
from collections import Counter, defaultdict
import itertools
from sklearn.preprocessing import LabelEncoder


def log2(n):
    return log(n or 1, 2)


def log_C_CAT(n: int, K: int):
    """Computes the normalizing term of NML distribution recursively. O(√n + K)
    For more detail, please refer to eq (19) (Theorem1) in
    "NML Computation Algorithms for Tree-Structured Multinomial Bayesian Networks"
    https://pubmed.ncbi.nlm.nih.gov/18382603/
    and algorithm 2 in
    "Computing the Multinomial Stochastic Complexity in Sub-Linear Time"
    http://pgm08.cs.aau.dk/Papers/31_Paper.pdf
    
    Args:
    ----------
        n (int): sample size of a dataset
        K (int): K-value multinomal distribution
    Returns:
    ----------
        float: (Approximated) Multinomal Normalizing Sum
    """

    total = 1
    b = 1
    d = 10 # 10 digit precision

    bound = int(ceil(2 + sqrt(2 * n * d * log(10))))  # using equation (38)

    for k in range(1, bound + 1):
        b = (n - k + 1) / n * b
        total += b

    log_old_sum = log2(1.0)
    log_total = log2(total)
    log_n = log2(n)
    for j in range(3, K + 1):
        log_x = log_n + log_old_sum - log_total - log2(j - 2)
        x = 2 ** log_x
        log_one_plus_x = log2(1 + x)
        log_new_sum = log_total + log_one_plus_x
        log_old_sum = log_total
        log_total = log_new_sum

    if K == 1:
        log_total = log2(1.0)

    return log_total


# ref: https://github.molgen.mpg.de/EDA/cisc/blob/master/formatter.py
def stratify(X, Y):
    """Stratifies Y based on unique values of X.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): list of Y-values for a X-value
    """
    Y_grps = defaultdict(list)
    for i, x in enumerate(X):
        Y_grps[x].append(Y[i])
    return Y_grps


def map_to_majority(X, Y):
    """Creates a function that maps x to most frequent y.
    Args:
        X (sequence): sequence of discrete outcomes
        Y (sequence): sequence of discrete outcomes
    Returns:
        (dict): map from Y-values to frequently co-occuring X-values
    """
    f = dict()
    Y_grps = stratify(X, Y)
    for x, Ys in Y_grps.items():
        frequent_y, _ = Counter(Ys).most_common(1)[0]
        f[x] = frequent_y
    return f


def update_regression(C, E, f, max_niterations=100):
    """Update discrete regression with C as a cause variable and Y as a effect variable
    so that it maximize likelihood
    Args
    -------
        C (sequence): sequence of discrete outcomes
        E (sequence): sequence of discrete outcomes
        f (dict): map from C to Y
    """
    supp_C = np.unique(C)
    supp_E = np.unique(E)
    mod_E = len(supp_E)
    n = len(C)

    # N_E's log likelihood
    # optimize f to minimize N_E's log likelihood
    cur_likelihood = 0
    res = np.mod(np.subtract(E, [f[c] for c in C]), mod_E)
    for freq in np.bincount(res):
        cur_likelihood += freq * (np.log2(n or 1) - np.log2(freq or 1))

    j = 0
    minimized = True
    while j < max_niterations and minimized:
        minimized = False

        for c_to_map in supp_C:
            best_likelihood = sys.float_info.max
            best_e = None

            for cand_e in supp_E:
                if cand_e == f[c_to_map]:
                    continue

                f_ = f.copy()
                f_[c_to_map] = cand_e


                neglikelihood = 0
                res = np.mod(np.subtract(E, [f_[c] for c in C]), mod_E)
                for freq in np.bincount(res):
                    neglikelihood += freq * (np.log2(n or 1) - np.log2(freq or 1))

                if neglikelihood < best_likelihood:
                    best_likelihood = neglikelihood
                    best_e = cand_e

            if best_likelihood < cur_likelihood:
                cur_likelihood = best_likelihood
                f[c_to_map] = best_e
                minimized = True
        j += 1

    return f


def cause_effect_negloglikelihood(C, E, func):
    """Compute negative log likelihood for cause & effect pair.
    Model type : C→E
    Args
    -------
        C (sequence): sequence of discrete outcomes (Cause)
        E (sequence): sequence of discrete outcomes (Effect)
        func (dict): map from C-value to E-value
    Returns
    -------
        (float): maximum log likelihood
    """
    supp_C = np.unique(C)
    supp_E = np.unique(E)
    mod_C = len(supp_C)
    mod_E = len(supp_E)

    n = len(C)

    pair_cnt = defaultdict(lambda: defaultdict(int))
    for c, e in zip(C, E):
        pair_cnt[c][e] += 1

    loglikelihood = 0

    for freq in np.bincount(C):
        loglikelihood += freq * (np.log2(n or 1) - np.log2(freq or 1))

    for e_E in supp_E:
        freq = 0
        for e in supp_E:
            for c in supp_C:
                if (func[c] + e_E) % mod_E == e:
                    freq += pair_cnt[c][e]
        loglikelihood += freq * (np.log2(n or 1) - np.log2(freq or 1))

    return loglikelihood


def univ_enc(n):
    """Computes the universal code length of the given integer.
    Reference: J. Rissanen. A Universal Prior for Integers and Estimation by
    Minimum Description Length. Annals of Statistics 11(2) pp.416-431, 1983.
    """
    from math import log
    ucl = log(2.86504, 2)
    previous = n
    while True:
        previous = log(previous, 2)
        if previous < 1.0:
            break
        ucl += previous
    return ucl


def discretize(X, m_X: int) -> np.ndarray:
    """
    Discretize sequence X into m_X bins between min(X) and max(X).

    Args
    ----
    X : array-like of shape (n_samples,)
        The sequence to be discretized.
    m_X : int
        The number of bins.

    Returns
    -------
    disc_X : numpy.ndarray of shape (n_samples,)
        Discretized sequence.

    """
    # add an offset to the max value to include the maximum value in the last bin
    offset = 1e-8

    # compute the minimum and maximum values of X
    min_X = np.min(X)
    max_X = np.max(X)

    # create a sequence of m_X equally spaced values between min_X and max_X
    bins = np.linspace(min_X, max_X + offset, m_X + 1)

    # bin the values in X and convert the resulting labels to integers
    disc_X = np.digitize(X, bins, right=False)
    disc_X = LabelEncoder().fit_transform(disc_X)
    
    return disc_X

