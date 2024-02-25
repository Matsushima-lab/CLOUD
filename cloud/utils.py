"""
This file is based on the implementation by K. Budhathoki.
https://github.molgen.mpg.de/EDA/cisc
"""
import sys
import numpy as np
import math
from collections import Counter, defaultdict
import itertools
from sklearn.preprocessing import LabelEncoder


def log_C_CAT(n: int, K: int) -> float:
    """
    Computes the parametric complexity (logarithm) of a categorical model recursively. O(√n + K)
    
    For more details, refer to Eq. (19) (Theorem 1) in:
    "NML Computation Algorithms for Tree-Structured Multinomial Bayesian Networks"
    https://pubmed.ncbi.nlm.nih.gov/18382603/
    
    and Algorithm 2 in:
    "Computing the Multinomial Stochastic Complexity in Sub-Linear Time"
    http://pgm08.cs.aau.dk/Papers/31_Paper.pdf
    
    Args:
        n (int): Sample size of the dataset.
        K (int): Number of categories in the multinomial distribution.
    
    Returns:
        float: Logarithm of the (approximated) parametric complexity
    """
    total_sum = 1
    product_term = 1
    precision_digits = 10  # 10 digit precision

    summation_bound = int(math.ceil(2 + math.sqrt(2 * n * precision_digits * math.log(10))))

    for k in range(1, summation_bound + 1):
        product_term *= (n - k + 1) / n
        total_sum += product_term

    log_previous_sum = np.log2(1.0)
    log_total_sum = np.log2(total_sum or 1)
    log_n = np.log2(n or 1)

    for k in range(3, K + 1):
        log_x = log_n + log_previous_sum - log_total_sum - np.log2(k - 2 or 1)
        x = 2 ** log_x
        log_one_plus_x = np.log2(1 + x or 1)
        log_new_sum = log_total_sum + log_one_plus_x
        
        log_previous_sum = log_total_sum
        log_total_sum = log_new_sum

    if K == 1:
        log_total_sum = np.log2(1.0)

    return log_total_sum


def map_to_most_frequent(X, Y):
    """
    Creates a mapping from each unique value in X to the most frequent corresponding value in Y.

    Args:
        X (array-like of shape): A sequence of discrete values 
        Y (array-like of shape): A sequence of discrete values 

    Returns:
        dict: A dictionary where keys are unique values from X and values are the 
              most frequently co-occurring values from Y.

    Example:
        X = ['a', 'a', 'b', 'b', 'b']
        Y = [1, 2, 2, 2, 3]
        result = map_to_most_frequent(X, Y)
        print(result)  # Output: {'a': 1, 'b': 2}
    """
    # Group Y values by corresponding X values
    x_to_y_values = defaultdict(list)
    for x, y in zip(X, Y):
        x_to_y_values[x].append(y)
    
    # Create a mapping from each unique X value to the most frequent Y value
    most_frequent_map = {x: Counter(y_values).most_common(1)[0][0] for x, y_values in x_to_y_values.items()}
    
    return most_frequent_map


def update_regression(C, E, f, max_niterations=100):
    """Update discrete regression with C as a cause variable and Y as a effect variable
    so that it maximize likelihood
    Args
    -------
        C (array-like of shape): A sequence of discrete values
        E (array-like of shape): A sequence of discrete values
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
        C (array-like of shape): A sequence of discrete outcomes (Cause)
        E (array-like of shape): A sequence of discrete outcomes (Effect)
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


def universal_codelength(n):
    """
    Rissanen, J. (1983). A universal prior for integers and estimation by minimum description length. The Annals of statistics, 11(2), 416-431.

    Args
    ----
        n: int
    """
    universal_code_length = math.log(2.86504, 2)
    current_value = n
    while current_value >= 1.0:
        current_value = np.log2(current_value or 1)
        if current_value < 1.0:
            break
        universal_code_length += current_value
    
    return universal_code_length


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

