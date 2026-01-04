# cloud/codelengths.py
from __future__ import annotations
import numpy as np


def nll_from_counts(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if n == 0:
        return 0.0
    cnt = np.bincount(labels)
    mask = cnt > 0
    cnt = cnt[mask]
    return float(np.sum(cnt * (np.log2(n) - np.log2(cnt))))


def universal_integer_code(n: int) -> float:
    """
    Rissanen, Jorma. "A universal prior for integers and estimation by minimum description length." The Annals of statistics 11.2 (1983): 416-431.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    L = np.log2(2.865064)
    x = float(n)
    while x >= 2.0:
        x = np.log2(x)
        L += x
    return float(L)


_param_complexity_cache: dict[tuple[int, int], float] = {}


def log_C_CAT(n: int, K: int) -> float:
    """
    Algorithm 2 in:
    Mononen, Tommi, and Petri Myllymäki. "Computing the multinomial stochastic complexity in sub-linear time." Proceedings of the 4th European Workshop on Probabilistic Graphical Models. 2008.
    """
    if n == 0 or K == 1:
        return np.log2(1.0)

    cache_key = (n, K)
    if cache_key in _param_complexity_cache:
        return _param_complexity_cache[cache_key]

    summ = 1.0
    b_term = 1.0
    precision_digits = 10  # 10 digit precision
    bound = int(np.ceil(2 + np.sqrt(2 * n * precision_digits * np.log(10))))

    for j in range(1, bound + 1):
        b_term *= (n - j + 1) / n
        summ += b_term

    log_prev = np.log2(1.0)
    log_curr = np.log2(summ)
    log_n = np.log2(n)

    for k in range(3, K + 1):
        log_new = np.logaddexp2(log_curr, log_n + log_prev - np.log2(k - 2))
        log_prev, log_curr = log_curr, log_new
    _param_complexity_cache[cache_key] = log_curr
    return log_curr
