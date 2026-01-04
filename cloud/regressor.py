# cloud/regressor.py
from __future__ import annotations
import numpy as np
import math
from .codelengths import nll_from_counts


class NotFittedError(Exception):
    pass


class DiscreteRegressor:
    """
    Greedy optimizer of f: C -> {0..m_E-1} minimizing
    NLL of (E - f(C)) mod m_E.
    """

    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = int(max_iterations)
        self._f: dict[int, int] | None = None
        self._m_E: int | None = None
        self._n_supp_C: int = 0
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def f(self) -> dict[int, int]:
        if not self._is_fitted or self._f is None:
            raise NotFittedError("Regressor is not fitted.")
        return self._f

    def fit(self, C: np.ndarray, E: np.ndarray, E_domain_size: int) -> "DiscreteRegressor":
        C = np.asarray(C, dtype=np.int64)
        E = np.asarray(E, dtype=np.int64)
        self._m_E = int(E_domain_size)
        m_E = self._m_E
        self._n_supp_C = int(np.unique(C).size)

        f: dict[int, int] = {}
        for c in np.unique(C):
            mask = C == c
            vals, cnts = np.unique(E[mask], return_counts=True)
            f[int(c)] = int(vals[np.argmax(cnts)])

        def _nll(fmap: dict[int, int]) -> float:
            preds = np.array([fmap[int(c)] for c in C], dtype=np.int64)
            resid = (E - preds) % m_E
            return nll_from_counts(resid)

        best = _nll(f)
        for _ in range(self.max_iterations):
            converged = False
            for c in np.unique(C):
                c = int(c)
                best_local = best
                best_e = f[c]
                for e in range(self._m_E):
                    if e == f[c]:
                        continue
                    trial = dict(f)
                    trial[c] = e
                    val = _nll(trial)
                    if val < best_local:
                        best_local = val
                        best_e = e
                if best_local < best:
                    f[c] = best_e
                    best = best_local
                    converged = True
            if not converged:
                break

        self._f = f
        self._is_fitted = True
        return self

    def predict(self, C: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._f is None:
            raise NotFittedError("Call fit before predict.")
        C = np.asarray(C, dtype=np.int64)
        return np.array([self._f[int(c)] for c in C], dtype=np.int64)

    @property
    def function_codelength(self) -> float:
        if not self._is_fitted or self._m_E is None:
            raise NotFittedError("Model not fitted.")
        # log2( m_E^(|supp C|-1) - 1 )
        # return float(np.log2(self._m_E ** (self._n_supp_C - 1) - 1.0))
        k = int(self._n_supp_C) - 1
        m = int(self._m_E)
        return float(k * math.log2(m) + math.log2(1.0 - m ** (-k)))
