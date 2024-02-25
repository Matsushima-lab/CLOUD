from typing import List, Dict, Tuple, Optional
import numpy as np
import math
from collections import Counter, defaultdict
import itertools
from sklearn.preprocessing import LabelEncoder

from .utils import *


class BaseModel:
    def __init__(self, X, Y, X_ndistinct_vals: Optional[int] = None, Y_ndistinct_vals: Optional[int] = None):
        """
        Args
        ------
            X (array-like of shape): A sequence of discrete outcomes
            Y (array-like of shape): A sequence of discrete outcomes
            X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
            Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.
        """
        self.X = LabelEncoder().fit_transform(X)
        self.Y = LabelEncoder().fit_transform(Y)
        self.X_ndistinct_vals = X_ndistinct_vals or len(set(self.X)) 
        self.Y_ndistinct_vals = Y_ndistinct_vals or len(set(self.Y))
        self.n = len(self.X)

        assert len(self.X) == len(self.Y)

        self.n_supp_X = len(set(self.X))
        self.n_supp_Y = len(set(self.Y))

    def max_log_likelihood(self, model_type: str) -> float:
        """Compute negative maximum log-likelihood of the observations z^n under a model_type.
        Args
        ------
            model_type (str): one of ["to", "gets", "indep", "confounder"]
            f (dict): map from Y-values to frequently co-occuring X-values
        Returns
        -----------
            (float): (negative) maximum log likelihood
        """
        loglikelihood = 0

        if model_type == "to":
            f = map_to_most_frequent(self.X, self.Y)
            f = update_regression(self.X, self.Y, f)
            loglikelihood = cause_effect_negloglikelihood(self.X, self.Y, f)

        elif model_type == "gets":
            g = map_to_most_frequent(self.Y, self.X)
            g = update_regression(self.Y, self.X, g)
            loglikelihood = cause_effect_negloglikelihood(self.Y, self.X, g)

        elif model_type == "indep":
            for freq in np.bincount(self.X):
                loglikelihood += freq * (np.log2(self.n) - np.log2(freq))
            for freq in np.bincount(self.Y):
                loglikelihood += freq * (np.log2(self.n) - np.log2(freq))

        elif model_type == "confounder":
            pair_cnt = defaultdict(lambda: defaultdict(int))
            for x, y in zip(self.X, self.Y):
                pair_cnt[x][y] += 1

            for x in list(set(self.X)):
                for y in list(set(self.Y)):
                    loglikelihood += pair_cnt[x][y] * (np.log2(self.n) - math.log(pair_cnt[x][y] or 1, 2))

        return loglikelihood

    def parametric_complexity(self, model_type: str) -> float:
        """Computes the Parametric Complexity of Multinomals.
        Args
        ----------
            model_type (str): ["to", "gets", "indep", "confounder"]
        Returns
        ----------
            float: Parametric Complexity of Multinomals
        """
        if model_type == "confounder":
            return log_C_CAT(n=self.n, K=self.X_ndistinct_vals * self.Y_ndistinct_vals)

        else:
            return log_C_CAT(n=self.n, K=self.X_ndistinct_vals) + log_C_CAT(n=self.n, K=self.Y_ndistinct_vals)

    def stochastic_complexity(self, model_type: str) -> float:
        """Computes the code length L(z^n; M) (stochastic complexity of the data z^n; two discrete sequences).
        Args
        ------
            model_type (str): ["to", "gets", "indep", "confounder"]
        Returns
        ----------
            float: Stochastic Complexity of an input data
        """

        data_comp = self.max_log_likelihood(model_type)
        model_comp = self.parametric_complexity(model_type)
        L = data_comp + model_comp

        self.codelength_likelihood_ = data_comp
        self.parametric_complexity_ = model_comp

        # add code length of function
        if model_type == "to":
            L += math.log(self.n_supp_Y**(self.n_supp_X - 1) - 1, 2)
        elif model_type == "gets":
            L += math.log(self.n_supp_X**(self.n_supp_Y - 1) - 1, 2)

        return L

    def predict(self, n_candidates=4)-> List[Tuple[float, str]]:
        """Predicts the best causal model among four candidates based on the code-length L of the data.

        Args
        ------
            n_candidates (int): Number of candidates to consider. Either 2, 3, or 4.

        Returns
        ------
            list: List of tuples containing the L (stochastic complexity) and model type (str) for each candidate.
        """
        if n_candidates == 4:
            MODEL_CANDIDATES = ["to", "gets", "indep", "confounder"]
        elif n_candidates == 2:
            MODEL_CANDIDATES = ["to", "gets"]
        else:
            MODEL_CANDIDATES = ["to", "gets", "confounder"]

        results = []

        for model_type in MODEL_CANDIDATES:
            L = self.stochastic_complexity(model_type)
            results.append((L, model_type))

        return results


class CLOUD:
    def __init__(self, X, Y, n_candidates: int,
                 X_ndistinct_vals: Optional[int] = None, Y_ndistinct_vals: Optional[int] = None,
                 is_X_continuous: bool = False, is_Y_continuous: bool = False, max_exponent: int = 6):
        """
        Args
        ------
            X (array-like of shape): A sequence of outcomes
            Y (array-like of shape): A sequence of outcomes
            n_candidates (int): Number of candidates to consider. Either 2, 3, or 4.
            X_ndistinct_vals (int): number of distinct values of the multinomial r.v X.
            Y_ndistinct_vals (int): number of distinct values of the multinomial r.v Y.
            is_X_continuous (bool): True if X is continuous variable, False otherwise.
            is_Y_continuous (bool): True if Y is continuous variable, False otherwise.
            max_exponent (int):  Power of two representing the maximum value of the search range of bin
        """
        self.X = X
        self.Y = Y
        self.X_ndistinct_vals = X_ndistinct_vals or len(np.unique(self.X))
        self.Y_ndistinct_vals = Y_ndistinct_vals or len(np.unique(self.Y))
        self.is_X_continuous = is_X_continuous
        self.is_Y_continuous = is_Y_continuous
        self.max_exponent = max_exponent

        # If X is continuous and Y is discrete, swap them
        # assert X: discrete, Y: continuous
        self._swap_flag = False
        if self.is_X_continuous and not self.is_Y_continuous:
            self.X, self.Y = self.Y, self.X
            self.X_ndistinct_vals, self.Y_ndistinct_vals = self.Y_ndistinct_vals, self.X_ndistinct_vals
            self.is_X_continuous, self.is_Y_continuous = 0, 1
            self._swap_flag = True

        # Initialize model candidates based on the number of candidates
        if n_candidates == 4:
            self.MODEL_CANDIDATES = ["to", "gets", "indep", "confounder"]
        elif n_candidates == 2:
            self.MODEL_CANDIDATES = ["to", "gets"]
        else:
            self.MODEL_CANDIDATES = ["to", "gets", "confounder"]

        # Check that X and Y have the same length
        self.n = len(self.X)
        assert len(self.X) == len(self.Y)

    def compute_codelength(self) -> Dict[str, float]:
        """
        Compute the code length for the data under each model candidate using the CLOUD algorithm.

        Returns:
            dict: a dictionary of code lengths for each model candidate
        """
        L_res = defaultdict(int)
        best_mXmY = {"to": None, "gets": None, "indep": None, "confounder": None}
        MAX_m_X = 2**self.max_exponent 
        MAX_m_Y = 2**self.max_exponent 
        
        # Case 1: X and Y are continuous variables
        # TODO: Record the previously searched m_X and m_Y to avoid unnecessary duplications.
        if self.is_X_continuous and self.is_Y_continuous:
            for model_type in self.MODEL_CANDIDATES:
                L_best = sys.float_info.max
                # Find the best m_X 
                for is_mX_linear, early_stopping_Xcount in [(0, 4), (1, 5)]:
                    consecutive_Xcount = 0
                    m_X = 2
                    while consecutive_Xcount < early_stopping_Xcount and m_X < MAX_m_X:
                        L_local_best = sys.float_info.max
                        # Find the best m_Y with fixed m_X
                        for is_mY_linear, early_stopping_Ycount in [(0, 4), (1, 5)]:
                            consecutive_Ycount = 0
                            m_Y = 2
                            while consecutive_Ycount < early_stopping_Ycount and m_Y < MAX_m_Y:
                                model = BaseModel(
                                    discretize(self.X, m_X),
                                    discretize(self.Y, m_Y),
                                    X_ndistinct_vals=m_X,
                                    Y_ndistinct_vals=m_Y
                                )

                                # Compute the code length for the current M_{model_type}^{m_X, m_Y}
                                L_cur = model.stochastic_complexity(model_type)
                                L_cur -=  self.n * np.log2(m_X) + self.n * np.log2(m_Y)
                                L_cur += universal_codelength(m_X) + universal_codelength(m_Y)

                                if L_cur < L_local_best:
                                    L_local_best = L_cur
                                    consecutive_Ycount = 0
                                    best_local_m_X, best_local_m_Y = m_X, m_Y
                                else:
                                    consecutive_Ycount += 1
                                if is_mY_linear:
                                    m_Y += 1
                                else:
                                    m_Y <<= 1

                        if L_local_best < L_best:
                            L_best = L_local_best
                            m_X_best, m_Y_best = best_local_m_X, best_local_m_Y
                            consecutive_Xcount = 0
                            L_res[model_type] = L_best
                        else:
                            consecutive_Xcount += 1
                        if is_mX_linear:
                            m_X += 1
                        else:
                            m_X <<= 1
                best_mXmY[model_type] = (m_X_best, m_Y_best)

        # Case 2: X is discrete and Y is continuous
        elif not self.is_X_continuous and self.is_Y_continuous:
            for model_type in self.MODEL_CANDIDATES:
                L_best = sys.float_info.max
                for is_linear, early_stopping_Ycount in [(0, 4), (1, 10)]:
                    m_Y = 2
                    consecutive_Ycount = 0

                    # Find the best m_Y under the R=model_type
                    while consecutive_Ycount < early_stopping_Ycount and m_Y < MAX_m_Y:
                        model = BaseModel(
                            self.X,
                            discretize(self.Y, m_Y),
                            X_ndistinct_vals=self.X_ndistinct_vals,
                            Y_ndistinct_vals=m_Y
                        )

                        # Compute the code length for the current M((m_X, m_Y), R=model_type)
                        L_cur = model.stochastic_complexity(model_type)
                        L_cur -= self.n * np.log2(m_Y)
                        L_cur += universal_codelength(m_Y)

                        if L_cur < L_best:
                            L_best = L_cur
                            consecutive_Ycount = 0
                            m_Y_best = m_Y
                            L_res[model_type] = L_best
                        else:
                            consecutive_Ycount += 1
                        if is_linear:
                            m_Y += 1
                        else:
                            m_Y <<= 1
                best_mXmY[model_type] = m_Y_best

        # Case 3: X & Y are discrete
        else:
            model = BaseModel(
                    self.X,
                    self.Y,
                    X_ndistinct_vals=self.X_ndistinct_vals,
                    Y_ndistinct_vals=self.Y_ndistinct_vals
                    )
            for model_type in self.MODEL_CANDIDATES:
                L_res[model_type] = model.stochastic_complexity(model_type)

        # Swap `to` and `gets` keys in L_res
        if self._swap_flag:
            to_val = L_res.pop("to")
            gets_val = L_res.pop("gets")
            L_res["to"] = gets_val
            L_res["gets"] = to_val

        self.L_res = L_res
        self.best_mXmY = best_mXmY
        return L_res

    def predict(self, report=False) -> str:
        """
        Predict the most likely model candidate based on the computed code lengths.

        Returns:
            str: The name of the most likely model candidate.
        """
        # Compute the code lengths for all model candidates
        Ls = self.compute_codelength()
        # Find the causal model with the shortest code length
        best_model_candidate = min(Ls, key=Ls.get)

        self.L_res = sorted(Ls.items(), key=lambda x: x[1]) 
        self.delta = - self.L_res[0][1] + self.L_res[1][1]
        if report:
            print("CLOUD's report")
            print(f"    shortest L_{self.L_res[0][0]} = {self.L_res[0][1]:.3f}")
            print(f"    2nd shortest L_{self.L_res[1][0]} = {self.L_res[1][1]:.3f}")
            print(f"    Δ = {self.delta / self.n}")
            print(f"    n = {self.n}")
        return best_model_candidate

