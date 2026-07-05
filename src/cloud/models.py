# src/cloud/models.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
from .enums import CausalGraph
from .codelengths import nll_from_counts, log_C_CAT, universal_integer_code
from .regressor import DiscreteRegressor


def discretize(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = np.min(x)
    hi = np.max(x)
    hi = np.nextafter(hi, np.inf)
    u = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    idx = np.floor(u * m).astype(np.int64)
    np.minimum(idx, m - 1, out=idx)
    return idx


def encode_categories(x: np.ndarray) -> tuple[np.ndarray, int]:
    x = np.asarray(x)
    _, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int64), int(np.max(inv) + 1)


@dataclass(frozen=True)
class CLOUDResult:
    inferred_graph: CausalGraph
    codelengths: dict[CausalGraph, float]
    delta_bits: float
    n: int


class BaseCLOUD(ABC):
    def __init__(
        self,
        X,
        Y,
        model_candidates: set[CausalGraph] | None = None,
    ):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.n = int(len(self.X))
        assert self.X.shape[0] == self.Y.shape[0]
        self.model_candidates = model_candidates or CausalGraph.default_all()
        self._fit: CLOUDResult | None = None

    @abstractmethod
    def _compute_codelengths(self) -> dict[CausalGraph, float]: ...

    def fit(self) -> "BaseCLOUD":
        Ls = self._compute_codelengths()
        sorted_items = sorted(Ls.items(), key=lambda kv: kv[1])
        inferred_graph, best_L = sorted_items[0]
        second_L = sorted_items[1][1] if len(sorted_items) > 1 else best_L
        self._fit = CLOUDResult(inferred_graph=inferred_graph, codelengths=Ls, delta_bits=float(second_L - best_L), n=self.n)
        return self

    def fit_predict(self) -> CausalGraph:
        return self.fit().predict()

    def predict(self) -> CausalGraph:
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        return self._fit.inferred_graph

    def summary(self) -> str:
        if self._fit is None:
            raise RuntimeError("Call fit() first.")
        lines = []
        lines.append(f"n = {self._fit.n}")
        for g, L in sorted(self._fit.codelengths.items(), key=lambda kv: kv[1]):
            lines.append(f"{g.ascii:8s}  L = {L:.4f} bits")
        lines.append(f"Inferred Causal Graph = {self._fit.inferred_graph.value}  (Δ = {self._fit.delta_bits:.3f} bits)")
        return "\n".join(lines)


class DiscreteCLOUD(BaseCLOUD):
    def __init__(
        self,
        X,
        Y,
        model_candidates: set[CausalGraph] | None = None,
        *,
        already_encoded: bool = False,
        domain_sizes: tuple[int, int] | None = None,
    ):
        super().__init__(X, Y, model_candidates)
        if already_encoded:
            self.Xd = np.asarray(X, dtype=np.int64)
            self.Yd = np.asarray(Y, dtype=np.int64)
            if domain_sizes is not None:
                self.mX, self.mY = map(int, domain_sizes)
            else:
                self.mX = int(self.Xd.max()) + 1 if self.Xd.size else 0
                self.mY = int(self.Yd.max()) + 1 if self.Yd.size else 0
        else:
            # encode to 0..m-1
            self.Xd, self.mX = encode_categories(self.X)
            self.Yd, self.mY = encode_categories(self.Y)

    def _data_nll(self, graph: CausalGraph) -> float:
        if graph is CausalGraph.X_causes_Y:
            nll_X = nll_from_counts(self.Xd)
            reg = DiscreteRegressor().fit(self.Xd, self.Yd, self.mY)
            pred = reg.predict(self.Xd)
            resid = (self.Yd - pred) % self.mY
            nll_resid = nll_from_counts(resid)
            self._f_codelength = reg.function_codelength
            return nll_X + nll_resid

        if graph is CausalGraph.Y_causes_X:
            nll_Y = nll_from_counts(self.Yd)
            reg = DiscreteRegressor().fit(self.Yd, self.Xd, self.mX)
            pred = reg.predict(self.Yd)
            resid = (self.Xd - pred) % self.mX
            nll_resid = nll_from_counts(resid)
            self._g_codelength = reg.function_codelength
            return nll_Y + nll_resid

        if graph is CausalGraph.INDEPENDENT:
            return nll_from_counts(self.Xd) + nll_from_counts(self.Yd)

        if graph is CausalGraph.CONFOUNDER:
            joint = self.Xd * self.mY + self.Yd
            return nll_from_counts(joint)

        raise ValueError("unknown causal graph")

    def _param_complexity(self, graph: CausalGraph) -> float:
        if graph is CausalGraph.CONFOUNDER:
            return log_C_CAT(self.n, self.mX * self.mY)
        return log_C_CAT(self.n, self.mX) + log_C_CAT(self.n, self.mY)

    def _function_codelength(self, graph: CausalGraph) -> float:
        if graph is CausalGraph.X_causes_Y:
            return getattr(self, "_f_codelength", 0.0)
        if graph is CausalGraph.Y_causes_X:
            return getattr(self, "_g_codelength", 0.0)
        return 0.0

    def _compute_codelengths(self) -> dict[CausalGraph, float]:
        Ls: dict[CausalGraph, float] = {}
        for g in self.model_candidates:
            nll = self._data_nll(g)
            pc = self._param_complexity(g)
            fc = self._function_codelength(g)
            Ls[g] = float(nll + pc + fc)
        return Ls


class MixedCLOUD(BaseCLOUD):
    def __init__(
        self,
        X,
        Y,
        *,
        X_is_continuous: bool,
        Y_is_continuous: bool,
        max_exponent: int = 9,
        model_candidates: set[CausalGraph] | None = None,
    ):
        super().__init__(X, Y, model_candidates)
        assert X_is_continuous ^ Y_is_continuous, "MixedCLOUD needs exactly one continuous variable."
        self.X_is_cont = bool(X_is_continuous)
        self.Y_is_cont = bool(Y_is_continuous)
        self.max_exp = int(max_exponent)

        # X : discrete, Y: continuous
        self._swap = False
        if self.X_is_cont and not self.Y_is_cont:
            self.X, self.Y = self.Y, self.X
            self.X_is_cont, self.Y_is_cont = False, True
            self._swap = True

        self.Xd, self.mX = encode_categories(self.X)

    def _compute_codelengths(self) -> dict[CausalGraph, float]:
        Ls: dict[CausalGraph, float] = {}

        MAX_m = min(2 ** (len(bin(self.n).lstrip("0b"))), 2**self.max_exp)
        for g in self.model_candidates:
            best = float("inf")
            for is_linear, patience in [(False, 4), (True, 10)]:
                consec = 0
                m = 2
                while consec < patience and m < MAX_m:
                    Yd = discretize(self.Y, m)
                    disc_cloud = DiscreteCLOUD(
                        self.Xd,
                        Yd,
                        model_candidates={g},
                        already_encoded=True,
                        domain_sizes=(self.mX, m),
                    )
                    Ld = disc_cloud._compute_codelengths()[g]
                    L_cur = Ld - self.n * np.log2(m) + universal_integer_code(m)
                    if L_cur < best:
                        best, consec = L_cur, 0
                    else:
                        consec += 1
                    m = (m + 1) if is_linear else (m << 1)
            Ls[g] = best

        if self._swap:
            if CausalGraph.X_causes_Y in Ls and CausalGraph.Y_causes_X in Ls:
                Ls[CausalGraph.X_causes_Y], Ls[CausalGraph.Y_causes_X] = Ls[CausalGraph.Y_causes_X], Ls[CausalGraph.X_causes_Y]
        return Ls


class ContinuousCLOUD(BaseCLOUD):
    def __init__(self, X, Y, *, max_exponent: int = 10, model_candidates: set[CausalGraph] | None = None):
        super().__init__(X, Y, model_candidates)
        self.max_exp = int(max_exponent)

    def _compute_codelengths(self) -> dict[CausalGraph, float]:
        Ls: dict[CausalGraph, float] = {}

        MAX_m = min(2 ** (len(bin(self.n).lstrip("0b"))), 2**self.max_exp)
        for g in self.model_candidates:
            best = float("inf")
            for is_lin_x, patience_x in [(False, 6), (True, 5)]:
                mX, consec_x = 2, 0
                while consec_x < patience_x and mX < MAX_m:
                    local_best = float("inf")
                    # best_local_mX, best_local_mY = mX, None
                    for is_lin_y, patience_y in [(False, 6), (True, 5)]:
                        mY, consec_y = 2, 0
                        while consec_y < patience_y and mY < MAX_m:
                            Xd = discretize(self.X, mX)
                            Yd = discretize(self.Y, mY)
                            disc_cloud = DiscreteCLOUD(
                                Xd,
                                Yd,
                                model_candidates={g},
                                already_encoded=True,
                                domain_sizes=(mX, mY),
                            )
                            Ld = disc_cloud._compute_codelengths()[g]
                            L_cur = Ld - self.n * (np.log2(mX) + np.log2(mY)) + universal_integer_code(mX) + universal_integer_code(mY)
                            if L_cur < local_best:
                                local_best, consec_y = L_cur, 0
                                # best_local_mX, best_local_mY = mX, mY
                            else:
                                consec_y += 1
                            mY = (mY + 1) if is_lin_y else (mY << 1)
                    if local_best < best:
                        best, consec_x = local_best, 0
                    else:
                        consec_x += 1
                    mX = (mX + 1) if is_lin_x else (mX << 1)
            Ls[g] = best
        return Ls


class CLOUD(BaseCLOUD):
    def __init__(
        self,
        X,
        Y,
        model_candidates: set[CausalGraph] | None = None,
        is_X_continuous: bool = True,
        is_Y_continuous: bool = True,
        max_exponent: int = 9,
    ):
        super().__init__(X, Y, model_candidates)

        self.X_cont = bool(is_X_continuous)
        self.Y_cont = bool(is_Y_continuous)
        self.max_exp = int(max_exponent)

        if not self.X_cont and not self.Y_cont:
            self._delegate = DiscreteCLOUD(self.X, self.Y, model_candidates=self.model_candidates)
        elif self.X_cont and self.Y_cont:
            self._delegate = ContinuousCLOUD(self.X, self.Y, max_exponent=self.max_exp, model_candidates=self.model_candidates)  # type: ignore
        else:
            self._delegate = MixedCLOUD(
                self.X,
                self.Y,
                X_is_continuous=self.X_cont,
                Y_is_continuous=self.Y_cont,
                max_exponent=self.max_exp,
                model_candidates=self.model_candidates,  # type: ignore
            )

    def _compute_codelengths(self) -> dict[CausalGraph, float]:
        return self._delegate._compute_codelengths()
