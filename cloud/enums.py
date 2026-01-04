# cloud/enums.py
from __future__ import annotations
from enum import Enum


class CausalGraph(Enum):
    X_causes_Y = "X→Y"
    Y_causes_X = "X←Y"
    INDEPENDENT = "X⫫Y"
    CONFOUNDER = "X←C→Y"

    @property
    def key(self) -> str:
        return {
            CausalGraph.X_causes_Y: "to",
            CausalGraph.Y_causes_X: "gets",
            CausalGraph.INDEPENDENT: "indep",
            CausalGraph.CONFOUNDER: "confounder",
        }[self]

    @property
    def ascii(self) -> str:
        return {
            CausalGraph.X_causes_Y: "X->Y",
            CausalGraph.Y_causes_X: "X<-Y",
            CausalGraph.INDEPENDENT: "X⫫Y",
            CausalGraph.CONFOUNDER: "X<-C->Y",
        }[self]

    @staticmethod
    def default_all() -> set["CausalGraph"]:
        return {
            CausalGraph.X_causes_Y,
            CausalGraph.Y_causes_X,
            CausalGraph.INDEPENDENT,
            CausalGraph.CONFOUNDER,
        }
