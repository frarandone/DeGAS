"""Joint distribution over named program variables."""

from __future__ import annotations

from pydegas.mixtures.gaussian_mix import GaussianMix


class Dist:
    """Variable names in order (`var_list`) and joint state as a Gaussian mixture (`gm`)."""

    def __init__(self, var_list: list[str], gm: GaussianMix) -> None:
        self.var_list = var_list
        self.gm = gm

    def __str__(self) -> str:
        return f"Dist<{self.var_list},{self.gm}>"

    def __repr__(self) -> str:
        return str(self)
