from __future__ import annotations
from typing import Any, Dict
import numpy as np
from cobaya.theory import Theory

class powerlaw_pk(Theory):
    """
    Provides primordial_scalar_pk as a pure power-law:
      P_R(k) = As * (k/pivot_k)^(ns-1)
    Used to validate CAMB external_primordial_pk + Planck likelihoods.
    """
    params = {"As": None, "ns": None}

    kmin: float = 1.0e-6
    kmax: float = 1.0
    nk: int = 220
    spacing: str = "log"
    pivot_k: float = 0.05

    def initialize(self):
        self._last = None

    def get_requirements(self):
        return {}

    def get_can_provide(self):
        return ["primordial_scalar_pk"]

    def calculate(self, state: Dict[str, Any], want_derived: bool = False, **params_values_dict):
        As = float(params_values_dict["As"])
        ns = float(params_values_dict["ns"])

        if str(self.spacing).lower() == "log":
            k = np.geomspace(self.kmin, self.kmax, self.nk)
            log_regular = True
        else:
            k = np.linspace(self.kmin, self.kmax, self.nk)
            log_regular = False

        Pk = As * (k / float(self.pivot_k)) ** (ns - 1.0)

        self._last = {
            "k": k,
            "Pk": Pk,
            "log_regular": log_regular,
            "kmin": float(k[0]),
            "kmax": float(k[-1]),
        }
        state["primordial_scalar_pk"] = self._last
        return True

    def get_primordial_scalar_pk(self):
        return self._last
