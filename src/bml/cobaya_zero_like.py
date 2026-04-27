from __future__ import annotations
from cobaya.likelihood import Likelihood

class cl_smoke_like(Likelihood):
    """
    Dummy likelihood that forces CAMB to compute Cl.
    Returns logp=0 always.
    """
    lmax: int = 200

    def get_requirements(self):
        # TT-only is enough for pipeline + much faster
        return {"Cl": {"tt": self.lmax}}

    def logp(self, **params_values):
        return 0.0
