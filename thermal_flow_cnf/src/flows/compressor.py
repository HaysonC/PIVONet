import numpy as np
from typing import Callable


def compressor_flow(Umax_in: float, H_in: float, H_out: float, L: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Planar compressor (converging channel) with half-height decreasing from H_in to H_out.

    Similar assumptions as diffuser_flow; conserves volumetric flow rate Q.
    """
    Umax_in = float(Umax_in)
    H_in = float(H_in)
    H_out = float(H_out)
    L = float(L)
    L = 1e-6 if L == 0.0 else L

    Q = (4.0 / 3.0) * H_in * Umax_in

    def Hx(x: float) -> float:
        s = np.clip(x / L, 0.0, 1.0)
        return H_in + s * (H_out - H_in)

    def u(xy: np.ndarray) -> np.ndarray:
        x = float(xy[0])
        y = float(xy[1])
        Hx_val = max(1e-8, Hx(x))
        Umax_x = Q / ((4.0 / 3.0) * Hx_val)
        ux = Umax_x * (1.0 - (y / Hx_val) ** 2)
        return np.array([ux, 0.0], dtype=float)

    return u
