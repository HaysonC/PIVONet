import numpy as np
from typing import Callable


def diffuser_flow(Umax_in: float, H_in: float, H_out: float, L: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Planar diffuser with linearly expanding half-height H(x) from H_in to H_out over length L.

    Assumptions:
    - Steady, incompressible, laminar flow
    - 2D planar Poiseuille at each x with parabolic profile and constant volumetric flow rate Q per unit depth
    - No-slip at y = ±H(x)

    Returns a callable u(xy) -> np.ndarray([ux, uy]) with uy = 0.
    """
    Umax_in = float(Umax_in)
    H_in = float(H_in)
    H_out = float(H_out)
    L = float(L)
    L = 1e-6 if L == 0.0 else L

    # For planar Poiseuille, Q = (4/3) H * Umax
    Q = (4.0 / 3.0) * H_in * Umax_in

    def Hx(x: float) -> float:
        # linear expansion along x in [0, L]; extrapolate outside with clamping
        s = np.clip(x / L, 0.0, 1.0)
        return H_in + s * (H_out - H_in)

    def u(xy: np.ndarray) -> np.ndarray:
        x = float(xy[0])
        y = float(xy[1])
        Hx_val = max(1e-8, Hx(x))
        # Umax(x) from constant Q relation
        Umax_x = Q / ((4.0 / 3.0) * Hx_val)
        ux = Umax_x * (1.0 - (y / Hx_val) ** 2)
        return np.array([ux, 0.0], dtype=float)

    return u
