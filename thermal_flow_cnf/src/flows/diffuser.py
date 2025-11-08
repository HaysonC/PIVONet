import numpy as np
from typing import Callable


def diffuser_flow(Umax_in: float, H_in: float, H_out: float, L: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Planar diffuser with linearly expanding half-height H(x) from H_in to H_out over length L.

    Use the incompressible continuity-consistent transverse velocity derived from a parabolic profile:
        u_x(x,y) = Umax(x) * (1 - (y/H)^2),  Q = (4/3) H Umax = const.
        u_y(x,y) = Umax(x) * H'(x) * y * (H(x)^2 - y^2) / H(x)^3.
    For a diffuser H'(x) > 0, so u_y has the same sign as y (outward).
    """
    Umax_in = float(Umax_in)
    H_in = float(H_in)
    H_out = float(H_out)
    L = float(L)
    L = 1e-6 if L == 0.0 else L

    Q = (4.0 / 3.0) * H_in * Umax_in
    dHdx = (H_out - H_in) / L  # positive

    def Hx(x: float) -> float:
        s = np.clip(x / L, 0.0, 1.0)
        return H_in + s * (H_out - H_in)

    def u(xy: np.ndarray) -> np.ndarray:
        x = float(xy[0]); y = float(xy[1])
        Hx_val = max(1e-8, Hx(x))
        Umax_x = Q / ((4.0 / 3.0) * Hx_val)
        ux = Umax_x * (1.0 - (y / Hx_val) ** 2)
        uy = Umax_x * dHdx * y * (Hx_val**2 - y**2) / (Hx_val**3)
        return np.array([ux, uy], dtype=float)
    setattr(u, "Hx", Hx)
    setattr(u, "H_in", H_in)
    setattr(u, "H_out", H_out)
    setattr(u, "L", L)
    setattr(u, "boundary_type", "varying-channel")
    return u
