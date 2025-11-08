import numpy as np
from typing import Callable


def bend_flow(Umax: float, H: float, bend_angle_deg: float = 90.0, L: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Simple 2D bend flow: parabolic speed profile with direction rotating along x.

    This is a kinematic model (not full CFD):
    - u_parabolic(y) = Umax * (1 - (y/H)^2)
    - Direction angle theta(x) varies linearly from 0 to bend_angle over length L
    - No-slip at y = ±H enforced by parabolic profile
    """
    Umax = float(Umax)
    H = float(H)
    L = float(L)
    L = 1e-6 if L == 0.0 else L
    theta_f = np.deg2rad(float(bend_angle_deg))

    def theta_of_x(x: float) -> float:
        s = np.clip(x / L, 0.0, 1.0)
        return s * theta_f

    def u(xy: np.ndarray) -> np.ndarray:
        x = float(xy[0])
        y = float(xy[1])
        up = Umax * (1.0 - (y / H) ** 2)
        th = theta_of_x(x)
        ux = up * np.cos(th)
        uy = up * np.sin(th)
        return np.array([ux, uy], dtype=float)

    return u
