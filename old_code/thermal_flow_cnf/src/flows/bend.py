import numpy as np
from typing import Callable


def bend_flow(
    Umax: float, H: float, bend_angle_deg: float = 90.0, L: float = 1.0
) -> Callable[[np.ndarray], np.ndarray]:
    """2D bend flow with parabolic speed and gentle secondary (Dean-like) transverse motion.

    Kinematic model:
    - Base parabolic magnitude: up(y) = Umax * (1 - (y/H)^2)
    - Direction rotates from 0 to bend_angle over length L
    - Add small curvature-induced inward/outward u_y component proportional to dθ/dx and y to mimic Dean vortices.
      uy_extra = k * (dθ/dx) * y * up / H, zero at centerline and walls; sign pushes toward inner wall near entrance.
    - No-slip at y = ±H via parabolic factor.
    """
    Umax = float(Umax)
    H = float(H)
    L = float(L)
    L = 1e-6 if L == 0.0 else L
    theta_f = np.deg2rad(float(bend_angle_deg))

    def theta_of_x(x: float) -> float:
        s = np.clip(x / L, 0.0, 1.0)
        return s * theta_f

    dtheta_dx = theta_f / L
    k = 0.3  # small coefficient for secondary flow strength (dimensionless)

    def u(xy: np.ndarray) -> np.ndarray:
        x = float(xy[0])
        y = float(xy[1])
        up = Umax * (1.0 - (y / H) ** 2)
        th = theta_of_x(x)
        # primary components
        ux = up * np.cos(th)
        uy = up * np.sin(th)
        # secondary transverse motion (in plane): inward/outward relative to curvature
        uy_extra = k * dtheta_dx * y * up / max(H, 1e-8)
        uy = uy + uy_extra
        return np.array([ux, uy], dtype=float)

    return u
