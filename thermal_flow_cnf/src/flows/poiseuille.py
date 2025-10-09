import numpy as np


def poiseuille_flow(Umax: float, H: float):
    """Parabolic profile u = (Umax * (1 - (y/H)^2), 0)."""
    Umax = float(Umax)
    H = float(H)

    def u(xy: np.ndarray) -> np.ndarray:
        y = float(xy[1])
        return np.array([Umax * (1.0 - (y / H) ** 2), 0.0], dtype=float)

    return u
