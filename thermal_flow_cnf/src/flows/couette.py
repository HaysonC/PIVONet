import numpy as np


def couette_flow(gamma: float):
    """Linear shear flow u = (gamma * y, 0)."""
    gamma = float(gamma)

    def u(xy: np.ndarray) -> np.ndarray:
        y = float(xy[1])
        return np.array([gamma * y, 0.0], dtype=float)

    return u
