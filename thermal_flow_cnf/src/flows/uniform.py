import numpy as np


def uniform_flow(U0: float):
    """Uniform flow with constant velocity in x-direction.

    Returns a callable u(xy) -> np.ndarray shape (2,)
    """
    U0 = float(U0)

    def u(xy: np.ndarray) -> np.ndarray:
        return np.array([U0, 0.0], dtype=float)

    return u
