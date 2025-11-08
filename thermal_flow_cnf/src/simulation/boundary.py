import numpy as np


def reflect_y(y: float, H: float):
    """Reflect y across boundaries at ±H, returning (y_reflected, bounced: bool).

    For a simple reflection: if y > H, y' = 2H - y; if y < -H, y' = -2H - y.
    """
    bounced = False
    if y > H:
        y = 2 * H - y
        bounced = True
    elif y < -H:
        y = -2 * H - y
        bounced = True
    return y, bounced


def no_slip_damping(y: float, H: float, p: float = 2.0) -> float:
    """Return a scalar damping factor in [0,1] that smoothly enforces no-slip near walls.

    factor = max(0, 1 - (|y|/H)^p). This yields 1 at centerline and 0 at the wall.
    """
    if H <= 0:
        return 1.0
    a = abs(float(y)) / float(H)
    return float(max(0.0, 1.0 - a ** float(p)))
