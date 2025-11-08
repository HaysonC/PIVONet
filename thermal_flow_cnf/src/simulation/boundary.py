import numpy as np


def reflect_y(y: float, H: float, overshoot_damp: float = 0.0):
    """Reflect y across boundaries at ±H, returning (y_reflected, bounced: bool).

    If overshoot_damp>0, shrink the overshoot distance before reflecting to reduce jitter
    when large time steps cause deep penetration beyond the wall.
    """
    bounced = False
    if y > H:
        if overshoot_damp > 0.0:
            y = H + (y - H) * (1.0 - overshoot_damp)
        y = 2 * H - y
        bounced = True
    elif y < -H:
        if overshoot_damp > 0.0:
            y = -H + (y + H) * (1.0 - overshoot_damp)
        y = -2 * H - y
        bounced = True
    return y, bounced


def reflect_y_variable(x: float, y: float, Hx_func, overshoot_damp: float = 0.0):
    """Reflect using a spatially varying half-height H(x).

    Args:
        x: current x-position
        y: current y-position
        Hx_func: callable H(x) -> half-height at x
        overshoot_damp: optional damping factor in [0,1]
    Returns:
        (y_reflected, bounced)
    """
    # Evaluate local half-height; assume callable is well-behaved and returns numeric
    Hx_val = float(Hx_func(float(x)))
    return reflect_y(y, Hx_val, overshoot_damp=overshoot_damp)


def no_slip_damping(y: float, H: float, p: float = 2.0) -> float:
    """Return a scalar damping factor in [0,1] that smoothly enforces no-slip near walls.

    factor = max(0, 1 - (|y|/H)^p). This yields 1 at centerline and 0 at the wall.
    """
    if H <= 0:
        return 1.0
    a = abs(float(y)) / float(H)
    return float(max(0.0, 1.0 - a ** float(p)))
