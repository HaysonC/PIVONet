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
