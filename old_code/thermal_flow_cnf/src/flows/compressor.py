import numpy as np
from typing import Callable


def compressor_flow(Umax_in: float, H_in: float, H_out: float, L: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Planar compressor (converging channel) with half-height decreasing from ``H_in`` to ``H_out``.

    This implementation enforces incompressible continuity for the transverse velocity instead of a heuristic.

    Derivation (incompressible, 2D):
      u_x(x,y) = Umax(x) * (1 - (y/H(x))^2),  with constant volumetric flow Q = (4/3) H(x) Umax(x).
      Continuity: du_x/dx + du_y/dy = 0  ⇒ du_y/dy = -du_x/dx.
      With H'(x)=dH/dx and Umax'(x)= -Umax(x)/H(x) * H'(x), algebra gives du_x/dx = -(Umax/H)H' + 3 Umax H' y^2 / H^3.
      Integrating du_y/dy from 0 to y with u_y(x,0)=0 yields:
          u_y(x,y) = Umax(x) * H'(x) * y * (H(x)^2 - y^2) / H(x)^3.
      For a compressor H'(x)<0 so u_y points inward (opposite sign of y).

    Parameters
    ----------
    Umax_in : float  Centerline velocity at x=0.
    H_in : float     Inlet half-height.
    H_out : float    Outlet half-height (< H_in for compression).
    L : float        Length of section.

    Returns
    -------
    Callable: (x,y) -> [u_x, u_y]
    """
    Umax_in = float(Umax_in)
    H_in = float(H_in)
    H_out = float(H_out)
    L = float(L)
    L = 1e-6 if L == 0.0 else L

    Q = (4.0 / 3.0) * H_in * Umax_in  # constant volumetric flow
    dHdx = (H_out - H_in) / L  # negative for compressor

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
    # Attach channel metadata for visualization (dynamic boundaries)
    setattr(u, "Hx", Hx)
    setattr(u, "H_in", H_in)
    setattr(u, "H_out", H_out)
    setattr(u, "L", L)
    setattr(u, "boundary_type", "varying-channel")
    return u
