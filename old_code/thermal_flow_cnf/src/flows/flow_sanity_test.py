"""Quick sanity tests for compressor and diffuser vector fields.
Run directly or import into a notebook. Prints sample vectors at symmetric y locations
and asserts inward/outward transverse velocity directions.
"""
from compressor import compressor_flow
from diffuser import diffuser_flow
import numpy as np


def sample_and_print(flow_fn, name: str, xs, ys):
    print(f"\n{name} samples:")
    for x in xs:
        for y in ys:
            v = flow_fn(np.array([x, y], dtype=float))
            print(f"x={x:.2f}, y={y:.2f} -> ux={v[0]:.4f}, uy={v[1]:.4f}")


def test_compressor_inward():
    comp = compressor_flow(Umax_in=1.0, H_in=1.0, H_out=0.5, L=1.0)
    # Positive y should have negative uy (inward), negative y positive uy
    xs = [0.0, 0.5, 1.0]
    ys = [0.25, 0.5]
    for x in xs:
        for y in ys:
            uy_pos = comp(np.array([x, y]))[1]
            uy_neg = comp(np.array([x, -y]))[1]
            assert uy_pos < 0.0, f"Compressor outward at y>0 (uy={uy_pos})"
            assert uy_neg > 0.0, f"Compressor inward at y<0 (uy={uy_neg})"


def test_diffuser_outward():
    diff = diffuser_flow(Umax_in=1.0, H_in=0.5, H_out=1.0, L=1.0)
    xs = [0.0, 0.5, 1.0]
    ys = [0.25, 0.4]
    for x in xs:
        for y in ys:
            uy_pos = diff(np.array([x, y]))[1]
            uy_neg = diff(np.array([x, -y]))[1]
            assert uy_pos > 0.0, f"Diffuser inward at y>0 (uy={uy_pos})"
            assert uy_neg < 0.0, f"Diffuser outward at y<0 (uy={uy_neg})"


if __name__ == "__main__":
    # Run tests and sample prints
    test_compressor_inward()
    test_diffuser_outward()
    comp_fn = compressor_flow(1.0, 1.0, 0.5, 1.0)
    diff_fn = diffuser_flow(1.0, 0.5, 1.0, 1.0)
    sample_and_print(comp_fn, "Compressor", xs=[0.0, 0.5, 1.0], ys=[0.25, -0.25])
    sample_and_print(diff_fn, "Diffuser", xs=[0.0, 0.5, 1.0], ys=[0.25, -0.25])
    print("\nSanity tests passed.")
