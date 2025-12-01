"""Quick check of checkpoint contents."""
import torch
import sys
from pathlib import Path

ckpt = torch.load(sys.argv[1], map_location="cpu")
print("Keys:", list(ckpt.keys()))
if "args" in ckpt:
    print("\nArgs:")
    for k, v in ckpt["args"].items():
        print(f"  {k}: {v}")
if "model" in ckpt:
    print("\nModel keys (first 5):", list(ckpt["model"].keys())[:5])
