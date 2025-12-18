from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from thermal_flow_cnf.src.config import CONFIG
from thermal_flow_cnf.src.flows import uniform_flow, couette_flow, poiseuille_flow
from thermal_flow_cnf.src.simulation.langevin import simulate_dataset
from thermal_flow_cnf.src.simulation.dataset import TrajectoryDataset
from thermal_flow_cnf.src.model.base_cnf import CNF
from thermal_flow_cnf.src.model.train import train_cnf
from thermal_flow_cnf.src.utils import set_seed


def get_flow(cfg):
    f = cfg.get("flow_type", "poiseuille")
    if f == "uniform":
        return uniform_flow(cfg.get("U0", 1.0))
    if f == "couette":
        return couette_flow(cfg.get("gamma", 0.5))
    if f == "poiseuille":
        return poiseuille_flow(cfg.get("Umax", 2.0), cfg.get("H", 1.0))
    raise ValueError(f"Unknown flow_type: {f}")


def cli_simulate(args):
    cfg = CONFIG.copy()
    cfg["flow_type"] = args.flow
    cfg["num_particles"] = args.num
    set_seed(cfg.get("seed", 42))

    flow_fn = get_flow(cfg)
    out = simulate_dataset(
        flow_fn,
        num_particles=cfg["num_particles"],
        D=cfg["D"],
        dt=cfg["dt"],
        T=cfg["T"],
        H=cfg["H"],
        save_dir=os.path.join("thermal_flow_cnf", "data", "raw"),
        prefix=f"{args.flow}",
        seed=cfg.get("seed", None),
    )
    print(f"Saved dataset: {out}")


def cli_train(args):
    cfg = CONFIG.copy()
    set_seed(cfg.get("seed", 42))

    if args.dataset is None:
        # Try to find a dataset
        data_dir = os.path.join("thermal_flow_cnf", "data", "raw")
        candidates = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ]
        if not candidates:
            raise FileNotFoundError(
                "No dataset found. Run simulate first or pass --dataset."
            )
        dataset_path = sorted(candidates)[-1]
    else:
        dataset_path = args.dataset

    ds = TrajectoryDataset(dataset_path)
    dl = DataLoader(
        ds, batch_size=args.batch or cfg["batch_size"], shuffle=True, num_workers=0
    )

    model = CNF(dim=2, cond_dim=3, hidden_dim=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.join("thermal_flow_cnf", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    train_cnf(
        model,
        dl,
        device=device,
        epochs=args.epochs or cfg["epochs"],
        lr=cfg["lr"],
        ckpt_dir=ckpt_dir,
    )


def cli_evaluate(args):
    from thermal_flow_cnf.src.evaluation.metrics import mean_squared_displacement

    if args.dataset is None:
        data_dir = os.path.join("thermal_flow_cnf", "data", "raw")
        candidates = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ]
        if not candidates:
            raise FileNotFoundError(
                "No dataset found. Run simulate first or pass --dataset."
            )
        dataset_path = sorted(candidates)[-1]
    else:
        dataset_path = args.dataset

    data = np.load(dataset_path)
    trajs = data["trajs"]
    if args.metric == "msd":
        msd = mean_squared_displacement(trajs)
        print(f"MSD: {msd:.6f}")
    else:
        print("Only msd metric implemented in CLI demo.")


def build_parser():
    p = argparse.ArgumentParser(description="thermal_flow_cnf CLI")
    sub = p.add_subparsers(dest="cmd")

    ps = sub.add_parser("simulate")
    ps.add_argument(
        "--flow",
        type=str,
        default="poiseuille",
        choices=["uniform", "couette", "poiseuille"],
    )
    ps.add_argument("--num", type=int, default=1000)
    ps.set_defaults(func=cli_simulate)

    pt = sub.add_parser("train")
    pt.add_argument("--dataset", type=str, default=None)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--batch", type=int, default=None)
    pt.set_defaults(func=cli_train)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--dataset", type=str, default=None)
    pe.add_argument("--metric", type=str, default="msd")
    pe.set_defaults(func=cli_evaluate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
