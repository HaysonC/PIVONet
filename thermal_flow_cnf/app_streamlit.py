from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from torch.utils.data import DataLoader
from thermal_flow_cnf.src.config import CONFIG
from thermal_flow_cnf.src.flows import poiseuille_flow, diffuser_flow, compressor_flow, bend_flow
from thermal_flow_cnf.src.simulation.langevin import simulate_dataset
from thermal_flow_cnf.src.simulation.dataset import TrajectoryDataset, TrajectorySequenceDataset
from thermal_flow_cnf.src.evaluation.visualize import plot_trajectories, plot_density_hist2d, animate_trajectories, animation_to_html
from thermal_flow_cnf.src.evaluation.metrics import mean_squared_displacement, kl_divergence_2d, overlap_ratio
from thermal_flow_cnf.src.model.base_cnf import CNF
from thermal_flow_cnf.src.model.variational_sde import VariationalSDEModel
from thermal_flow_cnf.src.model.train import train_cnf, train_variational_sde
from thermal_flow_cnf.src.utils.io import load_checkpoint
from thermal_flow_cnf.src.utils.data_manager import (
    save_simulation_data, save_training_data, save_animation_metadata
)


st.set_page_config(page_title="Thermal Flow CNF", layout="wide")

# --- Simple in-app logging utilities ---
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log(msg: str) -> None:
    # Ensure logs container exists and is a list
    if "logs" not in st.session_state or not isinstance(st.session_state["logs"], list):
        st.session_state["logs"] = []
    st.session_state["logs"].append(msg)

def device_options() -> List[str]:
    opts = ["cpu"]
    try:
        if torch.cuda.is_available():
            opts.insert(0, "cuda")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # prefer CUDA over MPS if both exist
            if "cuda" in opts:
                opts.append("mps")
            else:
                opts.insert(0, "mps")
    except Exception:
        pass
    return opts

st.title("Thermal Flow CNF: Simulate, Train, Visualize")

# Sidebar controls
with st.sidebar:
    st.header("Simulation Settings")
    flow = st.selectbox("Flow", ["poiseuille", "diffuser", "compressor", "bend"], index=0, key="flow_select")
    # Track flow changes to set sensible defaults for H_in/H_out
    prev_flow = st.session_state.get("_prev_flow")
    num_particles = st.number_input("Num Particles", value=200, min_value=10, step=10)
    T = st.number_input("Time Steps (T)", value=500, min_value=10, step=10)
    dt = st.number_input("dt", value=CONFIG["dt"], step=0.001, format="%.3f")
    D = st.number_input("D (diffusion)", value=CONFIG["D"], step=0.01, format="%.3f")
    H = st.number_input("H (half-height)", value=CONFIG["H"], step=0.1, format="%.2f")
    # Flow-specific parameters
    Umax = st.number_input("Umax (poiseuille/bend)", value=CONFIG["Umax"], step=0.1)
    L = st.number_input("Length L (diffuser/compressor/bend)", value=1.0, step=0.1, format="%.2f")
    # Persist H_in/H_out in session state to allow auto-adjustment
    if "H_in" not in st.session_state:
        st.session_state["H_in"] = float(CONFIG["H"])
    if "H_out" not in st.session_state:
        st.session_state["H_out"] = float(CONFIG["H"]) 
    # When flow changes, set a sensible default relationship between H_in and H_out
    if prev_flow != flow:
        if flow == "diffuser":
            # Make outlet wider than inlet by default
            st.session_state["H_out"] = max(st.session_state.get("H_out", H) , st.session_state.get("H_in", H) * 1.5)
        elif flow == "compressor":
            # Make outlet narrower than inlet by default
            st.session_state["H_out"] = max(0.05, min(st.session_state.get("H_out", H), st.session_state.get("H_in", H) * 0.7))
        st.session_state["_prev_flow"] = flow
    H_in = st.number_input("H_in (diffuser/compressor)", value=float(st.session_state["H_in"]), step=0.1, format="%.2f", key="H_in")
    H_out = st.number_input("H_out (diffuser/compressor)", value=float(st.session_state["H_out"]), step=0.1, format="%.2f", key="H_out")
    # Auto-validate and adjust inconsistent H_in/H_out with user feedback
    adjusted = False
    if flow == "diffuser":
        if H_out <= H_in:
            new_H_out = max(H_in * 1.2, H_in + 0.1)
            st.session_state["H_out"] = float(new_H_out)
            H_out = float(new_H_out)
            st.warning("For a diffuser, H_out should be greater than H_in. Adjusted H_out automatically.")
            adjusted = True
    elif flow == "compressor":
        if H_out >= H_in:
            new_H_out = max(0.05, min(H_in * 0.8, H_in - 0.1))
            st.session_state["H_out"] = float(new_H_out)
            H_out = float(new_H_out)
            st.warning("For a compressor, H_out should be less than H_in. Adjusted H_out automatically.")
            adjusted = True
    Umax_in = st.number_input("Umax_in (diffuser/compressor)", value=CONFIG["Umax"], step=0.1)
    bend_angle = st.number_input("Bend angle (deg)", value=90.0, step=5.0, format="%.1f")
    mlp_depth = st.number_input("MLP Depth", value=3, min_value=1, max_value=12, step=1)

    st.header("Model Settings")
    hidden_dim = st.number_input("Hidden Dim", value=64, min_value=16, step=16)
    dropout_p = st.slider("Dropout p (stochastic)", min_value=0.0, max_value=0.5, value=0.0, step=0.05, help="Introduces stochasticity for training/inference when model in train mode")
    epochs = st.number_input("Epochs", value=5, min_value=1, step=1)
    batch = st.number_input("Batch Size", value=128, min_value=8, step=8)
    dev_opts = device_options()
    default_idx = 0
    if "cuda" in dev_opts:
        default_idx = dev_opts.index("cuda")
    elif "mps" in dev_opts:
        default_idx = dev_opts.index("mps")
    selected_device = st.selectbox("Device", dev_opts, index=default_idx, help="Select hardware device for model training/inference")

    st.header("MSD Options")
    msd_mode = st.selectbox("MSD Mode", ["total", "transverse", "demeaned"], index=0)
    st.header("Plot Options")
    overlay_mode = st.selectbox("Flow overlay", ["quiver", "stream"], index=0, help="Show vector field as quiver or streamlines")

def list_datasets() -> List[str]:
    data_dir = os.path.join("thermal_flow_cnf", "data", "raw")
    if not os.path.isdir(data_dir):
        return []
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    return sorted(files)

def list_checkpoints() -> List[str]:
    ckpt_dir = os.path.join("thermal_flow_cnf", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    return sorted(files, key=lambda p: os.path.getmtime(p))

def list_vsde_checkpoints() -> List[str]:
    ckpt_dir = os.path.join("thermal_flow_cnf", "checkpoints", "variational")
    if not os.path.isdir(ckpt_dir):
        return []
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    return sorted(files, key=lambda p: os.path.getmtime(p))

def build_flow(name: str):
    if name == "poiseuille":
        return poiseuille_flow(Umax, H)
    if name == "diffuser":
        return diffuser_flow(Umax_in, H_in, H_out, L=L)
    if name == "compressor":
        return compressor_flow(Umax_in, H_in, H_out, L=L)
    if name == "bend":
        return bend_flow(Umax, H, bend_angle_deg=bend_angle, L=L)
    # default
    return poiseuille_flow(Umax, H)

tabs = st.tabs(["Data", "Train", "Inference & Animate", "Metrics", "Logs"])

with tabs[0]:
    st.subheader("Data")
    st.markdown("Configure initial distribution of particles")
    colA, colB, colC = st.columns(3)
    with colA:
        init_dist = st.selectbox("Initial distribution", ["uniform", "gaussian"], index=1)
    with colB:
        mu_x = st.number_input("mu_x", value=0.0, step=0.1)
        mu_y = st.number_input("mu_y", value=0.0, step=0.1)
    with colC:
        cov_xx = st.number_input("var_x", value=0.1, step=0.05, min_value=0.0, format="%.3f")
        cov_yy = st.number_input("var_y", value=0.1, step=0.05, min_value=0.0, format="%.3f")
        cov_xy = st.number_input("cov_xy", value=0.0, step=0.05, format="%.3f")
    init_mean = np.array([mu_x, mu_y], dtype=np.float32)
    init_cov = np.array([[cov_xx, cov_xy],[cov_xy, cov_yy]], dtype=np.float32)
    if st.button("Simulate", key="simulate_btn"):
        flow_fn = build_flow(flow)
        prog = st.progress(0, text="Simulating...")
        def cb(done: int, total: int):
            prog.progress(min(done / total, 1.0), text=f"Simulating... {done}/{total}")
            if done == total:
                st.toast("Simulation complete", icon="✅")
        log(f"[simulate] flow={flow}, N={int(num_particles)}, T={int(T)}, dt={float(dt):.3g}, D={float(D):.3g}, H={float(H):.3g}")
        path = simulate_dataset(
            flow_fn,
            num_particles=int(num_particles),
            D=float(D),
            dt=float(dt),
            T=int(T),
            H=float(H),
            init_dist=init_dist,
            init_mean=(float(init_mean[0]), float(init_mean[1])),
            init_cov=init_cov.astype(float),
            save_dir=os.path.join("thermal_flow_cnf", "data", "raw"),
            prefix=flow,
            progress_cb=cb,
        )
        st.success(f"Saved dataset: {path}")
        log(f"[simulate] saved: {path}")
        
        # Save simulation data to current/sim_data.csv with history backup
        try:
            data = np.load(path)
            trajs = data["trajs"]
            x0s = data["x0s"]
            sim_params = {
                "flow": flow,
                "num_particles": int(num_particles),
                "T": int(T),
                "dt": float(dt),
                "D": float(D),
                "H": float(H),
                "Umax": float(Umax),
                "init_dist": init_dist,
            }
            save_simulation_data(trajs, x0s, sim_params, backup=True)
            st.toast("Saved to data/current/sim_data.csv", icon="💾")
            log("[simulate] saved sim_data.csv to current/")
        except Exception as e:
            st.warning(f"Could not save CSV: {e}")
            log(f"[simulate] CSV save error: {e}")
    datasets = list_datasets()
    current_dataset = st.session_state.get("dataset_path")
    default_idx = 0
    if current_dataset in datasets:
        default_idx = datasets.index(current_dataset)
    dataset_path: Optional[str] = st.selectbox("Select Dataset", datasets if datasets else [""], index=default_idx, key="dataset_select") if datasets else None
    if dataset_path:
        st.session_state["dataset_path"] = dataset_path
        data = np.load(dataset_path)
        trajs = data["trajs"]
        init_mean_loaded = data.get("init_mean")
        init_cov_loaded = data.get("init_cov")
        st.write(f"Trajectories: {trajs.shape}")
        st.write(f"MSD ({msd_mode}): {mean_squared_displacement(trajs, mode=msd_mode):.4f}")
        log(f"[dataset] loaded: {os.path.basename(dataset_path)} | trajs={trajs.shape}")
        # Recompute flow_fn using current sidebar parameters to reflect latest selection
        fig = plot_trajectories(
            trajs,
            flow_fn=build_flow(flow),
            H=H,
            n_show=50,
            init_mean=init_mean_loaded,
            init_cov=init_cov_loaded,
            flow_overlay_mode=overlay_mode,
        )
        st.pyplot(fig, clear_figure=True)

with tabs[1]:
    st.subheader("Train CNF")
    dataset_path = st.session_state.get("dataset_path")
    # Stochastic training settings (visible before starting training)
    data_noise_std = st.number_input("Target jitter std (stochastic)", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    steps_jitter = st.number_input("ODE steps jitter (+/-)", min_value=0, max_value=8, value=0, step=1, help="Randomize ODE steps per batch to regularize")
    # Physics-informed options
    st.markdown("### Physics-informed losses")
    enable_no_slip = st.checkbox("Enable no-slip loss (u=0 at walls)", value=True)
    no_slip_coef = st.number_input("No-slip loss coeff", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    enable_bern = st.checkbox("Enable Bernoulli/Poiseuille shape loss", value=False, help="Encourage consistent kinetic energy / parabolic profile")
    bern_coef = st.number_input("Bernoulli loss coeff", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    if dataset_path and st.button("Train CNF", key="train_btn"):
        ds = TrajectoryDataset(dataset_path)
        dl = DataLoader(ds, batch_size=int(batch), shuffle=True)
        device = selected_device
        log(f"[train] device={device}, hidden_dim={int(hidden_dim)}, dropout={float(dropout_p):.2f}, epochs={int(epochs)}, batch={int(batch)}, noise_std={float(data_noise_std):.3f}, steps_jitter=+/-{int(steps_jitter)}")
        model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(device)
        # Attach physics configuration for training loop
        # Attach physics configuration (store in plain attribute; torch warns if not registered, so use setattr on object __dict__)
        object.__setattr__(model, "phys_cfg", {
            "no_slip_coef": float(no_slip_coef) if enable_no_slip else 0.0,
            "bernoulli_coef": float(bern_coef) if enable_bern else 0.0,
            "H": float(H),
            "mlp_depth": int(mlp_depth),
        })
        p_outer = st.progress(0, text="Training epoch 0")
        p_inner = st.progress(0, text="Batch 0")
        
        # Track training logs for CSV export
        training_log = []
        
        def tcb(epoch, epochs, step, total, loss):
            dim = 2
            avg_logp = -loss
            bpd = (loss / dim) / np.log(2) if dim > 0 else float('nan')
            p_outer.progress(min(epoch/epochs, 1.0), text=f"Training epoch {epoch}/{epochs} | NLL {loss:.4f} | avg logp {avg_logp:.4f} | bpd {bpd:.3f}")
            p_inner.progress(min(step/total, 1.0), text=f"Batch {step}/{total}")
            
            # Record training data
            training_log.append({
                'epoch': epoch,
                'step': step,
                'total_steps': total,
                'loss': loss,
                'avg_logp': avg_logp,
                'bpd': bpd
            })
            
            # Light logging every 10% of the epoch steps
            if total > 0 and (step == total or step % max(1, total // 10) == 0):
                log(f"[train] epoch {epoch}/{epochs} step {step}/{total} NLL={loss:.4f} avg_logp={avg_logp:.4f} bpd={bpd:.3f}")
            if step == total:
                st.toast(f"Epoch {epoch}/{epochs} completed", icon="📈")
        ckpt_dir = os.path.join("thermal_flow_cnf", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_cnf(
            model,
            dl,
            device=device,
            epochs=int(epochs),
            lr=CONFIG["lr"],
            ckpt_dir=ckpt_dir,
            progress_cb=tcb,
            data_noise_std=float(data_noise_std),
            steps_jitter=int(steps_jitter),
        )
        st.success("Training finished.")
        log("[train] finished")
        
        # Save training data to current/train_data.csv with history backup
        try:
            final_metrics = {
                "total_epochs": int(epochs),
                "batch_size": int(batch),
                "learning_rate": CONFIG["lr"],
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout_p),
                "device": device,
                "data_noise_std": float(data_noise_std),
                "steps_jitter": int(steps_jitter),
            }
            save_training_data(training_log, final_metrics, backup=True)
            st.toast("Saved to data/current/train_data.csv", icon="💾")
            log("[train] saved train_data.csv to current/")
        except Exception as e:
            st.warning(f"Could not save training CSV: {e}")
            log(f"[train] CSV save error: {e}")
        
        # Update last checkpoint selection
        ckpts = list_checkpoints()
        if ckpts:
            st.session_state["model_ckpt"] = ckpts[-1]
            log(f"[train] latest checkpoint: {ckpts[-1]}")

    # Checkpoint selector
    ckpts = list_checkpoints()
    current_ckpt = st.session_state.get("model_ckpt")
    if ckpts:
        default_ckpt_idx = ckpts.index(current_ckpt) if current_ckpt in ckpts else len(ckpts) - 1
        sel = st.selectbox("Select Checkpoint (optional)", ["<none>"] + ckpts, index=(default_ckpt_idx + 1))
        st.session_state["model_ckpt"] = None if sel == "<none>" else sel
        st.caption(f"Selected: {sel if sel!='<none>' else 'no checkpoint (use fresh model)'}")
    else:
        st.info("No checkpoints found. You can train a model or run without loading.")

    ckpt_path = st.session_state.get("model_ckpt")

    st.markdown("### Variational SDE (stochastic posterior)")
    vsde_cols = st.columns(3)
    with vsde_cols[0]:
        vsde_epochs = st.number_input("SDE epochs", min_value=1, max_value=200, value=5, step=1)
    with vsde_cols[1]:
        vsde_lr = st.number_input("SDE learning rate", value=5e-4, format="%.1e")
    with vsde_cols[2]:
        vsde_batch = st.number_input("SDE batch size", min_value=4, max_value=512, value=int(batch), step=4)
    vsde_cols2 = st.columns(3)
    with vsde_cols2[0]:
        vsde_particles = st.number_input("Particles", min_value=1, max_value=16, value=4, step=1)
    with vsde_cols2[1]:
        vsde_obs_std_start = st.number_input("Obs std start", min_value=0.001, max_value=0.5, value=0.15, step=0.01, format="%.3f")
    with vsde_cols2[2]:
        vsde_steps = st.number_input("SDE steps", min_value=10, max_value=400, value=120, step=10)
    vsde_cols3 = st.columns(3)
    with vsde_cols3[0]:
        vsde_kl = st.number_input("KL warmup", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    with vsde_cols3[1]:
        vsde_ctrl = st.number_input("Control cost scale", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with vsde_cols3[2]:
        default_ctx_dim = int(st.session_state.get("vsde_ctx_dim", 128))
        vsde_ctx_dim = st.number_input("Posterior ctx dim", min_value=32, max_value=512, value=default_ctx_dim, step=32)
    vsde_cols4 = st.columns(3)
    with vsde_cols4[0]:
        vsde_obs_std_final = st.number_input("Obs std final", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    with vsde_cols4[1]:
        vsde_diff_learn = st.checkbox("Learn diffusion", value=True)
    with vsde_cols4[2]:
        vsde_initial_log_diff = st.number_input("Initial log diff", value=-1.0, step=0.1, format="%.2f")

    if dataset_path and ckpt_path and st.button("Train Variational SDE", key="train_vsde_btn"):
        ds_seq = TrajectorySequenceDataset(dataset_path)
        dl_seq = DataLoader(ds_seq, batch_size=int(vsde_batch), shuffle=True)
        device = selected_device
        log(
            f"[vsde-train] device={device}, ctx_dim={int(vsde_ctx_dim)}, particles={int(vsde_particles)}, obs_std_sched=({float(vsde_obs_std_start):.3f}->{float(vsde_obs_std_final):.3f}), steps={int(vsde_steps)}, lr={float(vsde_lr):.1e}, epochs={int(vsde_epochs)}"
        )
        base_model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(device)
        if ckpt_path and os.path.isfile(ckpt_path):
            load_checkpoint(base_model, optimizer=None, path=ckpt_path)
            log(f"[vsde-train] loaded CNF checkpoint: {os.path.basename(ckpt_path)}")
        vsde_model = VariationalSDEModel(
            base_model,
            z_dim=2,
            ctx_dim=int(vsde_ctx_dim),
            encoder_cfg={"x_dim": 2},
            diffusion_learnable=vsde_diff_learn,
            initial_log_diff=float(vsde_initial_log_diff),
        ).to(device)
        vsde_outer = st.progress(0, text="Variational epoch 0")
        vsde_inner = st.progress(0, text="Batch 0")

        def vsde_cb(epoch, epochs, step, total, loss, stats=None):
            stats = stats or {}
            elbo = stats.get("elbo") if isinstance(stats, dict) else None
            ctrl = stats.get("control_cost") if isinstance(stats, dict) else None
            obs_std_cur = stats.get("obs_std") if isinstance(stats, dict) else None
            vsde_outer.progress(
                min(epoch / epochs, 1.0),
                text=f"Var-SDE epoch {epoch}/{epochs} | loss {loss:.4f} | elbo {elbo if elbo is not None else float('nan'):.4f} | obsσ {obs_std_cur if obs_std_cur is not None else float('nan'):.3f}",
            )
            vsde_inner.progress(min(step / total, 1.0), text=f"Batch {step}/{total} | control {ctrl if ctrl is not None else float('nan'):.4f}")
            if total > 0 and (step == total or step % max(1, total // 8) == 0):
                log(f"[vsde-train] epoch {epoch}/{epochs} step {step}/{total} loss={loss:.4f} stats={stats}")
            if step == total:
                st.toast(f"Variational epoch {epoch}/{epochs} completed", icon="🌀")

        vsde_ckpt_dir = os.path.join("thermal_flow_cnf", "checkpoints", "variational")
        os.makedirs(vsde_ckpt_dir, exist_ok=True)
        train_variational_sde(
            vsde_model,
            dl_seq,
            device=device,
            epochs=int(vsde_epochs),
            lr=float(vsde_lr),
            ckpt_dir=vsde_ckpt_dir,
            progress_cb=vsde_cb,
            n_particles=int(vsde_particles),
            obs_std=float(vsde_obs_std_final),
            obs_std_start=float(vsde_obs_std_start),
            obs_std_final=float(vsde_obs_std_final),
            kl_warmup=float(vsde_kl),
            control_cost_scale=float(vsde_ctrl),
            n_integration_steps=int(vsde_steps),
        )
        st.success("Variational SDE training finished.")
        st.session_state["vsde_ctx_dim"] = int(vsde_ctx_dim)
        st.session_state["vsde_diff_learn"] = bool(vsde_diff_learn)
        st.session_state["vsde_initial_log_diff"] = float(vsde_initial_log_diff)
        st.session_state["vsde_obs_std_start"] = float(vsde_obs_std_start)
        st.session_state["vsde_obs_std_final"] = float(vsde_obs_std_final)
        vsde_ckpts = list_vsde_checkpoints()
        if vsde_ckpts:
            st.session_state["vsde_ckpt"] = vsde_ckpts[-1]
            log(f"[vsde-train] latest checkpoint: {vsde_ckpts[-1]}")
    elif dataset_path and not ckpt_path:
        st.info("Select a CNF checkpoint above to condition the variational SDE trainer.")

with tabs[2]:
    st.subheader("Inference & Animate")
    dataset_path = st.session_state.get("dataset_path")
    ckpt_path = st.session_state.get("model_ckpt")
    cA, cB, cC = st.columns(3)
    with cA:
        max_frames = st.number_input("Max frames", min_value=50, max_value=1000, value=300, step=50, help="Cap frames to keep embed small")
    with cB:
        frame_stride_in = st.number_input("Frame stride (0=auto)", min_value=0, max_value=50, value=0, step=1)
    with cC:
        embed_limit_mb = st.number_input("Embed limit (MB)", min_value=5.0, max_value=200.0, value=20.0, step=1.0)
        # Finer inference controls
        infer_steps = st.number_input("Integration steps (fine)", min_value=10, max_value=500, value=150, step=10, help="ODE integration steps for model rollout (higher = smoother trajectory)")
        infer_solver = st.selectbox("Solver", ["dopri5", "rk4"], index=0)
        infer_rtol = st.number_input("rtol", min_value=1e-7, max_value=1e-2, value=1e-5, step=1e-6, format="%.1e")
        infer_atol = st.number_input("atol", min_value=1e-7, max_value=1e-2, value=1e-5, step=1e-6, format="%.1e")

    vsde_ckpts = list_vsde_checkpoints()
    current_vsde = st.session_state.get("vsde_ckpt")
    if vsde_ckpts:
        default_vsde = vsde_ckpts.index(current_vsde) if current_vsde in vsde_ckpts else len(vsde_ckpts) - 1
        sel_vsde = st.selectbox("Select Variational SDE checkpoint", ["<none>"] + vsde_ckpts, index=default_vsde + 1)
        st.session_state["vsde_ckpt"] = None if sel_vsde == "<none>" else sel_vsde
    else:
        st.caption("No variational SDE checkpoints yet — train one in the Train tab to enable stochastic inference.")

    vsde_ckpt_path = st.session_state.get("vsde_ckpt")
    vsde_eval_cols = st.columns(3)
    with vsde_eval_cols[0]:
        vsde_eval_particles = st.number_input("Posterior particles (eval)", min_value=1, max_value=32, value=4, step=1)
    with vsde_eval_cols[1]:
        vsde_eval_steps = st.number_input("SDE steps (eval)", min_value=10, max_value=500, value=150, step=10)
    with vsde_eval_cols[2]:
        vsde_eval_obs = st.number_input("Obs std (eval)", min_value=0.001, max_value=0.5, value=0.05, step=0.01, format="%.3f")
    vsde_eval_batch = st.number_input("Eval batch size", min_value=8, max_value=128, value=32, step=8)
    vsde_eval_anchor_noise = st.number_input("Anchor noise", min_value=0.0, max_value=0.5, value=0.02, step=0.01, format="%.2f")
    if dataset_path and st.button("Animate", key="animate_btn"):
        data = np.load(dataset_path)
        trajs = data["trajs"]
        init_mean_loaded = data.get("init_mean")
        init_cov_loaded = data.get("init_cov")
        # Build a simple context: [x0, theta] where theta=0
        x0s = data["x0s"].astype(np.float32)
        thetas = np.zeros((x0s.shape[0], 1), dtype=np.float32)
        context = torch.from_numpy(np.concatenate([x0s, thetas], axis=1)).to(torch.float32)
        device = selected_device
        context = context.to(device)
        model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(device)
        # Load checkpoint if available
        if ckpt_path and os.path.isfile(ckpt_path):
            try:
                load_checkpoint(model, optimizer=None, path=ckpt_path)
                log(f"[animate] loaded checkpoint: {os.path.basename(ckpt_path)}")
            except Exception as e:
                st.warning(f"Checkpoint incompatible or unreadable, running without loading: {e}")
                log(f"[animate][ckpt-warning] {e}")
        else:
            st.info("No checkpoint selected; using a fresh model for inference.")
            log("[animate] using randomly initialized model (no checkpoint selected)")
        model.eval()

        # For demo, use random-initialized model to generate a rough pred trajectory set
        n_samp = int(min(100, int(context.shape[0])))
        ctx_batch = context[:n_samp]
        log(f"[animate] device={device}, n_true={trajs.shape[0]}, T_true={trajs.shape[1]}, n_pred_req={n_samp}")
        with torch.no_grad():
            # Match predicted trajectory length to animation frame budget (avoid disappearance)
            steps_anim = int(min(int(max_frames), infer_steps))
            # Use rollout starting from the same initial positions as true trajectories
            x0_batch = torch.from_numpy(x0s[:n_samp]).to(device=device, dtype=torch.float32)
            z_pred = model.rollout_from(x0_batch, ctx_batch, steps=steps_anim, H=float(H), enforce_bounds=True)
        trajs_pred = z_pred.detach().cpu().numpy()
        log(f"[animate] pred trajs shape={trajs_pred.shape}")

        # Create and render the animation as HTML
        try:
            anim = animate_trajectories(
                trajs[:trajs_pred.shape[0]],
                trajs_pred,
                flow_fn=build_flow(flow),
                H=H,
                n_show=50,
                interval=25,
                tail=60,
                init_mean=init_mean_loaded,
                init_cov=init_cov_loaded,
                max_frames=int(max_frames),
                frame_stride=(int(frame_stride_in) if int(frame_stride_in) > 0 else None),
            )
            components.html(animation_to_html(anim, embed_limit_mb=float(embed_limit_mb)), height=420)
            st.toast("Animation ready", icon="🎞️")
            
            # Save animation metadata to current/meta_data.csv with history backup
            try:
                # Calculate trajectory statistics
                msd_true = mean_squared_displacement(trajs[:n_samp], mode=msd_mode)
                msd_pred = mean_squared_displacement(trajs_pred, mode=msd_mode)
                
                trajectory_stats = {
                    'msd_true': float(msd_true),
                    'msd_pred': float(msd_pred),
                    'true_x_min': float(trajs[:n_samp, :, 0].min()),
                    'true_x_max': float(trajs[:n_samp, :, 0].max()),
                    'true_y_min': float(trajs[:n_samp, :, 1].min()),
                    'true_y_max': float(trajs[:n_samp, :, 1].max()),
                    'pred_x_min': float(trajs_pred[:, :, 0].min()),
                    'pred_x_max': float(trajs_pred[:, :, 0].max()),
                    'pred_y_min': float(trajs_pred[:, :, 1].min()),
                    'pred_y_max': float(trajs_pred[:, :, 1].max()),
                }
                
                model_params = {
                    'hidden_dim': int(hidden_dim),
                    'mlp_depth': int(mlp_depth),
                    'dropout': float(dropout_p),
                    'infer_steps': int(steps_anim),
                    'solver': infer_solver,
                    'rtol': float(infer_rtol),
                    'atol': float(infer_atol),
                    'checkpoint': os.path.basename(ckpt_path) if ckpt_path else 'none',
                    'flow': flow,
                    'H': float(H),
                }
                
                save_animation_metadata(
                    n_true_particles=trajs.shape[0],
                    n_pred_particles=trajs_pred.shape[0],
                    n_frames=int(max_frames),
                    trajectory_stats=trajectory_stats,
                    model_params=model_params,
                    backup=True
                )
                st.toast("Saved to data/current/meta_data.csv", icon="💾")
                log("[animate] saved meta_data.csv to current/")
            except Exception as e:
                st.warning(f"Could not save metadata CSV: {e}")
                log(f"[animate] CSV save error: {e}")
        except Exception as e:
            log(f"[animate][error] {type(e).__name__}: {e}")
            st.error(f"Animation failed: {e}")

    if dataset_path and vsde_ckpt_path and ckpt_path and st.button("Sample Variational Posterior", key="sample_vsde_btn"):
        device = selected_device
        data = np.load(dataset_path)
        init_mean_loaded = data.get("init_mean")
        init_cov_loaded = data.get("init_cov")
        ds_seq = TrajectorySequenceDataset(dataset_path)
        dl_seq = DataLoader(ds_seq, batch_size=int(vsde_eval_batch), shuffle=False)
        x_seq_batch, t_seq_batch, ctx_batch, mask_batch = next(iter(dl_seq))
        x_seq_batch = x_seq_batch.to(device=device, dtype=torch.float32)
        t_seq_batch = t_seq_batch.to(device=device, dtype=torch.float32)
        ctx_batch = ctx_batch.to(device=device, dtype=torch.float32)
        mask_batch = mask_batch.to(device=device, dtype=torch.float32)

        base_model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(device)
        load_checkpoint(base_model, optimizer=None, path=ckpt_path)
        vsde_ctx_dim_eval = int(st.session_state.get("vsde_ctx_dim", 128))
        diff_learn_eval = bool(st.session_state.get("vsde_diff_learn", True))
        init_log_diff_eval = float(st.session_state.get("vsde_initial_log_diff", -1.0))
        vsde_model = VariationalSDEModel(
            base_model,
            z_dim=2,
            ctx_dim=vsde_ctx_dim_eval,
            encoder_cfg={"x_dim": 2},
            diffusion_learnable=diff_learn_eval,
            initial_log_diff=init_log_diff_eval,
        ).to(device)
        load_checkpoint(vsde_model, optimizer=None, path=vsde_ckpt_path)
        vsde_model.eval()

        with torch.no_grad():
            loss_eval, stats_eval = vsde_model.compute_elbo(
                x_seq_batch,
                t_seq_batch,
                context=ctx_batch,
                mask=mask_batch,
                n_particles=int(vsde_eval_particles),
                obs_std=float(vsde_eval_obs),
                n_integration_steps=int(vsde_eval_steps),
            )
            _, traj_samples, _ = vsde_model.sample_posterior(
                x_seq_batch,
                t_seq_batch,
                context=ctx_batch,
                mask=mask_batch,
                n_particles=int(vsde_eval_particles),
                n_integration_steps=int(vsde_eval_steps),
                anchor_noise=float(vsde_eval_anchor_noise),
            )

        traj_true = x_seq_batch.cpu().numpy()
        traj_samples_np = traj_samples.permute(1, 2, 0, 3).contiguous().cpu().numpy()  # (P, B, T, D)
        n_demo = int(min(10, traj_samples_np.shape[1]))
        p_demo = int(min(traj_samples_np.shape[0], 3))
        trajs_pred = traj_samples_np[:p_demo, :n_demo]  # (p_demo, n_demo, T, D)
        trajs_pred = trajs_pred.reshape(p_demo * n_demo, trajs_pred.shape[2], trajs_pred.shape[3])
        trajs_true = np.repeat(traj_true[:n_demo], p_demo, axis=0)
        anim_vsde = animate_trajectories(
            trajs_true=trajs_true,
            trajs_pred=trajs_pred,
            flow_fn=build_flow(flow),
            H=H,
            n_show=trajs_pred.shape[0],
            interval=30,
            tail=60,
            init_mean=init_mean_loaded,
            init_cov=init_cov_loaded,
            max_frames=int(max_frames),
            frame_stride=None,
        )
        components.html(animation_to_html(anim_vsde, embed_limit_mb=float(embed_limit_mb)), height=420)
        elbo_est = -float(loss_eval.item())
        st.write(
            f"Variational ELBO ≈ {elbo_est:.4f} | log_px={stats_eval['log_px_mean']:.4f} | control={stats_eval['control_cost']:.4f} | KL={stats_eval['kl_z0']:.4f} | traj σ={stats_eval['z_traj_std']:.4f}"
        )
        log(
            f"[vsde-eval] elbo={elbo_est:.4f} log_px={stats_eval['log_px_mean']:.4f} control={stats_eval['control_cost']:.4f} kl={stats_eval['kl_z0']:.4f}"
        )
    elif dataset_path and vsde_ckpt_path and not ckpt_path:
        st.warning("Select a CNF checkpoint to pair with the variational posterior sampler.")

        st.markdown("### Particle Position Showcase")
        st.caption("Animate individual particles starting near center, mid-channel, and near boundary for qualitative flow inspection.")
        showcase_btn = st.button("Showcase Center/Mid/Boundary", key="showcase_btn")
        if dataset_path and showcase_btn:
            data = np.load(dataset_path)
            trajs = data["trajs"]
            x0s = data["x0s"].astype(np.float32)
            thetas = np.zeros((x0s.shape[0], 1), dtype=np.float32)
            context = torch.from_numpy(np.concatenate([x0s, thetas], axis=1)).to(torch.float32)
            device = selected_device
            context = context.to(device)
            model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(device)
            if ckpt_path and os.path.isfile(ckpt_path):
                try:
                    load_checkpoint(model, optimizer=None, path=ckpt_path)
                    log(f"[showcase] loaded checkpoint: {os.path.basename(ckpt_path)}")
                except Exception as e:
                    st.warning(f"Checkpoint incompatible or unreadable, running without loading: {e}")
                    log(f"[showcase][ckpt-warning] {e}")
            model.eval()
            with torch.no_grad():
                # Pick representative starting points
                center = torch.tensor([[0.0, 0.0]], device=device, dtype=torch.float32)
                mid = torch.tensor([[0.2, 0.5 * float(H) * 0.5]], device=device, dtype=torch.float32)  # quarter height
                boundary = torch.tensor([[0.4, 0.9 * float(H)]], device=device, dtype=torch.float32)
                reps = torch.cat([center, mid, boundary], dim=0)
                ctx_rep = torch.zeros(reps.size(0), 3, device=device, dtype=torch.float32)
                trajs_rep = model.rollout_from(reps, ctx_rep, steps=int(min(infer_steps, 300)), H=float(H), enforce_bounds=True).cpu().numpy()
            # Build a bespoke pretty animation using existing helper
            anim_rep = animate_trajectories(
                trajs_true=trajs_rep, trajs_pred=None, flow_fn=build_flow(flow), H=H, n_show=3,
                interval=35, tail=80, max_frames=int(max_frames), frame_stride=None,
            )
            components.html(animation_to_html(anim_rep, embed_limit_mb=float(embed_limit_mb)), height=420)
            st.toast("Showcase animation ready", icon="🎯")

with tabs[3]:
    st.subheader("Metrics & Analysis")
    dataset_path = st.session_state.get("dataset_path")
    ckpt_path = st.session_state.get("model_ckpt")
    if dataset_path:
        do_compute = st.button("Compute Metrics", key="metrics_btn")
        if do_compute:
            data = np.load(dataset_path)
            trajs = data["trajs"]
            x0s = data["x0s"].astype(np.float32)
            thetas = np.zeros((x0s.shape[0], 1), dtype=np.float32)
            context = torch.from_numpy(np.concatenate([x0s, thetas], axis=1)).to(torch.float32).to(selected_device)
            model = CNF(dim=2, cond_dim=3, hidden_dim=int(hidden_dim), depth=int(mlp_depth), dropout_p=float(dropout_p)).to(selected_device)
            if ckpt_path and os.path.isfile(ckpt_path):
                load_checkpoint(model, optimizer=None, path=ckpt_path)
                log(f"[metrics] loaded checkpoint: {os.path.basename(ckpt_path)}")
            model.eval()
            with torch.no_grad():
                n_eval = int(min(500, context.shape[0]))
                pred = model.sample(n=n_eval, context=context[:n_eval], steps=int(min(64, trajs.shape[1])), H=float(H), enforce_bounds=True)
            samples_true = trajs[:n_eval, -1, :]
            samples_pred = pred.detach().cpu().numpy()
            fig = plot_density_hist2d(samples_true, samples_pred, bins=60)
            st.pyplot(fig)
            kl = kl_divergence_2d(samples_true, samples_pred, bins=60)
            ov = overlap_ratio(samples_true, samples_pred, eps=0.1)
            st.write(f"KL divergence (2D bins): {kl:.4f}")
            st.write(f"Overlap ratio (eps=0.1): {ov:.4f}")
            log(f"[metrics] KL={kl:.4f}, overlap={ov:.4f}")
    else:
        st.info("Select a dataset on the Data tab.")

with tabs[4]:
    st.subheader("Logs")
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Clear logs", key="clear_logs_btn"):
            st.session_state["logs"] = []
    st.code("\n".join(st.session_state["logs"][-400:]) or "<empty>", language="bash")
