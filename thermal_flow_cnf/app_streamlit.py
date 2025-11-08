from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from thermal_flow_cnf.src.config import CONFIG
from thermal_flow_cnf.src.flows import poiseuille_flow, diffuser_flow, compressor_flow, bend_flow
from thermal_flow_cnf.src.simulation.langevin import simulate_dataset
from thermal_flow_cnf.src.simulation.dataset import TrajectoryDataset
from thermal_flow_cnf.src.evaluation.visualize import plot_trajectories, plot_density_hist2d, animate_trajectories, animation_to_html
from thermal_flow_cnf.src.evaluation.metrics import mean_squared_displacement, kl_divergence_2d, overlap_ratio
from thermal_flow_cnf.src.model.base_cnf import CNF


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
    flow = st.selectbox("Flow", ["poiseuille", "diffuser", "compressor", "bend"], index=0)
    num_particles = st.number_input("Num Particles", value=200, min_value=10, step=10)
    T = st.number_input("Time Steps (T)", value=500, min_value=10, step=10)
    dt = st.number_input("dt", value=CONFIG["dt"], step=0.001, format="%.3f")
    D = st.number_input("D (diffusion)", value=CONFIG["D"], step=0.01, format="%.3f")
    H = st.number_input("H (half-height)", value=CONFIG["H"], step=0.1, format="%.2f")
    # Flow-specific parameters
    Umax = st.number_input("Umax (poiseuille/bend)", value=CONFIG["Umax"], step=0.1)
    L = st.number_input("Length L (diffuser/compressor/bend)", value=1.0, step=0.1, format="%.2f")
    H_in = st.number_input("H_in (diffuser/compressor)", value=CONFIG["H"], step=0.1, format="%.2f")
    H_out = st.number_input("H_out (diffuser/compressor)", value=CONFIG["H"], step=0.1, format="%.2f")
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
        from torch.utils.data import DataLoader
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
        from thermal_flow_cnf.src.model.train import train_cnf
        p_outer = st.progress(0, text="Training epoch 0")
        p_inner = st.progress(0, text="Batch 0")
        def tcb(epoch, epochs, step, total, loss):
            dim = 2
            avg_logp = -loss
            bpd = (loss / dim) / np.log(2) if dim > 0 else float('nan')
            p_outer.progress(min(epoch/epochs, 1.0), text=f"Training epoch {epoch}/{epochs} | NLL {loss:.4f} | avg logp {avg_logp:.4f} | bpd {bpd:.3f}")
            p_inner.progress(min(step/total, 1.0), text=f"Batch {step}/{total}")
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
            from thermal_flow_cnf.src.utils.io import load_checkpoint
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
            steps_anim = int(min(int(max_frames), trajs.shape[1]))
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
                tail=50,
                init_mean=init_mean_loaded,
                init_cov=init_cov_loaded,
                max_frames=int(max_frames),
                frame_stride=(int(frame_stride_in) if int(frame_stride_in) > 0 else None),
            )
            components.html(animation_to_html(anim, embed_limit_mb=float(embed_limit_mb)), height=420)
            st.toast("Animation ready", icon="🎞️")
        except Exception as e:
            log(f"[animate][error] {type(e).__name__}: {e}")
            st.error(f"Animation failed: {e}")

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
                from thermal_flow_cnf.src.utils.io import load_checkpoint
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
