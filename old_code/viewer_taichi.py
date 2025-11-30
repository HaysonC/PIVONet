"""Taichi-based particle trajectory viewer

Usage:
    python viewer_taichi.py --dataset run_demo [--scale 1.0]

Features:
- Native Taichi window (GPU accelerated)
- Plays time-resolved particle position frames saved as raw float32 triplets in
  ./data/<dataset>/positions_frame_###.bin
- Keyboard controls: Space play/pause, ←/→ step, [/] change size, C toggle color,
  G toggle grid. Right-mouse-drag to orbit, scroll to zoom.

Streamlit snippet to launch viewer (put in your app):

import streamlit as st, subprocess, sys
if st.button("Open Taichi Flow Viewer"):
    subprocess.Popen([sys.executable, "viewer_taichi.py", "--dataset", "run_demo"])
    st.success("Viewer launched.")

Notes:
- Requires `taichi` (pip install taichi). This script targets Taichi's modern UI API
  (ti.ui). Minor API differences between Taichi versions may require small edits.
"""

from pathlib import Path
import argparse
import numpy as np
import taichi as ti
import time

try:
    from matplotlib import cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def find_frame_files(data_dir: Path):
    files = sorted(data_dir.glob('positions_frame_*.bin'))
    return files


def viridis_colormap(t):
    """Map t in [0,1] to RGB using Matplotlib's viridis if available, otherwise fallback.
    Accepts scalar or array-like and returns Nx3 array or single tuple.
    """
    t_arr = np.array(t, copy=False)
    if _HAS_MPL:
        # cm.viridis expects values in [0,1] and returns RGBA
        rgba = cm.viridis(np.clip(t_arr, 0.0, 1.0))
        rgb = rgba[..., :3]
        return rgb
    # fallback: simple gradient approximation
    t = np.clip(t_arr, 0.0, 1.0)
    # ensure shape
    flat = t.ravel()
    out = np.empty((flat.size, 3), dtype=np.float32)
    for i, tt in enumerate(flat):
        if tt < 0.25:
            a = tt / 0.25
            c = (0.0 * (1 - a) + 0.1 * a,
                 0.0 * (1 - a) + 0.3 * a,
                 0.2 * (1 - a) + 0.9 * a)
        elif tt < 0.5:
            a = (tt - 0.25) / 0.25
            c = (0.1 * (1 - a) + 0.0 * a,
                 0.3 * (1 - a) + 0.6 * a,
                 0.9 * (1 - a) + 0.3 * a)
        elif tt < 0.75:
            a = (tt - 0.5) / 0.25
            c = (0.0 * (1 - a) + 0.7 * a,
                 0.6 * (1 - a) + 0.8 * a,
                 0.3 * (1 - a) + 0.1 * a)
        else:
            a = (tt - 0.75) / 0.25
            c = (0.7 * (1 - a) + 1.0 * a,
                 0.8 * (1 - a) + 0.9 * a,
                 0.1 * (1 - a) + 0.0 * a)
        out[i] = c
    if t_arr.shape == ():  # scalar input
        return tuple(out[0].tolist())
    return out.reshape(t_arr.shape + (3,))


def load_frame_numpy(path: Path, scale=1.0):
    arr = np.fromfile(str(path), dtype=np.float32)
    if arr.size % 3 != 0:
        raise ValueError(f"File {path} length not divisible by 3")
    arr = arr.reshape(-1, 3) * np.float32(scale)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='folder under ./data')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--stream-threshold-mb', type=float, default=500.0,
                        help='If dataset total size > threshold, stream frames from disk')
    args = parser.parse_args()

    data_dir = Path('data') / args.dataset
    if not data_dir.exists():
        print(f"Data folder {data_dir} not found")
        return

    frame_files = find_frame_files(data_dir)
    if len(frame_files) == 0:
        print(f"No frames found in {data_dir}")
        return

    # Determine particle count from first frame
    first = frame_files[0]
    first_arr = load_frame_numpy(first, scale=args.scale)
    N = first_arr.shape[0]
    num_frames = len(frame_files)

    total_bytes = num_frames * N * 3 * 4
    total_mb = total_bytes / (1024.0 * 1024.0)
    stream = total_mb > args.stream_threshold_mb
    print(f"Found {num_frames} frames, {N} particles per frame, total {total_mb:.1f} MB -> streaming={stream}")

    # If not streaming, load all frames into numpy array (num_frames, N, 3)
    frames_np = None
    if not stream:
        frames_np = np.empty((num_frames, N, 3), dtype=np.float32)
        frames_np[0] = first_arr
        for i, p in enumerate(frame_files[1:], start=1):
            frames_np[i] = load_frame_numpy(p, scale=args.scale)

    # Taichi initialization
    ti.init(arch=ti.gpu, kernel_profiler=False)

    # Taichi fields
    pos_field = ti.Vector.field(3, dtype=ti.f32, shape=N)
    prev_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=N)

    # Window and scene
    window = ti.ui.Window("Taichi Flow Viewer", (1280, 720), vsync=True)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(2.5, 1.6, 2.5)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 0.0, 1.0)

    # Controls state
    paused = False
    frame_idx = 0
    particle_radius = 0.01
    color_mode = 'constant'  # or 'speed'
    show_grid = True

    # Color arrays
    const_color = (0.4, 0.7, 1.0)

    # Keep running max speed for normalization
    running_max_speed = 1e-6

    # Camera orbit parameters
    azimuth = -45.0  # degrees
    elevation = -30.0
    distance = 4.0
    prev_mouse = None

    last_time = time.time()
    target_fps = 60.0

    # Helper to set camera from spherical params
    def update_camera():
        rad_az = np.radians(azimuth)
        rad_el = np.radians(elevation)
        x = distance * np.cos(rad_el) * np.cos(rad_az)
        y = distance * np.cos(rad_el) * np.sin(rad_az)
        z = distance * np.sin(rad_el)
        camera.position(x, y, z)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 0.0, 1.0)

    update_camera()

    # Main loop
    while window.running:
        for e in window.get_events():
            # keyboard events
            if e.key == 'space' and e.type == ti.ui.PRESS:
                paused = not paused
            if e.key == 'right' and e.type == ti.ui.PRESS:
                frame_idx = min(frame_idx + 1, num_frames - 1)
                paused = True
            if e.key == 'left' and e.type == ti.ui.PRESS:
                frame_idx = max(frame_idx - 1, 0)
                paused = True
            if e.key == ']' and e.type == ti.ui.PRESS:
                particle_radius *= 1.1
            if e.key == '[' and e.type == ti.ui.PRESS:
                particle_radius = max(1e-5, particle_radius / 1.1)
            if e.key in ('c', 'C') and e.type == ti.ui.PRESS:
                color_mode = 'speed' if color_mode == 'constant' else 'constant'
            if e.key in ('g', 'G') and e.type == ti.ui.PRESS:
                show_grid = not show_grid

            # Mouse interaction: right-button drag for orbit
            if e.type == ti.ui.MOTION and window.is_pressed(ti.ui.RMB):
                mx, my = window.get_cursor_pos()
                if prev_mouse is not None:
                    dx = mx - prev_mouse[0]
                    dy = my - prev_mouse[1]
                    azimuth -= dx * 180.0
                    elevation += dy * 180.0
                    elevation = np.clip(elevation, -89.0, 89.0)
                    update_camera()
                prev_mouse = (mx, my)
            if e.type == ti.ui.RELEASE:
                prev_mouse = None

            # Scroll to zoom
            if e.type == ti.ui.SCROLL:
                # e.delta is (dx, dy)
                ddy = e.delta[1] if hasattr(e, 'delta') else 0.0
                distance *= (1.0 - ddy * 0.1)
                distance = max(0.1, distance)
                update_camera()

        # Playback update
        now = time.time()
        dt = now - last_time
        if (not paused) and dt >= 1.0 / target_fps:
            frame_idx = (frame_idx + 1) % num_frames
            last_time = now

        # Load frame (from memory or disk)
        if frames_np is not None:
            cur_np = frames_np[frame_idx]
        else:
            cur_np = load_frame_numpy(frame_files[frame_idx], scale=args.scale)

        # Update taichi fields
        pos_field.from_numpy(cur_np.astype(np.float32))

        # Compute speeds if needed
        speeds = None
        if color_mode == 'speed':
            # get prev positions
            prev_np = prev_pos_field.to_numpy()
            if prev_np.shape[0] != N:
                prev_np = cur_np.copy()
            disp = cur_np - prev_np
            speed = np.linalg.norm(disp, axis=1)  # per-frame displacement
            # update running max for normalization
            running_max_speed = max(running_max_speed, speed.max() if speed.size>0 else 0.0)
            vmax = running_max_speed if running_max_speed > 1e-8 else 1e-8
            speed_norm = speed / vmax
            # build color array
            colors = np.array([viridis_colormap(t) for t in speed_norm], dtype=np.float32)
            speeds = colors

        # copy current to prev
        prev_pos_field.from_numpy(cur_np.astype(np.float32))

        # set up scene
        scene.set_camera(camera)
        scene.ambient_light((0.08, 0.08, 0.08))
        scene.point_light(pos=(2.5, 5.0, 2.5), color=(0.9, 0.9, 0.9))
        scene.point_light(pos=(-2.5, -3.5, 1.5), color=(0.4, 0.4, 0.5))

        # background: white
        canvas.set_background_color((1.0, 1.0, 1.0))

    # optional grid helper (simple XY grid at z=0)
        if show_grid:
            # draw faint grid lines using scene.lines if available
            try:
                grid_lines = []
                rng = np.linspace(-1.5, 1.5, 31)
                for x in rng:
                    grid_lines.append([[x, -1.5, 0.0], [x, 1.5, 0.0]])
                for y in rng:
                    grid_lines.append([[-1.5, y, 0.0], [1.5, y, 0.0]])
                # flatten
                pts = np.array(grid_lines, dtype=np.float32).reshape(-1, 3)
                # scene.lines expects a field or numpy in some taichi versions; try canvas.lines
                canvas.lines(pts, width=1.0, color=(0.18, 0.18, 0.2))
            except Exception:
                # ignore if API differs
                pass

        # --- compute previous positions and rotation to align mean flow left->right ---
        # get raw previous positions
        try:
            raw_prev = prev_pos_field.to_numpy()
        except Exception:
            raw_prev = cur_np.copy()

        if raw_prev.shape[0] != N:
            raw_prev = cur_np.copy()

        # displacement (raw)
        disp_raw = cur_np - raw_prev
        mean_disp = disp_raw[:, :2].mean(axis=0)
        mean_norm = np.linalg.norm(mean_disp)
        # compute yaw to rotate mean_disp to +x axis
        if mean_norm > 1e-8:
            ang = np.arctan2(mean_disp[1], mean_disp[0])
            rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]], dtype=np.float32)
        else:
            rot = np.eye(2, dtype=np.float32)

        def rotate_xy(arr2d, R):
            # arr2d: (M,3) -> rotate x,y components
            out = arr2d.copy()
            xy = out[:, :2].dot(R.T)
            out[:, :2] = xy
            return out

        rot_cur = rotate_xy(cur_np, rot)
        rot_prev = rotate_xy(raw_prev, rot)

        # update taichi fields with rotated positions so everything aligns left->right
        pos_field.from_numpy(rot_cur.astype(np.float32))
        prev_pos_field.from_numpy(rot_prev.astype(np.float32))

        # compute bounds for boundary box
        if frames_np is not None:
            # global bounds from loaded frames
            all_xy = frames_np.reshape(-1, 3)[:, :2]
            min_xy = all_xy.min(axis=0)
            max_xy = all_xy.max(axis=0)
        else:
            min_xy = np.minimum(rot_cur[:, :2].min(axis=0), rot_prev[:, :2].min(axis=0))
            max_xy = np.maximum(rot_cur[:, :2].max(axis=0), rot_prev[:, :2].max(axis=0))
        pad = 0.02 * max(max_xy - min_xy)
        min_xy -= pad
        max_xy += pad

        # draw black rectangular boundary
        try:
            bx0, by0 = min_xy[0], min_xy[1]
            bx1, by1 = max_xy[0], max_xy[1]
            rect_lines = np.array([
                [bx0, by0, 0.0], [bx1, by0, 0.0],
                [bx1, by0, 0.0], [bx1, by1, 0.0],
                [bx1, by1, 0.0], [bx0, by1, 0.0],
                [bx0, by1, 0.0], [bx0, by0, 0.0],
            ], dtype=np.float32)
            canvas.lines(rect_lines, width=2.0, color=(0.0, 0.0, 0.0))
        except Exception:
            pass

        # draw flowfield (quiver) sampled from a subset of particles
        try:
            max_arrows = 500
            idx = np.linspace(0, N - 1, min(N, max_arrows)).astype(int)
            starts = rot_cur[idx, :3]
            vecs = rot_cur[idx, :2] - rot_prev[idx, :2]
            # scale vectors for visibility
            vec_mag = np.linalg.norm(vecs, axis=1)
            vmax = vec_mag.max() if vec_mag.size > 0 else 1.0
            scale = 0.2 * max(max_xy - min_xy).max() / (vmax + 1e-9)
            ends = starts.copy()
            ends[:, 0] = starts[:, 0] + vecs[:, 0] * scale
            ends[:, 1] = starts[:, 1] + vecs[:, 1] * scale
            # build line segments array
            arrow_lines = np.empty((len(starts) * 2, 3), dtype=np.float32)
            arrow_lines[0::2] = starts
            arrow_lines[1::2] = ends
            canvas.lines(arrow_lines, width=1.0, color=(0.0, 0.0, 0.0))
        except Exception:
            pass

        # draw particles (already fed rotated positions into pos_field)
        if speeds is None:
            scene.particles(pos_field, radius=particle_radius, color=(0.2, 0.4, 0.8))
        else:
            try:
                scene.particles(pos_field, radius=particle_radius, color=speeds)
            except Exception:
                avgcol = tuple(speeds.mean(axis=0).tolist())
                scene.particles(pos_field, radius=particle_radius, color=avgcol)

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
