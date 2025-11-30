import subprocess
import os
import glob
import pyvista as pv
import numpy as np

# ----------------------------
# Paths
# ----------------------------
dir_name = "2d-double-mach-reflection"
file_names = "double-mach-reflection"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(BASE_DIR, "test-flows", dir_name)
MESH_FILE = os.path.join(SIM_DIR, f"{file_names}.msh")
PYFRM_FILE = os.path.join(SIM_DIR, f"{file_names}.pyfrm")
CFG_FILE = os.path.join(SIM_DIR, f"{file_names}.ini")
print(f"Simulation directory: {SIM_DIR}")
print(f"Mesh file: {MESH_FILE}")
print(f"Config file: {CFG_FILE}")
VTU_DIR = os.path.join(SIM_DIR, "tmp")
os.makedirs(VTU_DIR, exist_ok=True)
print(f"VTU output directory: {VTU_DIR}")

# ----------------------------
# Functions
# ----------------------------
def run_pyfr_simulation(backend="metal"):
    """Run the PyFR simulation."""
    print("Importing mesh into PyFR format...")
    subprocess.run([
        "pyfr", "import",
        MESH_FILE,
        PYFRM_FILE
    ], cwd=SIM_DIR, check=True)

    print("Starting PyFR simulation...")
   
    subprocess.run([
        "pyfr", "run",
        "-b", backend,
        PYFRM_FILE,
        CFG_FILE
    ], cwd=SIM_DIR, check=True)
    


def export_vtu_files():
    """Export .pyfrs solution files to VTU."""
    pyfrs_files = sorted(glob.glob(os.path.join(SIM_DIR, "*.pyfrs")))
    exported_files = []

    for step, sol_file in enumerate(pyfrs_files):
        try:
            out_file = os.path.join(VTU_DIR, f"double-mach-{step:02d}.vtu")
            subprocess.run([
                "pyfr", "export",
                PYFRM_FILE,  # mesh file
                sol_file,    # solution file
                out_file     # output VTU file
            ], cwd=SIM_DIR, check=True)
            print(f"Exported {sol_file} -> {out_file}")
            exported_files.append(out_file)
        except subprocess.CalledProcessError as e:
            print(f"Error exporting {sol_file}: {e}")
            raise

    return exported_files


def vtu_velocity_to_video(vtu_files, video_path="velocity.mp4", vector_field="Velocity", fps=10, scale=1.0, size=(1024, 768)):
    """
    Convert VTU files into a video showing velocity vectors (arrows).
    Works with 2D or 3D vector fields.
    """
    plotter = pv.Plotter(off_screen=True, window_size=size)
    frames = []

    for f in vtu_files:
        mesh = pv.read(f)
        if vector_field not in mesh.array_names:
            raise KeyError(f"Vector field '{vector_field}' not found in {f}")
        
        # Get velocity array
        vel = mesh[vector_field] # type: ignore
        
        # Ensure 3D vectors
        if vel.shape[1] == 2:  # 2D → add z=0
            vel = np.hstack([vel, np.zeros((vel.shape[0], 1))])
        
        # Ensure centers are 3D
        pts = mesh.points
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        
        if vel.shape[0] != pts.shape[0]:
            raise ValueError(f"Number of vectors ({vel.shape[0]}) does not match number of points ({pts.shape[0]})")
        
        plotter.clear()
        plotter.add_arrows(pts, vel, mag=scale)
        plotter.show_grid() # type: ignore
        img = plotter.screenshot(transparent_background=False, return_img=True)
        frames.append(img)
        print(f"Captured frame from {f}")
    
    print(f"Writing video to {video_path}...")
    import imageio
    imageio.mimsave(video_path, frames, fps=fps)
    plotter.close()
    print("Velocity vector video complete!")


# ----------------------------
# Main workflow
# ----------------------------

def main():
    # 1. Run the PyFR simulation
    run_pyfr_simulation(backend="metal") 

    # 2. Export VTU files
    vtu_files = export_vtu_files()

    # 3. Live-visualize with PyVista
    vtu_velocity_to_video(vtu_files, video_path="velocity_field.mp4", vector_field="Velocity", fps=10, scale=0.1)


    print("Simulation, export, and visualization complete!")


if __name__ == "__main__":
    main()

