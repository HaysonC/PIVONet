"""
Data management utility for saving simulation, training, and metadata with version control.
Maintains a current/ folder and history/ folder with timestamp-based versioning.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


def get_timestamp() -> str:
    """Generate a timestamp string for folder naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_data_root() -> Path:
    """Get the root data directory path."""
    # Assuming this is called from thermal_flow_cnf context
    return Path("thermal_flow_cnf") / "data"


def get_current_dir() -> Path:
    """Get the current data directory."""
    current = get_data_root() / "current"
    current.mkdir(parents=True, exist_ok=True)
    return current


def get_history_dir() -> Path:
    """Get the history directory."""
    history = get_data_root() / "history"
    history.mkdir(parents=True, exist_ok=True)
    return history


def prune_history(max_versions: int = 5) -> None:
    """
    Keep only the most recent max_versions timestamp folders in history.
    Deletes older versions.
    
    Args:
        max_versions: Maximum number of historical versions to keep (default 5)
    """
    history_dir = get_history_dir()
    
    # Get all timestamp directories
    timestamp_dirs = [
        d for d in history_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    if len(timestamp_dirs) <= max_versions:
        return
    
    # Sort by modification time (oldest first)
    timestamp_dirs.sort(key=lambda d: d.stat().st_mtime)
    
    # Remove oldest directories
    for old_dir in timestamp_dirs[:-max_versions]:
        try:
            shutil.rmtree(old_dir)
            print(f"Removed old history version: {old_dir.name}")
        except Exception as e:
            print(f"Warning: Could not remove {old_dir}: {e}")


def backup_current_to_history() -> Optional[Path]:
    """
    Copy the entire current/ directory to history/ with a timestamp.
    Only creates backup if current/ has any data files.
    
    Returns:
        Path to the new history folder, or None if no backup was made
    """
    current_dir = get_current_dir()
    
    # Check if there's anything to backup
    data_files = [
        f for f in current_dir.glob("*.csv")
        if f.is_file()
    ]
    
    if not data_files:
        return None
    
    # Create timestamped backup
    timestamp = get_timestamp()
    history_dir = get_history_dir()
    backup_dir = history_dir / timestamp
    
    try:
        shutil.copytree(current_dir, backup_dir)
        print(f"Created backup: {backup_dir.name}")
        
        # Prune old versions
        prune_history()
        
        return backup_dir
    except Exception as e:
        print(f"Warning: Backup failed: {e}")
        return None


def save_simulation_data(
    trajectories: np.ndarray,
    x0s: np.ndarray,
    params: Dict[str, Any],
    backup: bool = True
) -> Path:
    """
    Save simulation trajectory data to current/sim_data.csv
    
    Args:
        trajectories: Array of shape (N, T, D) - particle trajectories
        x0s: Array of shape (N, D) - initial positions
        params: Dictionary of simulation parameters
        backup: Whether to create a history backup before saving
        
    Returns:
        Path to the saved file
    """
    if backup:
        backup_current_to_history()
    
    current_dir = get_current_dir()
    output_path = current_dir / "sim_data.csv"
    
    # Flatten trajectories into a DataFrame
    N, T, D = trajectories.shape
    
    rows = []
    for particle_idx in range(N):
        for time_idx in range(T):
            row = {
                'particle_id': particle_idx,
                'time_step': time_idx,
                'x': trajectories[particle_idx, time_idx, 0],
                'y': trajectories[particle_idx, time_idx, 1],
                'x0': x0s[particle_idx, 0],
                'y0': x0s[particle_idx, 1],
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add parameters as metadata in the header (as comments)
    with open(output_path, 'w') as f:
        f.write("# Simulation Data\n")
        for key, value in params.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
    
    # Append the dataframe
    df.to_csv(output_path, mode='a', index=False)
    
    print(f"Saved simulation data: {output_path}")
    return output_path


def save_training_data(
    training_log: list[Dict[str, Any]],
    final_metrics: Dict[str, Any],
    backup: bool = True
) -> Path:
    """
    Save training metrics and loss history to current/train_data.csv
    
    Args:
        training_log: List of dictionaries containing epoch, step, loss, etc.
        final_metrics: Final training metrics and model info
        backup: Whether to create a history backup before saving
        
    Returns:
        Path to the saved file
    """
    if backup:
        backup_current_to_history()
    
    current_dir = get_current_dir()
    output_path = current_dir / "train_data.csv"
    
    # Convert training log to DataFrame
    df = pd.DataFrame(training_log)
    
    # Write with metadata header
    with open(output_path, 'w') as f:
        f.write("# Training Data\n")
        for key, value in final_metrics.items():
            f.write(f"# {key}: {value}\n")
        f.write("\n")
    
    # Append the dataframe
    df.to_csv(output_path, mode='a', index=False)
    
    print(f"Saved training data: {output_path}")
    return output_path


def save_animation_metadata(
    n_true_particles: int,
    n_pred_particles: int,
    n_frames: int,
    trajectory_stats: Dict[str, Any],
    model_params: Dict[str, Any],
    backup: bool = True
) -> Path:
    """
    Save animation metadata to current/meta_data.csv
    
    Args:
        n_true_particles: Number of true trajectory particles
        n_pred_particles: Number of predicted trajectory particles
        n_frames: Number of animation frames
        trajectory_stats: Statistics about trajectories (MSD, bounds, etc.)
        model_params: Model parameters used for inference
        backup: Whether to create a history backup before saving
        
    Returns:
        Path to the saved file
    """
    if backup:
        backup_current_to_history()
    
    current_dir = get_current_dir()
    output_path = current_dir / "meta_data.csv"
    
    # Create a simple metadata table
    metadata = {
        'n_true_particles': n_true_particles,
        'n_pred_particles': n_pred_particles,
        'n_frames': n_frames,
    }
    metadata.update(trajectory_stats)
    metadata.update(model_params)
    
    # Convert to single-row DataFrame
    df = pd.DataFrame([metadata])
    
    # Write with timestamp
    with open(output_path, 'w') as f:
        f.write("# Animation Metadata\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write("\n")
    
    df.to_csv(output_path, mode='a', index=False)
    
    print(f"Saved animation metadata: {output_path}")
    return output_path


def list_history_versions() -> list[tuple[str, Path]]:
    """
    List all history versions with their timestamps.
    
    Returns:
        List of (timestamp_string, path) tuples, sorted newest first
    """
    history_dir = get_history_dir()
    
    timestamp_dirs = [
        d for d in history_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    # Sort by modification time (newest first)
    timestamp_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    
    return [(d.name, d) for d in timestamp_dirs]


def load_data_from_version(version: Optional[str] = None) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load data from a specific version or current.
    
    Args:
        version: Timestamp string of version to load, or None for current
        
    Returns:
        Dictionary with 'sim_data', 'train_data', 'meta_data' DataFrames (or None if not found)
    """
    if version is None:
        data_dir = get_current_dir()
    else:
        data_dir = get_history_dir() / version
    
    result = {}
    
    for data_type in ['sim_data', 'train_data', 'meta_data']:
        file_path = data_dir / f"{data_type}.csv"
        if file_path.exists():
            # Skip comment lines when reading
            result[data_type] = pd.read_csv(file_path, comment='#')
        else:
            result[data_type] = None
    
    return result
