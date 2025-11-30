# Data Export and History Management

## Overview

The Streamlit app now automatically saves data from three key operations to CSV files with versioned history management.

## Directory Structure

```
thermal_flow_cnf/data/
├── current/              # Current/latest data
│   ├── sim_data.csv     # Latest simulation data
│   ├── train_data.csv   # Latest training data
│   └── meta_data.csv    # Latest animation metadata
└── history/             # Historical versions (max 5)
    ├── 20251109_143022/ # Timestamped snapshot
    │   ├── sim_data.csv
    │   ├── train_data.csv
    │   └── meta_data.csv
    ├── 20251109_151530/
    │   └── ...
    └── ...
```

## Features

### 1. Automatic Data Export

Each of the three main operations now saves data:

#### **Simulate Button** → `sim_data.csv`
- Saves particle trajectories with initial positions
- Includes simulation parameters as header comments
- Columns: `particle_id`, `time_step`, `x`, `y`, `x0`, `y0`

#### **Train CNF Button** → `train_data.csv`
- Saves training metrics for each epoch/step
- Includes final model configuration as header comments
- Columns: `epoch`, `step`, `total_steps`, `loss`, `avg_logp`, `bpd`

#### **Animate Button** → `meta_data.csv`
- Saves animation metadata and trajectory statistics
- Includes model parameters as header comments
- Columns: particle counts, frame count, MSD, trajectory bounds, model config

### 2. Version Control

- **Automatic Backups**: Before saving new data, the entire `current/` folder is backed up to `history/` with a timestamp
- **History Limit**: Automatically keeps only the 5 most recent versions
- **Timestamp Format**: `YYYYMMDD_HHMMSS` (e.g., `20251109_143022`)
- **Auto-Pruning**: Old versions are automatically deleted when limit exceeded

### 3. Data Recovery

You can manually access historical data:

```python
from thermal_flow_cnf.src.utils.data_manager import load_data_from_version, list_history_versions

# List available versions
versions = list_history_versions()
for timestamp, path in versions:
    print(f"Version: {timestamp} at {path}")

# Load current data
current = load_data_from_version(None)

# Load specific historical version
historical = load_data_from_version("20251109_143022")

# Access DataFrames
sim_df = current['sim_data']
train_df = current['train_data']
meta_df = current['meta_data']
```

## CSV File Formats

### sim_data.csv
```csv
# Simulation Data
# flow: poiseuille
# num_particles: 200
# T: 500
# dt: 0.01
# D: 0.1
# H: 1.0

particle_id,time_step,x,y,x0,y0
0,0,0.1,0.0,0.1,0.0
0,1,0.12,0.01,0.1,0.0
...
```

### train_data.csv
```csv
# Training Data
# total_epochs: 10
# batch_size: 128
# learning_rate: 0.001
# hidden_dim: 64

epoch,step,total_steps,loss,avg_logp,bpd
1,1,100,2.45,-2.45,1.73
1,2,100,2.43,-2.43,1.72
...
```

### meta_data.csv
```csv
# Animation Metadata
# Generated: 2024-11-09T14:30:22

n_true_particles,n_pred_particles,n_frames,msd_true,msd_pred,hidden_dim,checkpoint,flow
200,100,300,0.52,0.48,64,cnf_epoch10.pt,poiseuille
```

## Usage in Streamlit App

The data export happens automatically when you click:

1. **Simulate** → Creates/updates `current/sim_data.csv`
2. **Train CNF** → Creates/updates `current/train_data.csv`
3. **Animate** → Creates/updates `current/meta_data.csv`

Each operation:
- ✅ Backs up existing data to timestamped history folder
- ✅ Saves new data to current folder
- ✅ Prunes history to keep only 5 most recent versions
- ✅ Shows toast notification confirming save

## Implementation Details

### Core Module: `src/utils/data_manager.py`

Key functions:
- `save_simulation_data()` - Export simulation trajectories
- `save_training_data()` - Export training logs
- `save_animation_metadata()` - Export animation info
- `backup_current_to_history()` - Create timestamped backup
- `prune_history()` - Keep only N most recent versions
- `list_history_versions()` - List all saved versions
- `load_data_from_version()` - Load data from specific version

### Integration Points

Modified `app_streamlit.py`:
- Added import: `from thermal_flow_cnf.src.utils.data_manager import ...`
- Updated button handlers for Simulate, Train CNF, and Animate
- Added error handling and toast notifications

## Testing

Run tests to verify functionality:

```bash
# Simple directory and file creation test
python3 test_data_simple.py

# History pruning test
python3 test_pruning.py

# Full data manager test (requires dependencies)
python3 test_data_manager.py
```

## Benefits

1. **Data Preservation**: Never lose experimental results
2. **Easy Comparison**: Compare different runs side-by-side
3. **Reproducibility**: CSV files include all parameters
4. **Version Control**: Automatic timestamped backups
5. **Space Management**: Auto-pruning prevents disk bloat
6. **Standard Format**: CSV files easily imported into Excel, pandas, R, etc.

## Notes

- Files are saved with metadata as comment lines (starting with `#`)
- Pandas automatically skips comment lines when reading with `comment='#'`
- History is pruned AFTER creating new backup (keeps most recent 5)
- Backup only created if current folder has CSV files
- All file operations include error handling with user notifications
