# Quick Reference: Data Export Feature

## What Was Added

Three buttons in the Streamlit app now automatically save data:

| Button | Saves To | Contains |
|--------|----------|----------|
| **Simulate** | `data/current/sim_data.csv` | Particle trajectories (x, y positions over time) |
| **Train CNF** | `data/current/train_data.csv` | Training loss per epoch/step |
| **Animate** | `data/current/meta_data.csv` | Animation parameters & statistics |

## Directory Structure

```
thermal_flow_cnf/data/
├── current/              ← Latest data (3 files)
│   ├── sim_data.csv
│   ├── train_data.csv
│   └── meta_data.csv
│
└── history/              ← Previous versions (max 5)
    ├── 20251109_143022/  ← Oldest kept
    ├── 20251109_151530/
    ├── 20251109_162045/
    ├── 20251109_174512/
    └── 20251109_183759/  ← Most recent
```

## How It Works

1. **Click a button** (Simulate, Train CNF, or Animate)
2. **Current data backed up** to `history/[timestamp]/`
3. **New data saved** to `current/`
4. **Old versions pruned** (keeps only 5 most recent)
5. **Toast notification** confirms save

## Key Features

✅ **Automatic versioning** - Every save creates a timestamped backup  
✅ **History limit** - Keeps only 5 most recent versions  
✅ **Parameter tracking** - CSV headers include all settings  
✅ **Standard format** - Easy to import into Excel, pandas, MATLAB, etc.  
✅ **Error handling** - Graceful failures with user notifications  

## Example: Loading Saved Data

```python
import pandas as pd

# Read current simulation data
sim = pd.read_csv('thermal_flow_cnf/data/current/sim_data.csv', comment='#')

# Read training data
train = pd.read_csv('thermal_flow_cnf/data/current/train_data.csv', comment='#')

# Read animation metadata
meta = pd.read_csv('thermal_flow_cnf/data/current/meta_data.csv', comment='#')
```

## What Gets Saved

### Simulation Data
- Particle ID
- Time step
- Position (x, y)
- Initial position (x0, y0)
- Parameters: flow type, num particles, timesteps, etc.

### Training Data
- Epoch number
- Step within epoch
- Loss value
- Average log probability
- Bits per dimension
- Parameters: batch size, learning rate, hidden dim, etc.

### Animation Metadata
- Number of particles (true vs predicted)
- Frame count
- MSD (mean squared displacement)
- Trajectory bounds (min/max x, y)
- Model parameters: checkpoint, hidden dim, flow type, etc.

## Files Created

- `thermal_flow_cnf/src/utils/data_manager.py` - Core functionality
- `DATA_EXPORT_README.md` - Full documentation
- `test_data_simple.py` - Test script
- `test_pruning.py` - Pruning verification

## No Action Required!

The feature works automatically. Just use the Streamlit app as normal, and your data will be saved and versioned automatically.
