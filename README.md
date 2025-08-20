# Code Repository for Paper: VA-RBTC

Welcome to the code repository for the ongoing paper titled *Energy-saving Design of Underground Metro Vertical Alignment Considering Train Control with Onboard Energy Storage Devices*. This research is intended for potential publication in *Tunnelling and Underground Space Technology* (incorporating Trenchless Technology Research).

This repository contains an optimization framework for designing energy-efficient metro vertical alignments while considering regenerative braking train control (RBTC) and onboard energy storage devices (OESD). The framework uses Gurobi optimization to solve the integrated design problem.
All code is made available under the MIT License.

## Prerequisites
- Python 3.8+
- Gurobi optimizer (with valid license)
- Required packages:
  ```bash
  pip install gurobipy numpy pandas matplotlib pwlf
  ```

## Repository Structure

```bash
src/
├── optimize.py      # Main optimization model class
├── opt_model.py     # Model building functions (VA, RBTC, OESD, constraints)
├── opt_ws.py        # Warm start solution generation
├── opt_utils.py     # Optimization utilities
├── prepare.py       # Case preparation and configuration
├── train.py         # Train class and data
├── oesd.py          # OESD class and characteristics
├── analyze.py       # Results analysis functions
├── plot.py          # Visualization functions
└── pwa.py           # Piecewise linear approximation utilities

data/
├── train/          # Train configuration files
└── OESD/           # OESD specification files
```

## Quick Start

```python
from src.optimize import VA_RBTC_OESD

# Create optimization model with case ID
model = VA_RBTC_OESD(case_id=10030222, save_dir="results")

# Build model with default settings
model.build(
    va_solution: dict | None = None,      # if VA fixed, provide the solution
    EC_on: bool = True,                   # Elevation-based cuts
    TC_on: bool = True,                   # Time-based cuts
    SC_on: bool = True,                   # Slope-based cuts
    FC_on: bool = True,                   # Force-based cuts (not used in this paper)
    VC_on: bool = True,                   # Velocity-based cuts (not used in this paper)
    MS_on: bool = True,                   # Mode-switching cuts (not used in this paper)
    obj_net: bool = False,                # Optimize total energy (net + OESD) or net energy
    with_nonlinear_t: bool = False,       # use gurobi's embedded nonlinear method for time variable
    use_warm_start: bool = True,          # Implement warm-start pipeline
    lift_va: bool | float | int = True,   # Use VA sweep for warm start
    debug_mode: bool = False,             # Extensive log info and test warm-start solution feasibility with locked variables
    opt_depth: bool = False,              # Change objective to lifting VA depth (very difficult to solve)
    energy_cap: float | None = None,      # Energy upper bound, should be the objVal of basedline model 
    pareto_alpha: float | None = None,    # NOT USED FOR NOW
)

# Optimize with time limit
model.optimize_(save_on=True, MIPGap=0.01, TimeLimit=300, **Gurobi_parameters)

```

## Case id systems

Case ID System
The 8-digit case ID encodes problem parameters:

- Digit 1: Section ID (1-3)
- Digit 2: Train type (0-2)
  - 0: Wu2021Train
  - 1: Wu2024Train
  - 2: Scheepmaker2020
- Digit 3: Train load (0-3, passengers/m²)
  - 0: No load
  - 1: 3 passengers/m²
  - 2: 6 passengers/m²
  - 3: 9 passengers/m²
- Digit 4: OESD type (0: None, 1: Supercapacitor, 2: Li-ion, 3: Flywheel)
- Digit 5: Initial SOE (0: 60%, 1: 100%)
- Digit 6: Number of intervals (0: 25, 1: 50, 2: 100, etc.)
- Digit 7: Direction (0: L→R, 1: R→L, 2: Both)
- Digit 8: Section time range index
  - (98, 108)  # very likely to be infeasible
  - (98, 118)
  - (98, 128)
  - (98, 138)
  - (98, 180)

Example: 10020222 means Section 1, Wu2021Train, no load, Li-ion battery, 60% initial SOE, 100 intervals, both directions, time range (98, 128).

## Modules

### `optimize.py` - Main optimization model

Contains the `VA_RBTC_OESD` class which is the main optimization model for Vertical Alignment, Regenerative Braking Train Control, and Onboard Energy Storage Devices. This is an inherited class of `gurobipy.Model`, so all methods like `.getVars()`, `.Status`, `.getObjVal()`, etc. are available.

Key methods:
- `__init__(self, case_id:int, save_dir:str|None=None)`: Initialize the optimization model with case ID and optional save directory.
- 
- `build(self)`: Use either fixed VA solution or build the optimization model.
- `optimize_(self, save_on:bool=True, MIPGap:float=0.01, TimeLimit:int=300, **Gurobi_parameters)`: Optimize the model with specified parameters. 

Attributes:
- case: dict, the case parameters
- train: Train object
- oesd: OESD object
- va_variables: dict, the vertical alignment variables
- rbtc_variables: dict, the regenerative braking train control variables
- oesd_variables: dict, the onboard energy storage device variables


### `opt_model.py` - Optimization model

Contains all mathemetical models for the optimization problem.

- `model_va()`: Vertical alignment model
- `model_rbtc()`: Regenerative braking train control model
- `model_oesd()`: Onboard energy storage device model
- `solve_edge_va()`: Solve the vertical alignment model for the edge case (min/max sum of e)
- `model_EC()`: Elevation-based cuts
- `model_TC()`: Time-based cuts
- `model_SC()`: Slope-based cuts
- `model_FC()`: Force-based cuts (not used in this paper)
- `model_VC()`: Velocity-based cuts (not used in this paper)
- `model_MS()`: Mode-switching cuts (not used in this paper)

### `opt_ws.py` - warm-start pipeline

```python
# Generate single warm start solution
from src.opt_ws import gen_warm_start_sol

ws_solution = gen_warm_start_sol(
    case: dict,
    use_running_time_exhaustive: bool = True,
    lift_va: bool | float = False,  # False: lowest VA, True: sweep, float: fixed lift
    delta_e_lift: float = 0.5,      # Lift amount if lift_va is True
    tighten_energy: bool = True,    # Additional energy optimization
    obj_net: bool = True
)

# Run VA floor sweep to find best warm start
from src.opt_ws import run_va_floor_sweep, SweepConfig

config = SweepConfig(
    delta_e_step=0.2,    # Step size for lifting VA
    rounds=25,           # Number of sweep rounds
    early_stop_tol=0.0,  # Early stopping tolerance
)

best_sol, best_obj, best_delta, history, info = run_va_floor_sweep(
    case: dict,
    obj_net: bool,
    cfg: config
)
```

### `analyze.py` - Analysis tools for the saved .json files

```python
from src.analyze import anl_results, anl_variables, anl_one_case

# Load optimization results
results = anl_results(case_id=10030222, folder="results")

# Extract specific variables
elevations = anl_variables("e", results)
speeds_l2r = anl_variables("L2R_v", results)

# Get comprehensive case analysis
case_info = anl_one_case(case_id=10030222, folder="results")
# Returns: train type, OESD type, energy consumption, runtime, gap, etc.
```

## Citation

If you wish to cite this work, please use the following BibTeX entry:

```bibtex
@article{
    TO BE ADDED
}
```