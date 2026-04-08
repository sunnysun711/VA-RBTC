# Code Repository for Paper: VA-RBTC

Welcome to the code repository for the ongoing paper titled *Battery as a lifter: Decoupling energy efficiency from deep tunneling in metro vertical alignment*. The goal of this project is to jointly optimize underground metro track vertical alignment and train control to minimize energy consumption, together with Pareto analysis trading off alignment depth vs. energy. All code is made available under the MIT License.

The solver backend is **Gurobi**. A valid Gurobi license is required to run any optimization.

## Environment

- Python 3.12, Gurobi 13.0.0
- Install dependencies: `pip install -r requirements.txt`
- All scripts must be run from the **project root** (not from `src/`), as file paths like `data/OESD/...` and `result/...` are relative to root.

## Running the Code

Run from the project root:
```bash
python -m src.model_full      # single-case build + warm-start + solve demo
python -m src.pareto          # Pareto depth–energy sweep over configured cases
python -m src.plot            # plots compact result for a saved .sol file
```

### Single-case run ([src/model_full.py](src/model_full.py))

The `__main__` block in [src/model_full.py](src/model_full.py) shows the canonical single-case workflow: instantiate `FullModel(case_id, save_dir=...)`, call `build(...)` with the desired cut/feature switches and caps, generate a warm start via `get_sequential_solution()`, inject it through `inject_warm_start(...)`, then call `optimize_(...)` with Gurobi parameters (e.g. `TimeLimit`, `MIPGap`, `Heuristics`, `MIPFocus`). Results are written to `result/<save_dir>/<case_id>.{log,sol,json}`.

**Re-solving with different caps via `reconfigure()`.** After a first solve, you can change the `depth_cap` / `energy_cap` or flip between energy- and depth-minimization objectives **without rebuilding the model** — `reconfigure()` only swaps the objective and removes/re-adds the cap constraints. This is much cheaper than instantiating a new `FullModel`, and it is exactly what the Pareto sweep relies on internally. A typical pattern (see the commented block at the bottom of [src/model_full.py](src/model_full.py)):

```python
fm = FullModel(case_id, save_dir=f"{case_id}/rectangle")
fm.build(depth_cap=2.0, cyclic=True, ...)       # tight depth cap first
fm.optimize_(save_on=True, TimeLimit=1200, MIPGap=0.01)

sol_tight = Solution.from_model(fm)              # keep the previous solution

fm.reconfigure(depth_cap=6.0)                    # loosen the depth cap, same model
fm.inject_warm_start(sol_tight, use_hint=True)   # optional: reuse previous solution
fm.optimize_(save_on=True, TimeLimit=30)
```

Other `reconfigure()` use cases: set `opt_depth=True, energy_cap=...` to switch from "min energy s.t. depth ≤ d" to "min depth s.t. energy ≤ E"; pass `energy_cap=None` / `depth_cap=None` to drop an existing cap constraint; toggle `is_oesd_included` to rebuild the objective with/without OESD losses.

### Pareto sweep ([src/pareto.py](src/pareto.py))

The `__main__` block in [src/pareto.py](src/pareto.py) is the entry point for depth–energy Pareto analysis. The usage pattern is:

```python
from src.pareto import ParetoConfig, pareto_sweep

cfg = ParetoConfig(
    n_points=20,                 # number of depth_cap steps
    time_limit_per_point=15,     # Gurobi TimeLimit per step
    mip_gap=0.01,
    TC_on=False, VC_on=False, MS_on=False,   # valid inequalities (proven)
    EC_on=True,                               # valid inequalities (proven)
    SC_on=False, FC_on=False,                 # heuristic accelerators
    cyclic=True,                              # cyclic SOE condition
)

results = pareto_sweep(
    case_id=10000024,
    cfg=cfg,
    save_dir=f"pareto_test/10000024",
    Heuristics=0.7, MIPFocus=1,  # forwarded to Gurobi as model params
)
```

`pareto_sweep` builds the model once, then calls `FullModel.reconfigure(depth_cap=...)` per step to avoid rebuilding. Each step warm-starts from the previous step's solution (step 0 uses `get_sequential_solution()`). Per-step artifacts land in `result/<save_dir>/step{i:03d}_depth{d:.2f}/`; the sweep-level outputs `pareto_results.csv`, `pareto_front.pdf`, and `pareto_front_gaps.pdf` land directly in `result/<save_dir>/`.

Run Jupyter notebooks for analysis:
```bash
jupyter notebook scripts/
```

## Case ID Encoding

A case is fully identified by an 8-digit integer (e.g., `10030222`). Each digit encodes a parameter:

| Digit position | Parameter | Lookup list in `prepare.py` |
|---|---|---|
| 0 | `section_id` (1–3) | `SECTIONS` |
| 1 | `train_id` | `TRAINS` |
| 2 | `train_load_id` | `TRAIN_LOADS` |
| 3 | `oesd_id` | `OESDS` |
| 4 | `varpi_0_id` | `VARPI_0s` |
| 5 | `S_id` (num intervals) | `NUM_INTERVALS` |
| 6 | `direction_id` | `DIRECTIONS` |
| 7 | `section_time_id` | `SECTION_TIMES` |

To add new parameter variants, append to the corresponding list in `src/prepare.py`. **Changing `SECTION_TIMES` shifts all section_time_id mappings** — existing result directories are named by case_id only, so re-running with modified lists may mismatch saved results.

## Architecture

### Data flow

```
data/train/*.json + data/OESD/*.json
         ↓
   src/prepare.py :: get_case(case_id) → case dict
         ↓
   src/model_core.py :: model_va / model_rbtc / model_oesd / [cut functions]
         ↓
   src/model_full.py :: FullModel(gp.Model) + Solution
         ↓
   src/pareto.py :: pareto_sweep → ParetoPoint list → CSV + PDFs
   src/plot.py   :: plot_compact(case, sol) → matplotlib Figure
         ↓
   result/<save_dir>/<case_id>.{log,sol,json}
```

### Key modules

**[src/prepare.py](src/prepare.py)** — Decodes `case_id` into a `case` dict containing all physical parameters (geometry, train specs, OESD specs, time bounds). `get_case()` is the single entry point used everywhere.

**[src/model_core.py](src/model_core.py)** — Pure Gurobi model-building functions; no solving here:

- `model_va` — vertical alignment (VA) MIP: elevation variables `e[s]`, slope binaries, lowest-elevation variable `e_bar`
- `model_rbtc` — regenerative braking train control (RBTC): per-interval velocity, force, energy, time; supports `l2r` (left-to-right) and `r2l` directions; variables prefixed `L2R_` or `R2L_`
- `model_oesd` — onboard energy storage (OESD): SOE `varpi`, charge/discharge power `kappa_plus/minus`, energy `xi`; supports cyclic SOE condition
- Cut functions — `model_EC`, `model_TC`, `model_VC`, `model_MS` are **proven valid inequalities**; `model_SC`, `model_FC` are **heuristic accelerators**. All are optional and independently toggled in `FullModel.build()`.
- Helpers — `solve_edge_va` computes the edge-case lowest-elevation VA used to derive the depth range; `get_energy_expr_one_direction` builds the directional traction-energy expression.

**[src/model_full.py](src/model_full.py)** — `FullModel(gp.Model)` orchestrates build → warm start → solve → save:

- `build()` assembles all sub-models and cuts according to the `*_on` flags, then delegates to `reconfigure()` for objective and cap constraints. Key switches: `is_oesd_included`, `opt_depth`, `depth_cap`, `energy_cap`, `cyclic`, `power_time_trapezoid`.
- `reconfigure()` swaps objective and `depth_cap` / `energy_cap` constraints **without rebuilding** — this is what lets the Pareto sweep reuse a single model across all steps.
- `get_sequential_solution()` — 3-stage sequential heuristic (VA → RBTC → OESD) producing a feasible `Solution` for warm start.
- `inject_warm_start()` — injects a `Solution` as either MIP start hints (`VarHintVal`, `use_hint=True`) or hard starts (`Start`).
- `optimize_()` — wraps `optimize()` with log/solution file saving; writes `.log`, `.sol`, `.json` (and `.ilp` on infeasibility) under `result/<save_dir>/`. Extra `**kwargs` are forwarded to Gurobi as model parameters.
- `Solution` dataclass — holds `va_sol`, `rbtc_sol`, `oesd_sol` dicts plus `obj_energy`/`obj_depth`. Can be constructed from a live model (`Solution.from_model`) or loaded from `.sol`/`.json` files (`Solution.from_file`).

**[src/pareto.py](src/pareto.py)** — ε-constraint Pareto sweep over alignment depth:

- `ParetoConfig` — configures the sweep: `n_points`, `time_limit_per_point`, `mip_gap`, `is_oesd_included`, cut toggles (`EC_on`/`TC_on`/`VC_on`/`MS_on`/`SC_on`/`FC_on`), and `cyclic`.
- `pareto_sweep(case_id, cfg, save_dir, **gurobi_kwargs)` — automatically derives the depth range from `solve_edge_va`, builds the model once, then `reconfigure()`s `depth_cap` across `n_points` uniformly spaced values. Each step warm-starts from the previous solution; step 0 uses `get_sequential_solution()`. Per-step artifacts go to `result/<save_dir>/step{i:03d}_depth{d:.2f}/`.
- `ParetoPoint` — per-step record with `energy`, `energy_bound`, `depth`, `gap`, `runtime`, `status`, and the full `Solution`.
- `filter_non_dominated` / `filter_non_dominated_bound` — extract the incumbent / lower-bound Pareto fronts.
- `export_csv`, `plot_pareto`, `plot_pareto_with_gaps` — write `pareto_results.csv`, `pareto_front.pdf`, and `pareto_front_gaps.pdf` (the gaps plot visualizes inner/outer approximations and the gap region).

**[src/plot.py](src/plot.py)** — `plot_compact(case, sol, l2r)` produces a 3-or-4-panel figure (elevation+speed, force+time, cumulative energy, SOE).

### Result files

Each solved case writes to `result/<save_dir>/<case_id>.{log,sol,json}`:

- `.log` — Gurobi solver log + case quick-text + warm-start info + detailed variable/constraint dump
- `.sol` — Gurobi solution file (variable name → value, space-separated)
- `.json` — Gurobi JSON solution (array of `{VarName, X}`)
- `.ilp` — written only when the model is infeasible (IIS constraints)

## Objective and Constraints

The optimization jointly minimizes total traction energy (net + OESD losses). The Pareto sweep uses an ε-constraint on `depth_cap` (max track depth below lower platform elevation) to trace the energy–depth frontier; alternatively, `FullModel.reconfigure(opt_depth=True, energy_cap=...)` can minimize depth subject to an energy budget. The `is_oesd_included` flag controls whether OESD energy flows enter the objective.

## Citation

If you wish to cite this work, please use the following BibTeX entry:

```bibtex
@article{
    TO BE ADDED
}
```

## Acknowledgement

This README was drafted by [Claude Code](https://claude.com/claude-code) and proofread by the author.
