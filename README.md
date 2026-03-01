# Code Repository for Paper: VA-RBTC

Welcome to the code repository for the ongoing paper titled *Battery as a lifter: Decoupling energy efficiency from deep tunneling in metro vertical alignment*. The goal of this project is to jointly optimize underground metro track vertical alignment and train control to minimize energy consumption, with optional Pareto analysis trading off alignment depth vs. energy. All code is made available under the MIT License.

The solver backend is **Gurobi** (v12). A valid Gurobi license is required to run any optimization.

## Environment

- Python 3.12, Gurobi 12.0.1
- Install dependencies: `pip install -r requirements.txt`
- All scripts must be run from the **project root** (not from `src/`), as file paths like `data/OESD/...` and `result/...` are relative to root.

## Running the Code

Run a single case optimization (from project root):
```bash
python src/model_full.py      # runs case 10030222 with depth_cap demo
python src/pareto.py          # runs Pareto sweep for multiple cases
python src/plot.py            # plots compact result for a saved .sol file
```

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
   src/pareto.py :: pareto_sweep → ParetoPoint list → CSV + PDF
   src/plot.py   :: plot_compact(case, sol) → matplotlib Figure
         ↓
   result/<save_dir>/<case_id>.{log,sol,json}
```

### Key modules

**`src/prepare.py`** — Decodes case_id into a `case` dict containing all physical parameters (geometry, train specs, OESD specs, time bounds). `get_case()` is the single entry point used everywhere.

**`src/model_core.py`** — Pure Gurobi model-building functions; no solving here:
- `model_va` — vertical alignment (VA) MIP: elevation variables `e[s]`, slope binary `pi[s]`
- `model_rbtc` — regenerative braking train control (RBTC): per-interval velocity, force, energy, time; supports `l2r` (left-to-right) and `r2l` directions; variables prefixed `L2R_` or `R2L_`
- `model_oesd` — onboard energy storage (OESD): SOE `varpi`, charge/discharge power `kappa_plus/minus`, energy `xi`
- Cut functions (`model_EC`, `model_SC`, `model_TC`, `model_FC`, `model_VC`, `model_MS`) — valid inequalities / logic cuts to tighten the relaxation; all enabled by default in `FullModel.build()`

**`src/model_full.py`** — `FullModel(gp.Model)` orchestrates build → warm start → solve → save:
- `build()` assembles all sub-models and cuts, sets objective
- `reconfigure()` swaps objective/caps without rebuilding (used in Pareto sweep)
- `get_sequential_solution()` — 3-stage sequential heuristic (VA → RBTC → OESD) for warm start
- `inject_warm_start()` — injects solution as MIP start hints (`VarHintVal`) or hard starts (`Start`)
- `optimize_()` — wraps `optimize()` with log/solution file saving; writes `.log`, `.sol`, `.json` to `result/<save_dir>/`
- `Solution` dataclass — holds `va_sol`, `rbtc_sol`, `oesd_sol` dicts; can be constructed from a live model or loaded from `.sol`/`.json` files

**`src/pareto.py`** — ε-constraint Pareto sweep over alignment depth:
- Builds model once, then `reconfigure()` per step (efficient)
- Warm-starts each step from previous solution; step 0 uses `get_sequential_solution()`
- Saves per-step results under `result/<save_dir>/step{i:03d}_depth{d:.2f}/`
- Exports `pareto_results.csv` and two PDF plots per sweep

**`src/plot.py`** — `plot_compact(case, sol, l2r)` produces a 3-or-4-panel figure (elevation+speed, force+time, cumulative energy, SOE)

### Result files

Each solved case writes to `result/<save_dir>/<case_id>.{log,sol,json}`:
- `.log` — Gurobi solver log + case quick-text + detailed variable/constraint dump
- `.sol` — Gurobi solution file (variable name → value, space-separated)
- `.json` — Gurobi JSON solution (array of `{VarName, X}`)

## Objective and Constraints

The optimization jointly minimizes total traction energy (net + OESD losses). The Pareto sweep uses an ε-constraint on `depth_cap` (max track depth below lower platform elevation) to trace the energy–depth frontier. The `is_oesd_included` flag controls whether OESD energy flows enter the objective.

## Citation

If you wish to cite this work, please use the following BibTeX entry:

```bibtex
@article{
    TO BE ADDED
}
```
