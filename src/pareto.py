"""
Pareto analysis: Alignment Depth vs Energy Consumption.

Strategy: ε-constraint sweep on depth_cap (shallow → deep), single model + reconfigure.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib
params = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",  
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
matplotlib.rcParams.update(params)
import matplotlib.pyplot as plt
import numpy as np

from src.prepare import get_case
from src.model_core import solve_edge_va
from src.model_full import FullModel, Solution
from src.utils import log_info


# ═══════════════════════════════════════════════════════════════
#  Config & Result
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParetoConfig:
    n_points: int = 12
    time_limit_per_point: float = 120.0
    mip_gap: float = 0.05
    is_oesd_included: bool = True
    TC_on: bool = True  # valid inequalities proofed
    VC_on: bool = True  # valid inequalities proofed
    MS_on: bool = True  # valid inequalities proofed
    EC_on: bool = True  # valid inequalities proofed
    SC_on: bool = False # heuristic accelerators
    FC_on: bool = False # heuristic accelerators
    cyclic: bool = True


@dataclass
class ParetoPoint:
    step: int
    depth_cap: float
    energy: float
    energy_bound: float
    depth: float
    gap: float
    runtime: float
    status: int
    solution: Solution
    is_feasible: bool = True

    @property
    def abs_gap(self) -> float:
        """Absolute optimality gap in energy units."""
        return self.energy - self.energy_bound

# ═══════════════════════════════════════════════════════════════
#  Sweep
# ═══════════════════════════════════════════════════════════════

def _compute_depth_range(case: dict) -> tuple[float, float]:
    lower_platform_ele = min(case['P1'][1], case['P2'][1])
    _, e_lowest_dict = solve_edge_va(case, output_flag=0, lowest_edge=True)
    depth_max = lower_platform_ele - min(e_lowest_dict.values())
    return 2.0, depth_max


def pareto_sweep(
    case_id: int,
    cfg: ParetoConfig | None = None,
    save_dir: str = "pareto",
    **kwargs,
) -> list[ParetoPoint]:
    if cfg is None:
        cfg = ParetoConfig()

    case = get_case(case_id)
    depth_min, depth_max = _compute_depth_range(case)
    depth_caps = np.linspace(depth_min, depth_max, cfg.n_points)

    log_info(f"[Pareto] depth ∈ [{depth_min:.3f}, {depth_max:.3f}] m, {cfg.n_points} points")

    fm = FullModel(case_id, save_dir=save_dir)
    fm.build(
        TC_on=cfg.TC_on,  # valid inequalities proofed
        VC_on=cfg.VC_on,  # valid inequalities proofed
        MS_on=cfg.MS_on,  # valid inequalities proofed
        EC_on=cfg.EC_on,  # valid inequalities proofed
        SC_on=cfg.SC_on, # heuristic accelerators
        FC_on=cfg.FC_on,  # heuristic accelerators
        is_oesd_included=cfg.is_oesd_included,
        depth_cap=depth_caps[0],
        cyclic=cfg.cyclic,
    )

    results: list[ParetoPoint] = []
    prev_solution: Solution | None = None
    t_total = time.time()

    for i, d_cap in enumerate(depth_caps):
        log_info(f"\n{'='*60}")
        log_info(f"[Pareto] Step {i}/{cfg.n_points - 1}:  depth_cap = {d_cap:.4f} m")

        # ===== reconfigure =====
        if i > 0:
            fm.reconfigure(
                depth_cap=d_cap,
                is_oesd_included=cfg.is_oesd_included,
            )
            
        step_dir = f"{save_dir}/step{i:03d}_depth{d_cap:.2f}"
        os.makedirs(f"result/{step_dir}", exist_ok=True)
        fm._log_file_path = f"result/{step_dir}/{case_id}.log"
        
        # ===== warm start =====
        if prev_solution is not None and not prev_solution.is_empty:
            fm.inject_warm_start(prev_solution, binary_only=False, use_hint=True)
        elif i == 0:
            try:
                seq_sol = fm.get_sequential_solution()
                assert not seq_sol.is_empty, "sequential solution is empty"
                fm.inject_warm_start(seq_sol, binary_only=False, use_hint=True)
            except (RuntimeError, AssertionError):
                log_info(f"[Pareto] step {i}: sequential solution failed, solving cold.")

        # ===== solve =====
        t0 = time.time()
        fm.optimize_(
            save_on=True,
            TimeLimit=cfg.time_limit_per_point,
            MIPGap=cfg.mip_gap,
            **kwargs,
        )
        runtime = time.time() - t0

        # ===== extract =====
        if fm.SolCount >= 1:
            sol = Solution.from_model(fm)
            point = ParetoPoint(
                step=i, depth_cap=d_cap,
                energy=sol.obj_energy, 
                energy_bound=fm.ObjBound,
                depth=sol.obj_depth,
                gap=fm.MIPGap, runtime=runtime, status=fm.Status,
                solution=sol,
            )
            results.append(point)
            prev_solution = sol
            log_info(
                f"[Pareto] ✓ energy={point.energy:.1f}  bound={point.energy_bound:.1f}  "
                f"depth={point.depth:.3f}  gap={point.gap:.4f}  time={runtime:.1f}s"
            )
        else:
            log_info(f"[Pareto] ✗ INFEASIBLE/NO SOL  status={fm.Status}  time={runtime:.1f}s")

    fm.dispose()
    log_info(f"\n[Pareto] Done: {len(results)}/{cfg.n_points} solved in {time.time()-t_total:.1f}s")
    
    export_csv(results, f"result/{save_dir}/pareto_results.csv")
    plot_pareto(results, save_path=f"result/{save_dir}/pareto_front.pdf")
    plot_pareto_with_gaps(results, save_path=f"result/{save_dir}/pareto_front_gaps.pdf")
    return results


# ═══════════════════════════════════════════════════════════════
#  Pareto front filter
# ═══════════════════════════════════════════════════════════════

def filter_non_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """(energy ↓, depth ↓) — 按 depth 升序扫描，保留 energy 递减的点"""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p.depth)
    front, best_e = [], np.inf
    for pt in sorted_pts:
        if pt.energy < best_e - 1e-6:
            front.append(pt)
            best_e = pt.energy
    return front

def filter_non_dominated_bound(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Pareto front through energy_bound (optimistic / best-case front)."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p.depth)
    front, best_e = [], np.inf
    for pt in sorted_pts:
        if pt.energy_bound < best_e - 1e-6:
            front.append(pt)
            best_e = pt.energy_bound
    return front


# ═══════════════════════════════════════════════════════════════
#  Export & Plot
# ═══════════════════════════════════════════════════════════════

def export_csv(points: list[ParetoPoint], filepath: str):
    front_steps = {pt.step for pt in filter_non_dominated(points)}
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "depth_cap", "energy", "energy_bound", "depth", 
                    "gap", "runtime", "status", "is_pareto"])
        for pt in points:
            w.writerow([pt.step, pt.depth_cap, pt.energy, pt.energy_bound, pt.depth,
                        pt.gap, pt.runtime, pt.status, pt.step in front_steps])


def plot_pareto(points: list[ParetoPoint], save_path: str | None = None):
    import matplotlib.pyplot as plt

    front = filter_non_dominated(points)
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.scatter([p.depth for p in points], [p.energy / 3600 for p in points],
               c="silver", s=40, edgecolors="grey", zorder=2, label="Feasible")
    ax.plot([p.depth for p in front], [p.energy / 3600 for p in front],
            "ro-", ms=7, zorder=3, label="Pareto front")
    # for pt in front:
    #     ax.annotate(f"s{pt.step}", (pt.depth, pt.energy / 3600),
    #                 textcoords="offset points", xytext=(5, 5), fontsize=7, color="red")

    ax.set_xlabel("Alignment Depth (m)")
    ax.set_ylabel("Energy Consumption (kWh)")
    # ax.set_title("Pareto Front: Depth vs Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig



# ═══════════════════════════════════════════════════════════════
#  Plot: gap bars + dual Pareto fronts
# ═══════════════════════════════════════════════════════════════

def plot_pareto_with_gaps(
    points: list[ParetoPoint],
    save_path: Optional[str] = None,
    energy_unit: str = "kWh",
    show_bound_front: bool = True,
    figsize: tuple = (5, 4),
):
    """
    Pareto plot where each solution is a bar from [ObjBound @ depth_cap] to [ObjVal @ depth].

    Preprocessing:
      1. Sort points by depth_cap ascending.
      2. Reverse-cummax on energy_bound → monotonic outer approximation.
      3. Raw bounds kept for scatter dots; corrected bounds for outer front line.
      4. Gap bars are diagonal: (depth_cap, bound) → (depth, incumbent).
    """
    scale = 3600.0 if energy_unit == "kWh" else 1.0

    # ── 0. Sort by depth_cap ──
    pts_sorted = sorted(points, key=lambda p: p.depth_cap)

    # ── 1. Reverse cummax on energy_bound → valid outer approximation ──
    raw_bounds = np.array([p.energy_bound for p in pts_sorted])
    corrected_bounds = raw_bounds.copy()
    running_max = -np.inf
    for i in range(len(corrected_bounds) - 1, -1, -1):
        running_max = max(running_max, corrected_bounds[i])
        corrected_bounds[i] = running_max

    # ── 2. Inner Pareto front (incumbent, non-dominated) ──
    front = filter_non_dominated(pts_sorted)

    fig, ax = plt.subplots(figsize=figsize)

    depths_all   = [p.depth     for p in pts_sorted]
    dcaps_all    = [p.depth_cap for p in pts_sorted]
    bw = (max(depths_all) - min(depths_all)) / len(pts_sorted) * 0.35

    # ── 3. Gap bars: diagonal (depth_cap, bound) → (depth, incumbent) ──
    for idx, pt in enumerate(pts_sorted):
        e_lo = corrected_bounds[idx] / scale
        e_hi = pt.energy / scale
        label = "Solve pair" if idx == len(pts_sorted) - 1 else None
        ax.plot([pt.depth_cap, pt.depth], [e_lo, e_hi],
                color="#B0B0B0", linewidth=1.2, solid_capstyle="round",
                zorder=2, label=label)

    # ── 4. Incumbent dots (at depth) ──
    ax.scatter(depths_all, [p.energy / scale for p in pts_sorted],
               c="white", s=30, edgecolors="#555555", linewidths=0.8,
               zorder=4, label="Incumbent")

    # ── 5. Bound dots (raw bounds, at depth_cap) ──
    ax.scatter(dcaps_all, raw_bounds / scale,
               c="white", s=18, edgecolors="#AAAAAA", linewidths=0.6,
               marker="v", zorder=4, label="Lower Bound")

    # ── 6. Pareto front – inner approximation (incumbent) ──
    if front:
        ax.plot([p.depth for p in front], [p.energy / scale for p in front],
                "o-", color="#E74C3C", ms=5, linewidth=1.5,
                zorder=5, label="Pareto (Inner approx.)")

    # ── 7. Pareto front – outer approximation (corrected bounds) ──
    if show_bound_front:
        ax.plot(dcaps_all, corrected_bounds / scale,
                "v--", color="#3498DB", ms=4, linewidth=1.2,
                zorder=5, label="Pareto (Outer approx.)")

    # ── 8. Gap region shade ──
    if front and show_bound_front:
        d_inc = np.array([p.depth for p in front])
        e_inc = np.array([p.energy / scale for p in front])
        d_bnd = np.array(dcaps_all)
        e_bnd = corrected_bounds / scale

        d_common = np.sort(np.unique(np.concatenate([d_inc, d_bnd])))
        e_inc_interp = np.interp(d_common, d_inc, e_inc)
        e_bnd_interp = np.interp(d_common, d_bnd, e_bnd)

        ax.fill_between(d_common, e_bnd_interp, e_inc_interp,
                        alpha=0.10, color="#9B59B6", zorder=1,
                        label="Gap region")

    ax.set_xlabel("Alignment Depth (m)")
    ax.set_ylabel(f"Energy Consumption ({energy_unit})")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for case_id in [
        # 10000022, 10000023, 10000024, 10000025, 10000026,
        # 10010022, 10010023, 10010024, 10010025, 10010026,
        # 10020022, 10020023, 10020024, 10020025, 10020026,
        # 10030022, 10030023, 10030024, 10030025, 10030026,
        # 10040022, 10040023, 10040024, 10040025, 10040026,
        
        10000024
    ]:
        cfg = ParetoConfig(
            n_points=20, 
            time_limit_per_point=15, 
            mip_gap=0.01,
            TC_on=False,
            VC_on=False,
            MS_on=False,
            EC_on=True,
            SC_on=False,
            FC_on=False,
            cyclic=True,
        )
        results = pareto_sweep(case_id, cfg=cfg, save_dir=f"pareto_test/{case_id}", Heuristics=0.7, MIPFocus=1)
