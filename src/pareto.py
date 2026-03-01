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

    # ===== 单次 build =====
    fm = FullModel(case_id, save_dir=save_dir)
    fm.build(
        depth_cap=depth_caps[0],
        is_oesd_included=cfg.is_oesd_included,
    )

    results: list[ParetoPoint] = []
    prev_solution: Solution | None = None
    t_total = time.time()

    for i, d_cap in enumerate(depth_caps):
        log_info(f"\n{'='*60}")
        log_info(f"[Pareto] Step {i}/{cfg.n_points - 1}:  depth_cap = {d_cap:.4f} m")

        # ===== reconfigure（首步已在 build 中配好，跳过） =====
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
    Pareto plot where each solution is a vertical bar [ObjBound, ObjVal].

    - Top of bar   = incumbent (ObjVal, current best feasible)
    - Bottom of bar = ObjBound (proven lower bound from MIP solver)
    - Bar height    = absolute gap → shorter = tighter = more confidence
    - Red line      = Pareto front through incumbents (conservative)
    - Blue dashed   = Pareto front through bounds (optimistic best-case)
    - Purple shade  = gap region between the two fronts
    """
    scale = 3600.0 if energy_unit == "kWh" else 1.0

    front = filter_non_dominated(points)
    front_bound = filter_non_dominated_bound(points) if show_bound_front else []

    fig, ax = plt.subplots(figsize=figsize)

    depths_all = [p.depth for p in points]
    bar_width = (max(depths_all) - min(depths_all)) / len(points) * 0.35

    # ── 1. Gap bars (vertical segments with caps) ──
    for pt in points:
        e_lo = pt.energy_bound / scale
        e_hi = pt.energy / scale
        color = "#B0B0B0"
        ax.plot([pt.depth, pt.depth], [e_lo, e_hi],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=2)
        for e_end in (e_lo, e_hi):
            ax.plot([pt.depth - bar_width / 2, pt.depth + bar_width / 2], [e_end, e_end],
                    color=color, linewidth=1.2, zorder=2)

    # ── 2. Incumbent dots (top) ──
    ax.scatter(depths_all, [p.energy / scale for p in points],
               c="white", s=30, edgecolors="#555555", linewidths=0.8,
               zorder=4, label="Incumbent (ObjVal)")

    # ── 3. Bound dots (bottom) ──
    ax.scatter(depths_all, [p.energy_bound / scale for p in points],
               c="white", s=18, edgecolors="#AAAAAA", linewidths=0.6,
               marker="v", zorder=4, label="Lower bound (ObjBound)")

    # ── 4. Pareto front (incumbent) ──
    if front:
        ax.plot([p.depth for p in front], [p.energy / scale for p in front],
                "o-", color="#E74C3C", ms=5, linewidth=1.5,
                zorder=5, label="Pareto front (incumbent)")

    # ── 5. Pareto front (bound, optimistic) ──
    if front_bound and show_bound_front:
        ax.plot([p.depth for p in front_bound], [p.energy_bound / scale for p in front_bound],
                "v--", color="#3498DB", ms=4, linewidth=1.2,
                zorder=5, label="Pareto front (bound)")

    # ── 6. Gap region shade between the two fronts ──
    if front and front_bound and show_bound_front:
        d_inc = np.array([p.depth for p in front])
        e_inc = np.array([p.energy / scale for p in front])
        d_bnd = np.array([p.depth for p in front_bound])
        e_bnd = np.array([p.energy_bound / scale for p in front_bound])

        d_common = np.sort(np.unique(np.concatenate([d_inc, d_bnd])))
        e_inc_interp = np.interp(d_common, d_inc, e_inc)
        e_bnd_interp = np.interp(d_common, d_bnd, e_bnd)

        ax.fill_between(d_common, e_bnd_interp, e_inc_interp,
                        alpha=0.10, color="#9B59B6", zorder=1,
                        label="Optimality gap region")

    ax.set_xlabel("Alignment Depth (m)")
    ax.set_ylabel(f"Energy Consumption ({energy_unit})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for case_id in [10030022, 10030023, 10030024, 10030025, 10030026]:
        cfg = ParetoConfig(n_points=20, time_limit_per_point=1200, mip_gap=0.01)
        results = pareto_sweep(case_id, cfg=cfg, save_dir=f"pareto_convex/{case_id}", Heuristics=0.7, MIPFocus=1)
