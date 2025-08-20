from dataclasses import dataclass
import time
from typing import Any, Dict, Tuple, List

import gurobipy as gp
from gurobipy import GRB

from src.opt_model import model_EC, model_FC, model_MS, model_SC, model_TC, model_VC, model_oesd, model_rbtc, model_va, solve_edge_va
from src.opt_utils import log_info, log_timing


# =========================
# Global WS tuning knobs
# =========================

# VA (lowest elevation)
WS_VA_MIPGAP: float     = 0.0
WS_VA_TIMELIMIT: int    = 1
WS_VA_OUTPUTFLAG:int    = 0

# SOTC (min time, regen off)
WS_SOTC_MIPGAP: float   = 0.01
WS_SOTC_TIMELIMIT: int  = 1  # for each direction
WS_SOTC_OUTPUTFLAG: int = 0

# EETC (min sum of traction force)
WS_EETC_MIPGAP: float   = 0.05
WS_EETC_TIMELIMIT: int  = 3  # for each direction
WS_EETC_OUTPUTFLAG: int = 0

# OESD maximize stage
WS_OESD_MIPGAP: float   = 0.01
WS_OESD_TIMELIMIT: int  = 1  # for each direction
WS_OESD_OUTPUTFLAG: int = 0

# Tighten-energy stage (single-direction RBTC+OESD, energy objective)
WS_TIGHTEN_MIPGAP: float   = 0.02
WS_TIGHTEN_TIMELIMIT: int  = 5  # for each direction
WS_TIGHTEN_OUTPUTFLAG: int  = 0

# Time-adjustment (cruise speed binary search)
WS_TIMEADJ_THRESH: float = 1.0
WS_TIMEADJ_MAX_IT: int   = 50

@dataclass
class SweepConfig:
    delta_e_step: float = 0.10
    rounds: int = 50
    early_stop_tol: float = 0.0
    stop_on_infeasible: bool = True

# Sweep defaults
DEFAULT_SWEEP_CFG = SweepConfig(
    delta_e_step=0.2,
    rounds=25,
    early_stop_tol=0.0,
    stop_on_infeasible=False,
)

# =========================
# Config & Simple Utilities
# =========================

def cal_objval_from_solution(solution_dict: dict[str, float], obj_net: bool) -> float:
    """Compute warm-start objective on a solution dict (net or gross energy)."""
    sum_phi_n = sum(v for k, v in solution_dict.items() if k.startswith(("L2R_phi_n[", "R2L_phi_n[")))
    if obj_net:
        return sum_phi_n
    sum_phi_b = sum(v for k, v in solution_dict.items() if k.startswith(("L2R_phi_b[", "R2L_phi_b[")))
    sum_xi   = sum(v for k, v in solution_dict.items() if k.startswith(("L2R_xi[",   "R2L_xi[")))
    return sum_phi_n + sum_phi_b - sum_xi


def _enabled_directions(case: dict[str, Any]) -> List[Tuple[str, bool]]:
    """Get direction prefix and l2r tag tuple lists

    :param dict[str, Any] case: case parameters dict
    :raises ValueError: If case['direction'] is (False, False)
    :return List[Tuple[str, bool]]: [("L2R_", True), ("R2L_", False)], if both direction included.
    """
    dirs: List[Tuple[str, bool]] = []
    if case["direction"][0]:
        dirs.append(("L2R_", True))
    if case["direction"][1]:
        dirs.append(("R2L_", False))
    if not dirs:
        raise ValueError("No direction enabled in case['direction']")
    return dirs


def _collect_all_vars(m: gp.Model) -> Dict[str, float]:
    return {v.VarName: v.X for v in m.getVars()}


def _collect_prefixed_vars(m: gp.Model, prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for v in m.getVars():
        if v.VarName.startswith(prefix):
            out[v.VarName] = v.X
    return out


def _set_ws_from_current(m: gp.Model) -> None:
    m.update()
    for v in m.getVars():
        x = getattr(v, "X", None)
        if x is None:
            continue
        if v.VType == GRB.BINARY:
            v.Start = 1.0 if x >= 0.5 else 0.0
        elif v.VType == GRB.INTEGER:
            v.Start = round(x)
        else:
            v.Start = x


def _phi_b_ub_from_case(case: dict[str, Any]) -> float:
    M_f   = case["train"].data["max_force"] * 1000
    ds    = case["ds"]
    S_    = case["S_"]
    v_max = case["train"].data["v_max"]
    T_max = case["T_range"][1]
    _t_ub = T_max - ds * (S_ - 1) / v_max
    mu    = case["train"].data["mu"]
    eta_b = case["train"].data["eta_b"]
    return (M_f * ds + _t_ub * mu) / eta_b


# ===============
# VA solve stages
# ===============

def _solve_va_lowest(case: dict[str, Any], *, debug_mode: bool, force_quiet: bool) -> Tuple[gp.Model, dict[int, float]]:
    log_info("VA generation: Computing lowest elevation curve...", debug_mode=debug_mode, force_quiet=force_quiet)
    t0 = time.time()
    m_va, e_dict = solve_edge_va(case=case, output_flag=0)
    if m_va.Status != GRB.OPTIMAL:
        raise RuntimeError("Failed to generate lower VA curve for warm start solution")
    log_timing("VA generation", time.time() - t0, is_debug=True, debug_mode=debug_mode, force_quiet=force_quiet)
    log_info(f"VA generation: Generated {len(e_dict)} elevation points", is_debug=True, debug_mode=debug_mode, force_quiet=force_quiet)
    return m_va, e_dict


def _lift_va_if_needed(
    case: dict[str, Any],
    e_lowest: dict[int, float],
    *,
    do_lift: bool,
    delta_e_lift: float,
    debug_mode: bool,
    force_quiet: bool,
) -> Tuple[gp.Model | None, dict[int, float]]:
    if (not do_lift) or delta_e_lift <= 0.0:
        return None, e_lowest
    floor = min(e_lowest.values()) + delta_e_lift
    m = gp.Model("temp_model_va_lifted")
    va_vars = model_va(m, case=case, e_floor=floor)
    model_EC(m, case=case, va_variables=va_vars, e_lowest_dict=e_lowest)
    model_SC(m, case=case, va_variables=va_vars)
    e = va_vars["e"]
    m.setObjective(e.sum(), GRB.MINIMIZE)
    m.setParam("OutputFlag", WS_VA_OUTPUTFLAG)
    m.setParam("MIPGap", WS_VA_MIPGAP)
    m.setParam("TimeLimit", WS_VA_TIMELIMIT)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("VA solve failed after applying floor (LB lift)")
    e_lifted = {s: e[s].X for s in e.keys()}
    log_info(f"VA lift: delta={delta_e_lift:.3f} done.", is_debug=True, debug_mode=debug_mode, force_quiet=force_quiet)
    return m, e_lifted


# ===========================
# Per-direction small model(s)
# ===========================

def _solve_model_retry(m: gp.Model, output_flag: int, mip_gap: float, time_limit: float, err_msg:str, max_retries:int=3, extra_time:float=5.0, **kwargs) -> None:
    """Solve optimization model with retries if TimeLimit is reached.

    :param gp.Model m: Gurobi model
    :param int output_flag: Gurobi output flag
    :param float mip_gap: Gurobi MIP gap
    :param float time_limit: Gurobi time limit
    :param str err_msg: error message to raise if failed
    :param int max_retries: maximum number of retries if TimeLimit is hit, defaults to 3
    :param float extra_time: additional seconds added to TimeLimit each retry, defaults to 5.0
    :param kwargs: additional keyword arguments to pass to the default gurobi optimizer
    
    :raise RuntimeError: if failed after max_retries
    """
    m.update()
    m.setParam("OutputFlag", output_flag)
    m.setParam("MIPGap", mip_gap)
    m.setParam("TimeLimit", time_limit)
    for k, v in kwargs.items():
        m.setParam(k, v)
    attempt = 0
    while attempt <= max_retries:
        m.optimize()

        if m.SolCount > 0:
            return

        if m.Status == 9:  # TimeLimit
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"{err_msg}. Tried {max_retries} times with TimeLimit={m.Params.TimeLimit}. (Status={m.Status})")
            new_timelimit = m.Params.TimeLimit + extra_time
            m.setParam("TimeLimit", new_timelimit)
        else:
            if m.Status == GRB.INFEASIBLE:
                # m.setParam("OutputFlag", 1)
                m.computeIIS()
                m.write(f"Model_{m.ModelName}.ilp")
            raise RuntimeError(f"{err_msg}. (Status={m.Status})")


def _solve_sotc_obj_time( m: gp.Model) -> None:
    _solve_model_retry(
        m=m,
        output_flag=WS_SOTC_OUTPUTFLAG,
        mip_gap=WS_SOTC_MIPGAP,
        time_limit=WS_SOTC_TIMELIMIT,
        err_msg="Failed to find RBTC warm start solution (SOTC)",
        max_retries=3,
        extra_time=5.0,
    )

def _solve_eetc_obj_energy(m: gp.Model) -> None:
    m.update()
    m.setParam("OutputFlag", WS_EETC_OUTPUTFLAG)
    m.setParam("MIPGap", WS_EETC_MIPGAP)
    m.setParam("TimeLimit", WS_EETC_TIMELIMIT)
    # m.setParam("FeasibilityTol", 1e-8)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"Failed to assign SOTC solution to EETC. (Status={m.Status})")

    # SOTC gives feasibility; even if not optimal we proceed.

def _solve_oesd_max(m: gp.Model) -> None:
    _solve_model_retry(
        m=m,
        output_flag=WS_OESD_OUTPUTFLAG,
        mip_gap=WS_OESD_MIPGAP,
        time_limit=WS_OESD_TIMELIMIT,
        err_msg="Failed to find OESD max warm start solution",
        max_retries=3,
        extra_time=5.0,
        ### 
        # NumericFocus=2, 
        # ScaleFlag=1,
        # FeasibilityTol=1e-5,
        # IntFeasTol=1e-9,
        
    )


def _cal_travel_time_from_v_vals(v_vals:list[float], ds: float) -> float:
    T = 0
    for v1, v2 in zip(v_vals[:-1], v_vals[1:]):
        T += 2 * ds / (v1+v2)
    return T

def _clamp_v_vals(v_vals:list[float], v_cruise: float) -> list[float]:
    new_vals = []
    for v in v_vals:
        if v < v_cruise:
            new_vals.append(v)
        else:
            new_vals.append(v_cruise)
    return new_vals


def _time_adjustment_one_dir(case: dict[str, Any], m: gp.Model, prefix: str) -> None:
    """Binary search on cruise limit, then re-optimize (single-direction model)."""
    T_max        = case["T_range"][1]
    ds           = case["ds"]
    S_           = case["S_"]
    a_max        = case["train"].data["a_max"]
    time_thresh  = WS_TIMEADJ_THRESH

    T_var = m.getVarByName(f"{prefix}T")
    if T_var is None:
        return
    cur_T = T_var.X
    dt    = T_max - cur_T
    if dt <= time_thresh:
        return

    v_vals = []
    for i in range(S_):
        v = m.getVarByName(f"{prefix}v[{i}]")
        v_vals.append(0.0 if v is None else v.X)

    if not v_vals:
        return
    
    
    ################
    v_cruise = max(v_vals)
    cruise_intervals = sum(1 for v in v_vals if v >= v_cruise - 0.1)
    cruising_distance = cruise_intervals * ds

    lo    = cruising_distance / (T_max - cur_T + (cruising_distance / (v_cruise if v_cruise > 0 else 1.0)))
    hi    = v_cruise
    new_v = hi

    it, max_it = 0, WS_TIMEADJ_MAX_IT
    while it < max_it:
        new_v = 0.5 * (lo + hi)
        # analytic delta-time approximation
        time_old = cur_T
        time_new = _cal_travel_time_from_v_vals(_clamp_v_vals(v_vals, new_v), ds=ds)
        # time_old = cruising_distance / (v_cruise if v_cruise > 0 else 1.0) + 2.0 * (v_cruise - new_v) / a_max
        # time_new = (cruising_distance + (v_cruise**2 - new_v**2) / a_max) / (new_v if new_v > 0 else 1.0)
        delta = time_new - time_old
        if delta > dt:
            lo = new_v
        elif delta < dt - time_thresh:
            hi = new_v
        else:
            break
        it += 1

    # apply UB and re-optimize
    for i in range(S_):
        vv = m.getVarByName(f"{prefix}v[{i}]")
        if vv is not None:
            vv.UB = new_v + 2
    m.setObjective(T_var, GRB.MINIMIZE)
    #########################
    
    
    # m.addConstr(T_var >= T_max - time_thresh, name="Time_adjust")
    # m.setObjective(T_var, GRB.MAXIMIZE)
    
    _solve_sotc_obj_time(m)
    # _set_ws_from_current(m)
    # _solve_eetc_obj_energy(m)


def _pin_with_tol(var, val, rel=1e-6, abs_=1e-4):
    eps = max(abs_, rel * max(1.0, abs(val)))
    var.LB = max(var.LB, val - eps)
    var.UB = min(var.UB, val + eps)


def _fix_speed_force_dir(tc_vars: dict, prefix: str) -> int:
    """Fix RBTC-related vars at current values for OESD stage (single direction), using tupledicts."""
    names = [
        "E_hat", "a", "gamma", "t", "E_s", "R_s", 
        "f", "b", 
        "w0","wi","wtn",   # optionally fixed, can be derived from v/E_hat
        # "f_max", "b_max", "v", "E_hat_times_2", "Psi_t",   # always skipped, as PWL/POW outputs are too tight sometimes. 
    ]
    fixed = 0
    for nm in names:
        key = f"{prefix}{nm}"
        if key not in tc_vars:
            continue
        td = tc_vars[key]                  # gp.tupledict
        for i, var in td.items():
            # _pin_with_tol(var, var.X)
            var.LB=var.X
            var.UB=var.X
            fixed += 1
    return fixed


def _set_starts_from_solution_dict(m: gp.Model, sol: Dict[str, float], prefix: str) -> int:
    """Write starts from a flat solution dict into model variables (prefix-filtered)."""
    if not sol:
        return 0
    cnt = 0
    for v in m.getVars():
        if v.VarName.startswith(prefix):
            val = sol.get(v.VarName, None)
            if val is None:
                continue
            if v.VType == GRB.BINARY:
                v.Start = 1.0 if val >= 0.5 else 0.0
            elif v.VType == GRB.INTEGER:
                v.Start = float(int(round(val)))
            else:
                v.Start = val
            cnt += 1
    return cnt


def _energy_expr_for_direction_tupledict(tc_vars: dict, oesd_vars: dict, prefix: str, obj_net: bool) -> gp.LinExpr:
    """Build energy objective. phi_n.sum() if `obj_net` else sum(phi_n + phi_b - xi)"""
    expr = gp.LinExpr(0)
    expr += tc_vars[f"{prefix}phi_n"].sum()
    if not obj_net:
        expr += tc_vars[f"{prefix}phi_b"].sum()
        expr -= oesd_vars[f"{prefix}xi"].sum()
    return expr


def _solve_one_direction(
    case: dict[str, Any],
    e_fixed: dict[int, float],
    is_l2r: bool,
    prefix: str,
    use_running_time_exhaustive: bool,
    debug_mode: bool,
    force_quiet: bool,
    tighten_energy: bool,
    obj_net: bool,
) -> Dict[str, float]:
    """SOTC -> EETC -> (optional) time adjust -> OESD maximize, then export only this direction vars."""
    m = gp.Model(f"ws_{prefix.rstrip('_').lower()}")

    # fixed VA
    va_vars = model_va(m, case=case, va_solution=e_fixed)
    e = va_vars["e"]

    # SOTC (min time, regen off)
    Tm = case['T_range'][1]
    tc_vars = model_rbtc(m, l2r=is_l2r, nonlinear_t=False, case=case, e=e)
    model_TC(m, case, rbtc_variables=tc_vars, directions=(is_l2r, not is_l2r))
    model_FC(m, case, rbtc_variables=tc_vars, directions=(is_l2r, not is_l2r))
    model_VC(m, case, rbtc_variables=tc_vars, directions=(is_l2r, not is_l2r))
    model_MS(m, case, rbtc_variables=tc_vars, directions=(is_l2r, not is_l2r))
    T: gp.Var = tc_vars[f"{prefix}T"]  # type: ignore
    phi_b: gp.tupledict = tc_vars[f"{prefix}phi_b"]  # type: ignore
    for i in range(1, case["S_"] + 1):
        phi_b[i].UB = 0.0
    m.setObjective(T, GRB.MINIMIZE)
    t0 = time.time()
    _solve_sotc_obj_time(m)  # always solve, raise RuntimeError if no solution
    T_value = getattr(T, "X", None)
    log_timing(f"SOTC {prefix.rstrip('_')} with {m.SolCount} solutions (T={T_value:.1f}/{Tm}=Tm)", time.time() - t0, is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)

    # EETC (min f.sum, WS from SOTC)
    traction_sum = tc_vars[f"{prefix}f"].sum()  # type: ignore
    _set_ws_from_current(m)
    m.setObjective(traction_sum, GRB.MINIMIZE)
    t1 = time.time()
    _solve_eetc_obj_energy(m)
    T_value = getattr(T, "X", None)
    log_timing(f"EETC {prefix.rstrip('_')} with {m.SolCount} solutions (T={T_value:.1f}/{Tm}=Tm)", time.time() - t1, is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)

    # time adjustment (single dir)
    if use_running_time_exhaustive:
        t2 = time.time()
        _time_adjustment_one_dir(case, m, prefix)
        T_value = getattr(T, "X", None)
        log_timing(f"Time adjust {prefix.rstrip('_')} with {m.SolCount} solutions (T={T_value:.1f}/{Tm}=Tm)", time.time() - t2, is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)

    out_dir = _collect_prefixed_vars(m, prefix)  # collect vars

    # OESD maximize (phi_b + xi) with RBTC fixed
    _fix_speed_force_dir(tc_vars, prefix)
    phi_b_ub = _phi_b_ub_from_case(case)
    for i in range(1, case["S_"] + 1):
        phi_b[i].UB = phi_b_ub
        phi_b[i].LB = 0.0

    # add OESD and maximize
    oesd_vars = model_oesd(m=m, varpi_0=case["varpi_0"], l2r=is_l2r, case=case, rbtc_variables=tc_vars)
    obj = gp.LinExpr(0)
    obj += tc_vars[f"{prefix}phi_b"].sum()  # type: ignore
    obj += oesd_vars[f"{prefix}xi"].sum()

    m.setObjective(obj, GRB.MAXIMIZE)
    t3 = time.time()
    _solve_oesd_max(m=m)  # always solve, raise RuntimeError if no solution
    log_timing(f"OESD maximize {prefix.rstrip('_')} with {m.SolCount} solutions", time.time() - t3, is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)
    out_dir = _collect_prefixed_vars(m, prefix)

    if tighten_energy:
        log_info(f"Build tighten energy model for {prefix.rstrip('_')}", is_debug=True, debug_mode=debug_mode, force_quiet=force_quiet)

        m_tight = gp.Model(f"ws_tight_{prefix.rstrip('_').lower()}")

        # fixed VA
        va_vars2 = model_va(m_tight, case=case)  # this returns all VA variables dict
        e2 = va_vars2["e"]
        m_tight.addConstrs((e2[s] == e_fixed[s] for s in range(1, case["S_"] + 2)), name="va_solution:e")
        model_EC(m_tight, case=case, va_variables=va_vars2, e_lowest_dict=None)
        model_SC(m_tight, case=case, va_variables=va_vars2)

        # RBTC + OESD
        tc_vars2 = model_rbtc(m_tight, l2r=is_l2r, nonlinear_t=False, case=case, e=e2)
        model_TC(m_tight, case, rbtc_variables=tc_vars2, directions=(is_l2r, not is_l2r))
        model_FC(m_tight, case, rbtc_variables=tc_vars2, directions=(is_l2r, not is_l2r))
        model_VC(m_tight, case, rbtc_variables=tc_vars2, directions=(is_l2r, not is_l2r))
        model_MS(m_tight, case, rbtc_variables=tc_vars2, directions=(is_l2r, not is_l2r))
        oesd_vars2 = model_oesd(m=m_tight, varpi_0=case["varpi_0"], l2r=is_l2r, case=case, rbtc_variables=tc_vars2)
        energy_obj = _energy_expr_for_direction_tupledict(tc_vars2, oesd_vars2, prefix, obj_net=obj_net)
        m_tight.setObjective(energy_obj, GRB.MINIMIZE)
        m_tight.update()

        # warm start
        ws_variable_cnt   = _set_starts_from_solution_dict(m_tight, out_dir, prefix)  # only those with prefix
        for s, val in e_fixed.items():  # also set start for VA variables
            e2[s].Start = val

        # opt param setting
        m_tight.setParam("OutputFlag", WS_TIGHTEN_OUTPUTFLAG)
        m_tight.setParam("MIPGap", WS_TIGHTEN_MIPGAP)
        m_tight.setParam("TimeLimit", WS_TIGHTEN_TIMELIMIT)
        m_tight.Params.Heuristics = 0.8  # 0~1, default 0.05
        m_tight.Params.RINS = 5  # default 0
        # m_tight.Params.MIPFocus = 1  # 1=solutions, 2=optimality, 3=balance
        # m_tight.Params.FeasibilityTol = 1e-5
        # m_tight.Params.NumericFocus = 0

        t4 = time.time()
        m_tight.optimize()
        
        # --- Record gap & objective into the returned dict ---
        gap_key = f"__GAP__{prefix.rstrip('_')}"
        obj_key = f"__OBJ__{prefix.rstrip('_')}"
        sol_cnt_key = f"__SOL_CNT__{prefix.rstrip('_')}"
        out_dir[sol_cnt_key] = m_tight.SolCount
        out_dir[gap_key] = float(m_tight.MIPGap) if m_tight.SolCount >= 1 else float('nan')
        out_dir[obj_key] = float(m_tight.ObjVal) if m_tight.SolCount >= 1 else float('nan')

        if m_tight.SolCount >= 1:
            out_dir.update(_collect_prefixed_vars(m_tight, prefix))
            log_timing(f"Tighten energy {prefix.rstrip('_')} with {m_tight.SolCount} solutions", 
                       time.time() - t4, 
                       is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)
        else:
            log_timing(f"Tighten energy {prefix.rstrip('_')} failed with status {m_tight.Status}. Fall back to previous solution.", 
                       time.time()-t4, 
                       is_debug=False, debug_mode=debug_mode, force_quiet=force_quiet)

        m_tight.dispose()
    
    m.dispose()
    return out_dir


# =======================
# Public sweep & warmstart
# =======================

def run_va_floor_sweep(
    case: dict[str, Any],
    *, 
    obj_net: bool,
    cfg: SweepConfig = DEFAULT_SWEEP_CFG,
    debug_mode: bool = False,
) -> tuple[dict[str, float], float, float, list[tuple[dict[str, float], float, float]], str]:
    """Runs a delta-e sweep optimization to find the best solution by iteratively calling gen_warm_start_sol.

    This function performs multiple rounds of optimization with increasing delta-e lift values, tracking
    the best solution based on the specified objective function. It supports early stopping and collects
    optimization history for analysis.

    :param dict[str, Any] case: Configuration dictionary containing problem parameters and constraints
    :param bool obj_net: If True, calculates objective value using net formulation; if False, uses gross formulation
    :param SweepConfig cfg: Sweep configuration object with parameters controlling the optimization rounds
                            (rounds, delta_e_step, early_stop_tol, stop_on_infeasible)
    :param bool debug_mode: If True, enables additional debugging output during optimization runs
    :return: tuple containing:
    
             - Best solution found (dictionary of variable values)
             - Best metric value achieved
             - Delta-e value that produced the best solution
             - History list of tuples (solution, delta, metric) for each round
             - Formatted string with sweep information and results summary
    :rtype: tuple[dict[str, float], float, float, list[tuple[dict[str, float], float, float]], str]
    """
    best_sol: dict[str, float] | None = None
    best_metric = float("inf")
    best_delta = 0.0
    history: list[tuple[dict[str, float], float, float]] = []

    sweep_info_txt = ""

    sweep_info_txt += log_info(f"======== [SWEEP] Running floor sweep with {cfg} =========\n")

    for r in range(0, cfg.rounds + 1):
        t0 = time.time()
        delta = r * cfg.delta_e_step
        try:
            sol_r = gen_warm_start_sol(
                case=case,
                use_running_time_exhaustive=True,
                debug_mode=debug_mode,
                lift_va=(r > 0),
                delta_e_lift=delta,
                force_quiet=not debug_mode,
            )
            metric_r = cal_objval_from_solution(sol_r, obj_net=obj_net)
            
            # --- Extract recorded per-direction gaps if present ---
            tight_model_info = []
            for prefix, _ in _enabled_directions(case):
                k = f"__GAP__{prefix.rstrip('_')}"
                if k in sol_r:
                    gap_val = float(sol_r[k])
                    sol_cnt = sol_r[f"__SOL_CNT__{prefix.rstrip('_')}"]
                    tight_model_info.append(f"{prefix.rstrip('_')} sol={sol_cnt:02} gap={gap_val:.4f}")
            gap_txt = (" | " + " ; ".join(tight_model_info)) if tight_model_info else ""
            
            history.append((sol_r, delta, metric_r))
            took = time.time() - t0

            if metric_r + 1e-12 < best_metric:
                best_metric = metric_r
                best_sol = sol_r
                best_delta = delta

            sweep_info_txt += log_info(f"[SWEEP] r={r:02d} delta={delta:.2f}  metric={metric_r:.1f}  best={best_metric:.1f}  time={took:.3f}s{gap_txt}") + "\n"

            if cfg.early_stop_tol > 0 and (metric_r - best_metric) > cfg.early_stop_tol:
                sweep_info_txt += log_info("[SWEEP] early stop: metric worsened beyond tolerance.") + "\n"
                break

        except RuntimeError as e:
            took = time.time() - t0
            sweep_info_txt += log_info(f"[SWEEP] r={r:02d} delta={delta:.2f}  FAILED: {e}  time={took:.3f}s") + "\n"
            if cfg.stop_on_infeasible:
                break
            else:
                continue

    if best_sol is None:
        raise RuntimeError("No feasible warm start found in sweep.")
    return best_sol, best_metric, best_delta, history, sweep_info_txt


def gen_warm_start_sol(
    case: dict[str, Any],
    use_running_time_exhaustive: bool = True,
    debug_mode: bool = False,
    lift_va: bool = False,
    delta_e_lift: float = 0.5,
    force_quiet: bool = False,
    tighten_energy:bool=True,
    obj_net:bool=True,
) -> dict[str, float]:
    """
    Final warm-start orchestrator:
    1) VA lowest (+ optional lift)
    2) Per-direction independent models (SOTC->EETC->time adjust->OESD->RBTC+OESD opt)
    3) Merge VA vars + per-direction vars into a single solution dict
    
    :param dict[str, Any] case: Configuration dictionary containing problem parameters and constraints
    :param bool use_running_time_exhaustive: If True, uses exhaustive search for running time.
    :param bool debug_mode: If True, enables additional debugging output during optimization runs
    :param bool lift_va: If True, lifts the VA model by delta_e_lift
    :param float delta_e_lift: The lift value to apply to the VA model if lift_va is True
    :param bool force_quiet: If True, suppresses all output
    :param bool tighten_energy: If True, tightens the energy constraints
    :param bool obj_net: If True, calculates objective value using net formulation; if False, uses gross formulation
    :return: Dictionary of variable values for the warm-start solution
    :rtype: dict[str, float]
    """
    log_info("Warm start solution generation started...", debug_mode=debug_mode, force_quiet=force_quiet)
    t_total = time.time()

    # Step 1: VA lowest
    temp_mod_va, e_dict = _solve_va_lowest(case, debug_mode=debug_mode, force_quiet=force_quiet)

    # Step 1b: optional lift
    m_lift, e_dict = _lift_va_if_needed(
        case, e_dict, do_lift=lift_va, delta_e_lift=delta_e_lift, debug_mode=debug_mode, force_quiet=force_quiet
    )

    # Collect VA vars as the base solution
    solution_dict = _collect_all_vars(temp_mod_va if m_lift is None else m_lift)

    # Step 2: per-direction models (completely separable given fixed VA)
    for prefix, is_l2r in _enabled_directions(case):
        sol_dir = _solve_one_direction(
            case=case,
            e_fixed=e_dict,
            is_l2r=is_l2r,
            prefix=prefix,
            use_running_time_exhaustive=use_running_time_exhaustive,
            debug_mode=debug_mode,
            force_quiet=force_quiet,
            tighten_energy=tighten_energy,
            obj_net=obj_net,
        )
        solution_dict.update(sol_dir)

    # cleanup VA models
    temp_mod_va.dispose()
    if m_lift is not None:
        m_lift.dispose()

    log_timing("Warm start generation", time.time() - t_total, debug_mode=debug_mode, force_quiet=force_quiet)
    return solution_dict