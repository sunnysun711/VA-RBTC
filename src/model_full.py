from dataclasses import dataclass, field
import os
from typing import Any, Optional

import gurobipy as gp
from gurobipy import GRB, GurobiError
import numpy as np

from src.prepare import get_case
from src.model_core import get_energy_expr_one_direction, model_EC, model_FC, model_MS, model_SC, model_TC, model_VC, model_va, model_oesd, model_rbtc, solve_edge_va
from src.utils import get_detail_results_decorated_txt, log_info


@dataclass
class Solution:
    va_sol: dict = field(default_factory=dict)
    rbtc_sol: dict = field(default_factory=dict)
    oesd_sol: dict = field(default_factory=dict)
    obj_energy: float = np.nan
    obj_depth: float = np.nan

    @property
    def full_sol(self) -> dict:
        """合并三部分变量为一个完整的 {VarName: value} 字典"""
        return {**self.va_sol, **self.rbtc_sol, **self.oesd_sol}

    @property
    def is_empty(self) -> bool:
        return not self.va_sol and not self.rbtc_sol and not self.oesd_sol

    @staticmethod
    def _extract_var_values(variables: dict) -> dict:
        """
        从 gurobi 变量字典中提取 {VarName: X} 。
        variables 的 value 可能是 gp.Var, 也可能是 dict/list 嵌套了 gp.Var。
        """
        result = {}
        def _collect(obj):
            if isinstance(obj, gp.Var):
                try:
                    result[obj.VarName] = obj.X
                except GurobiError:
                    pass  # 变量没有解值，跳过
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
            elif isinstance(obj, (list, tuple, np.ndarray)):
                for v in obj:
                    _collect(v)
        for v in variables.values():
            _collect(v)
        return result

    @classmethod
    def from_model(cls, model: "FullModel") -> "Solution":
        if model.SolCount == 0:
            return cls()

        va_sol = cls._extract_var_values(model._va_variables)
        rbtc_sol = cls._extract_var_values(model._rbtc_variables)
        oesd_sol = cls._extract_var_values(model._oesd_variables)

        # === get objective values ===
        lower_platform_ele = min(model._case['P1'][1], model._case['P2'][1])
        e_min = min([model._va_variables['e'][i].X for i in range(1, model._case['S_'] + 2)])
        # print(model._va_variables['e_bar'][None].X, e_min)
        obj_depth = lower_platform_ele - e_min  # lowest track elevation relative to platforms
        energy_exp = gp.LinExpr(0)
        if model._case['direction'][0]:
            energy_exp += get_energy_expr_one_direction(model._rbtc_variables, model._oesd_variables, prefix="L2R_", is_oesd_included=model._is_oesd_included)
        if model._case['direction'][1]:
            energy_exp += get_energy_expr_one_direction(model._rbtc_variables, model._oesd_variables, prefix="R2L_", is_oesd_included=model._is_oesd_included)
        obj_energy = energy_exp.getValue()

        return cls(
            va_sol=va_sol,
            rbtc_sol=rbtc_sol,
            oesd_sol=oesd_sol,
            obj_energy=obj_energy,
            obj_depth=obj_depth,
        )
        
    @staticmethod
    def from_file(filepath: str) -> "Solution":
        """
        从 Gurobi 导出的 .json 或 .sol 文件加载解。
        变量会按前缀自动分类到 va_sol / rbtc_sol / oesd_sol。
        """
        import json

        flat: dict[str, float] = {}

        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
            for entry in data.get("Vars", []):
                flat[entry["VarName"]] = float(entry["X"])
        elif filepath.endswith(".sol"):
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        flat[parts[0]] = float(parts[1])
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # 按前缀分类
        _OESD_PREFIXES = ("L2R_varpi", "R2L_varpi", "L2R_kappa", "R2L_kappa", "L2R_xi", "R2L_xi")
        _RBTC_PREFIXES = ("L2R_", "R2L_")

        va_sol, rbtc_sol, oesd_sol = {}, {}, {}
        for name, val in flat.items():
            if name.startswith(_OESD_PREFIXES):
                oesd_sol[name] = val
            elif name.startswith(_RBTC_PREFIXES):
                rbtc_sol[name] = val
            else:
                va_sol[name] = val

        return Solution(va_sol=va_sol, rbtc_sol=rbtc_sol, oesd_sol=oesd_sol)


class FullModel(gp.Model):
    """
    Full model for the VA-RBTC-OESD.
    """
    _ws_info: str = ''
    def __init__(self, case_id: int, save_dir: Optional[str] = None):
        super().__init__()
        na = f"{case_id}"
        if save_dir is None:
            self._name: str = na
        else:
            self._name: str = f"{save_dir}/{na}"
            os.makedirs(f"result/{save_dir}", exist_ok=True)
        self._log_file_path = f"result/{self._name}.log"
        self._case: dict = get_case(case_id)
        self._info: str = self._case["quick_text"]
        self._train = self._case["train"]
        self._oesd = self._case["oesd"]
        self._va_variables: dict[str, gp.tupledict] = {}
        self._rbtc_variables: dict[str, gp.tupledict] = {}
        self._oesd_variables: dict[str, gp.tupledict] = {}
        self._is_oesd_included: bool = True
        self._is_opt_depth: bool = False
        self._depth_cap: float | None = None
        self._energy_cap: float | None = None
        pass
    
    def reconfigure(self, *, is_oesd_included: bool = True, opt_depth: bool = False, depth_cap: float | None = None, energy_cap: float | None = None):
        self._is_oesd_included = is_oesd_included
        self._is_opt_depth = opt_depth
        self._depth_cap = depth_cap
        self._energy_cap = energy_cap
        
        # ==== Energy expression ====
        energy_exp = gp.LinExpr(0)
        if self._case['direction'][0]:
            energy_exp += get_energy_expr_one_direction(self._rbtc_variables, self._oesd_variables, prefix="L2R_", is_oesd_included=is_oesd_included)
        if self._case['direction'][1]:
            energy_exp += get_energy_expr_one_direction(self._rbtc_variables, self._oesd_variables, prefix="R2L_", is_oesd_included=is_oesd_included)
        
        # === Depth expression ===
        lower_platform_ele = min(self._case['P1'][1], self._case['P2'][1])
        e_bar = self._va_variables['e_bar'][None]
        depth_exp = lower_platform_ele - e_bar

        # ===== update cap constraints =====
        for name in ["energy_cap_constr", "depth_cap_constr"]:
            c = self.getConstrByName(name)
            if c is not None:
                self.remove(c)
        if energy_cap is not None:
            self.addConstr(energy_exp <= energy_cap, name="energy_cap_constr")
        if depth_cap is not None:
            self.addConstr(depth_exp <= depth_cap, name="depth_cap_constr")
        
        # ===== set objective =====
        obj_exp = depth_exp if opt_depth else energy_exp
        self.setObjective(obj_exp, GRB.MINIMIZE)
        self.update()
        return
    
    def build(
        self,
        *,
        va_solution: dict | None = None,
        EC_on: bool = True,
        TC_on: bool = True,
        SC_on: bool = True,
        FC_on: bool = True,
        VC_on: bool = True,
        MS_on: bool = True,
        is_oesd_included: bool = True,
        opt_depth: bool = False,
        energy_cap: float | None = None,
        depth_cap: float | None = None,
        cyclic: bool | float | int = False,
        power_time_trapezoid: bool = False,
    ):
        r"""
        Build the optimization model for vertical alignment, regenerative braking train control, and onboard energy storage devices.

        This method sets up the objective function and constraints for the VA-RBTC-OESD optimization model, with options for enabling or disabling
        specific constraint sets, using warm start solutions, and customizing the objective.

        :param dict va_solution: Optional. Predefined solution for vertical alignment variables. If provided, these values are used to initialize the VA variables.
        :param bool EC_on: Optional. If True, enables elevation-based constraints (calls `model_EC`). Default is True.
        :param bool TC_on: Optional. If True, enables time-based constraints (calls `model_TC`). Default is True.
        :param bool SC_on: Optional. If True, enables slope-based constraints (calls `model_SC`). Default is True.
        :param bool FC_on: Optional. If True, enables force-based constraints (calls `model_FC`). Default is True.
        :param bool VC_on: Optional. If True, enables velocity-based constraints (calls `model_VC`). Default is True.
        :param bool MS_on: Optional. If True, enables maximum switches constraints (calls `model_MS`). Default is True.
        :param bool is_oesd_included: Optional. If True, sets the objective to minimize total energy; if False, minimizes net energy. Default is True.
        :param bool opt_depth: Optional. If True, optimizes (MIN) the depth of the VA curve (relative to lower platform elevation). Default is False.
                               If True, either `energy_cap` or `pareto_alpha` must be provided.
                               Enable it with `energy_cap` provided will make the model optimize *only* the depth of the VA curve with at most `energy_cap` energy consumption.
                               Enable it with `pareto_alpha` provided will make the model optimize *both* the depth of the VA curve and energy consumption.
        :param float | None energy_cap: Optional. If provided, sets an upper bound on the total energy consumption. Default is None.
        :param bool | float | int cyclic: Optional. Whether to apply cyclic soe condition, defaults to False.
            if True: varpi_end = varpi_start = varpi_0
            if False: no constraint
            if float | int: varpi_end \in [varpi_0 - cyclic, varpi_0 + cyclic]
        :return: None
        """
        # ===== build core models =====
        self._va_variables.update(model_va(m=self, case=self._case, va_solution=va_solution))
        if self._case['direction'][0]:
            self._rbtc_variables.update(
                model_rbtc(self, case=self._case, e=self._va_variables['e'], l2r=True)
            )
            self._oesd_variables.update(
                model_oesd(
                    self, case=self._case, rbtc_variables=self._rbtc_variables, varpi_0=self._case["varpi_0"], 
                    l2r=True, cyclic=cyclic, power_time_trapezoid=power_time_trapezoid)
            )
        if self._case['direction'][1]:
            self._rbtc_variables.update(model_rbtc(self, case=self._case, e=self._va_variables['e'], l2r=False))
            self._oesd_variables.update(
                model_oesd(
                    self, case=self._case, rbtc_variables=self._rbtc_variables, varpi_0=self._case["varpi_0"], 
                    l2r=False, cyclic=cyclic, power_time_trapezoid=power_time_trapezoid)
            )
        
        # ===== add logic cuts =====
        if EC_on: model_EC(self, case=self._case, va_variables=self._va_variables)
        if SC_on: model_SC(self, case=self._case, va_variables=self._va_variables)
        if TC_on: model_TC(self, case=self._case, rbtc_variables=self._rbtc_variables, directions=self._case['direction'])
        if FC_on: model_FC(self, case=self._case, rbtc_variables=self._rbtc_variables, directions=self._case['direction'])
        if VC_on: model_VC(self, case=self._case, rbtc_variables=self._rbtc_variables, directions=self._case['direction'])
        if MS_on: model_MS(self, case=self._case, rbtc_variables=self._rbtc_variables, directions=self._case['direction'])
        self.update()
        
        # ===== configure obj and cap constraints =====
        self.reconfigure(is_oesd_included=is_oesd_included, opt_depth=opt_depth, depth_cap=depth_cap, energy_cap=energy_cap)
        return
    
    def optimize_(self, save_on=True, **kwargs):
        for paramname, param in kwargs.items():
            self.setParam(paramname, param)
        if save_on:
            # self.write(f"{self._log_file_path[:-4]}.mps")
            # self.write(f"{self._log_file_path[:-4]}.lp")
            self.setParam("LogFile", self._log_file_path)
            # create result directory if not exists
            os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)
            with open(self._log_file_path, "w", encoding="utf-8") as f:
                f.write(self._case["quick_text"])
                # f.write("\n")
                # f.write(self.case["INFO_TEXT"])
                f.write("\n")
                f.write(self._ws_info)
                f.write("\n")
        self.optimize()

        if not save_on:
            return
        if self.Status == GRB.INFEASIBLE:  # model infeasible
            self.setParam("TimeLimit", 5)
            self.computeIIS()
            self.write(f"{self._log_file_path[:-4]}.ilp")
            log_info(f">>> Model is infeasible for case {self._case['case_id']}. Saved conflict constraints file at {self._log_file_path[:-4]}.ilp")
        elif self.Status == GRB.TIME_LIMIT and self.SolCount == 0:  # Time limit reached without finding feasible solution
            print(f"Can't find feasible solutions within {kwargs['TimeLimit']}s for case {self._case['case_id']}.")
        elif self.Status == GRB.OPTIMAL or self.SolCount >= 1:  # TimeLimit reached OR Gap < MIPGap
            self.write(f"{self._log_file_path[:-4]}.json")  # solution file
            self.write(f"{self._log_file_path[:-4]}.sol")
            # append into log files
            with open(self._log_file_path, "a") as f:
                f.write(get_detail_results_decorated_txt(self))
        else:
            raise Exception(f">>>\t\tModel status is {self.Status}!\t\t<<<")
        return
    
    def load_solution_from_file(self, sol_file_path: Optional[str] = None):
        if sol_file_path is None:
            sol_file_path = f"{self._log_file_path[:-4]}.sol"
        try:
            self.read(sol_file_path)
        except (FileNotFoundError, GurobiError):
            print(f"Solution file {sol_file_path} not found!")
        return
    
    def inject_warm_start(self, solution: Solution, binary_only: bool = True, debug_mode: bool = False, use_hint: bool = False):
        if solution.is_empty:
            log_info("Warning: empty solution, skip warm start.")
            return

        sol = solution.full_sol
        n_set = 0
        for var in self.getVars():
            if var.VarName not in sol:
                continue
            val = sol[var.VarName]

            if var.VType == GRB.BINARY:
                val = round(val)
            elif binary_only:
                continue

            if use_hint:
                var.VarHintVal = val
                var.VarHintPri = 2 if var.VType == GRB.BINARY else 1
            else:
                var.Start = val

            if debug_mode:
                var.LB = val
                var.UB = val

            n_set += 1

        mode = "hint" if use_hint else "start"
        self._ws_info = (
            f"Warm {mode} injected: {n_set}/{len(sol)} variables set "
            f"(VA={len(solution.va_sol)}, RBTC={len(solution.rbtc_sol)}, OESD={len(solution.oesd_sol)})"
        )
        log_info(self._ws_info)
        return
    
    def get_sequential_solution(self) -> Solution:
        S_ = self._case['S_']
        lower_platform_ele = min(self._case['P1'][1], self._case['P2'][1])
        names = [
                "gamma", "f", "b", "E_hat", "a", "t", 
                # "w0","wi","wtn",   # optionally fixed, can be derived from v/E_hat
                # "f_max", "b_max", "v", "E_hat_times_2", "Psi_t",   # always skipped, as PWL/POW outputs are too tight sometimes. 
            ]
        
        # ========== Stage 1: VA ==========
        m1 = gp.Model("seq_va")
        m1.setParam(GRB.Param.OutputFlag, 0)
        m1.setParam(GRB.Param.TimeLimit, 2)
        va_vars = model_va(m1, case=self._case)
        model_EC(m1, case=self._case, va_variables=va_vars)
        model_SC(m1, case=self._case, va_variables=va_vars)
        if self._depth_cap is not None:
            m1.addConstr(lower_platform_ele - va_vars['e_bar'][None] <= self._depth_cap)
        m1.setObjective(gp.quicksum(va_vars['e'][s] for s in range(1, S_ + 2)), GRB.MINIMIZE)
        m1.optimize()
        if m1.SolCount == 0:
            log_info("WARNING: Stage 1 (VA) not solved")
            return Solution()
        e_fixed = {s: va_vars['e'][s].X for s in range(1, S_ + 2)}
        va_sol = Solution._extract_var_values(va_vars)
        obj_depth = lower_platform_ele - min(e_fixed.values())

        # ========== Stage 2: RBTC (no OESD) ==========
        m2 = gp.Model("seq_rbtc")
        m2.setParam(GRB.Param.OutputFlag, 0)
        m2.setParam(GRB.Param.MIPGap, 0.99)
        m2.setParam(GRB.Param.TimeLimit, 10)
        rbtc_vars_stage2 = {}
        energy_exp = gp.LinExpr(0)
        if self._case['direction'][0]:
            rv = model_rbtc(m2, case=self._case, e=e_fixed, l2r=True)
            rbtc_vars_stage2.update(rv)
            energy_exp += get_energy_expr_one_direction(rv, {}, prefix="L2R_", is_oesd_included=False)
        if self._case['direction'][1]:
            rv = model_rbtc(m2, case=self._case, e=e_fixed, l2r=False)
            rbtc_vars_stage2.update(rv)
            energy_exp += get_energy_expr_one_direction(rv, {}, prefix="R2L_", is_oesd_included=False)
        model_TC(m2, case=self._case, rbtc_variables=rbtc_vars_stage2, directions=self._case['direction'])
        model_FC(m2, case=self._case, rbtc_variables=rbtc_vars_stage2, directions=self._case['direction'])
        model_VC(m2, case=self._case, rbtc_variables=rbtc_vars_stage2, directions=self._case['direction'])
        model_MS(m2, case=self._case, rbtc_variables=rbtc_vars_stage2, directions=self._case['direction'])
        m2.setObjective(energy_exp, GRB.MINIMIZE)
        m2.optimize()
        if m2.SolCount == 0:
            log_info("WARNING: Stage 2 (RBTC) not solved")
            return Solution()

        # ========== Stage 3: OESD (RBTC fixed) ==========
        m3 = gp.Model("seq_oesd")
        m3.setParam(GRB.Param.OutputFlag, 0)
        m3.setParam(GRB.Param.TimeLimit, 5)
        # 重建 rbtc 变量但固定为 stage2 的值
        rbtc_vars = {}
        oesd_vars = {}
        energy_exp_oesd = gp.LinExpr(0)
        
        if self._case['direction'][0]:
            rv = model_rbtc(m3, case=self._case, e=e_fixed, l2r=True)
            # 固定部分 RBTC 变量
            for na in names:
                for k, v in rv[f"L2R_{na}"].items():
                    v.LB = rbtc_vars_stage2[f"L2R_{na}"][k].X
                    v.UB = rbtc_vars_stage2[f"L2R_{na}"][k].X
            ov = model_oesd(m3, case=self._case, rbtc_variables=rv, varpi_0=self._case['varpi_0'], l2r=True)
            rbtc_vars.update(rv)
            oesd_vars.update(ov)
            energy_exp_oesd += get_energy_expr_one_direction(rv, ov, prefix="L2R_", is_oesd_included=True)
        if self._case['direction'][1]:
            rv = model_rbtc(m3, case=self._case, e=e_fixed, l2r=False)
            # 固定部分 RBTC 变量
            for na in names:
                for k, v in rv[f"R2L_{na}"].items():
                    v.LB = rbtc_vars_stage2[f"R2L_{na}"][k].X
                    v.UB = rbtc_vars_stage2[f"R2L_{na}"][k].X
            ov = model_oesd(m3, case=self._case, rbtc_variables=rv, varpi_0=self._case['varpi_0'], l2r=False)
            rbtc_vars.update(rv)
            oesd_vars.update(ov)
            energy_exp_oesd += get_energy_expr_one_direction(rv, ov, prefix="R2L_", is_oesd_included=True)
        m3.setObjective(energy_exp_oesd, GRB.MINIMIZE)
        m3.optimize()
        if m3.SolCount == 0:
            log_info("WARNING: Stage 3 (OESD) not solved")
            return Solution()
        rbtc_sol = Solution._extract_var_values(rbtc_vars)
        oesd_sol = Solution._extract_var_values(oesd_vars)
        obj_energy = m3.ObjVal

        # ========== Merge ==========
        return Solution(
            va_sol=va_sol,
            rbtc_sol=rbtc_sol,       # 来自 stage2
            oesd_sol=oesd_sol,       # 来自 stage3
            obj_energy=obj_energy,   # 含 OESD 的总能耗
            obj_depth=obj_depth,
        )

if __name__ == "__main__":
    case_id = 10010124
    
    fm1 = FullModel(case_id, save_dir=f"{case_id}/rectangle")
    fm1.build(
        depth_cap=2.0,
        TC_on=True,
        VC_on=True,
        MS_on=True,
        EC_on=True,
        FC_on=False,
        SC_on=False,
        cyclic=True,
        power_time_trapezoid=False,
    )
    sol = fm1.get_sequential_solution()
    # print(sol.obj_depth, sol.obj_energy)
    fm1.inject_warm_start(sol, binary_only=False, debug_mode=False, use_hint=True)
    fm1.optimize_(save_on=True, TimeLimit=1200, Heuristics=0.7, MIPFocus=1, MIPGap=0.01)

    # solution = Solution.from_model(fm1)
    # fm1.reconfigure(depth_cap=6.0)
    # fm1.optimize_(save_on=True, TimeLimit=30)
    
    
    pass