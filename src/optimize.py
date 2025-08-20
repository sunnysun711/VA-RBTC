from doctest import debug
import os
import time
from typing import Any

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from src.prepare import get_case
from src.opt_utils import get_detail_results_decorated_txt, log_info, log_timing, parse_var_name
from src.opt_model import model_EC, model_FC, model_MS, model_SC, model_TC, model_VC, model_va, model_oesd, model_rbtc, solve_edge_va
from src.opt_ws import SweepConfig, _energy_expr_for_direction_tupledict, gen_warm_start_sol, run_va_floor_sweep, DEFAULT_SWEEP_CFG


class VA_RBTC_OESD(gp.Model):
    """
    Optimization model for Vertical Alignment, Regenerative Braking Train Control, and Onboard Energy Storage Devices (VA-RBTC-OESD).

    This class encapsulates the Gurobi model for optimizing the vertical alignment of a railway section, integrating regenerative
    braking train control and onboard energy storage devices.

    :param int case_id: The ID of the selected case. Determines the scenario and data used for optimization.
    :param str | None save_dir: Optional. The parent directory for saving results. If None, results are saved in the default "result" folder.

    :ivar str _name: The name of the case, including the save directory if provided.
    :ivar str _log_file_path: The file path for the log file.
    :ivar dict _case: The case details obtained from the `get_case` function.
    :ivar str _info: Quick description text for the case.
    :ivar dict _train: Details of the train configuration for the case.
    :ivar dict _oesd: Details of the onboard energy storage devices configuration for the case.
    :ivar dict _va_variables: Dictionary to store vertical alignment variables.
    :ivar dict _rbtc_variables: Dictionary to store regenerative braking train control variables.
    :ivar dict _oesd_variables: Dictionary to store onboard energy storage devices variables.
    """
    _cfg: SweepConfig = DEFAULT_SWEEP_CFG
    _ws_info: str = ''

    def __init__(self, case_id: int, save_dir: str | None = None):
        """

        :param case_id:
        :param save_dir: only the parent directory of the saved model.
        """
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
        self._va_variables: dict = {}
        self._rbtc_variables: dict = {}
        self._oesd_variables: dict = {}
        print(self._info)
        # print(self._case["INFO_TEXT"])
    
    @property
    def case(self):
        return self._case

    @property
    def train(self):
        return self._train

    @property
    def oesd(self):
        return self._oesd

    @property
    def va_variables(self):
        return self._va_variables

    @property
    def rbtc_variables(self):
        return self._rbtc_variables

    @property
    def oesd_variables(self):
        return self._oesd_variables 
    
    
    def _warm_start_pipeline(
        self,
        *,
        use_warm_start: bool,
        lift_va: bool | float | int,
        obj_net: bool,
        debug_mode: bool,
        energy_cap: float | None = None,
    ) -> str:
        """
        Compute and apply warm starts.
        Unified behavior:
          - If energy_cap is None, treat as +Inf for filtering.
          - Sweep mode (lift_va is True): run floor sweep, then filter by energy_cap.
          - Direct modes (lift_va is False or a number): single-shot warm start.

        Side effects:
          - Sets self.NumStart and StartNumber when multiple starts are available.
          - Calls set_warm_start_sol(self, ...) to apply starts.
          
        :param bool use_warm_start: Whether to use warm start. If False, this method does nothing and return a string "No warm start."
        :param bool | float | int lift_va: Whether to or how much to lift the vertical alignment.
        :param bool obj_net: Whether to use the net power only as objective. If False, use both net and OESD power.
        :param bool debug_mode: Whether to use debug mode.
        :param float | None energy_cap: The maximal energy consumption accepted for the solutions. Should be derived from cases with no OESDs. 
        
        :raise ValueError: If energy_cap is None and obj_net is False.
        :return: A string indicating the warm start status.
        :rtype: str
        """
        if not use_warm_start:
            return "No warm start."

        cap = np.inf if energy_cap is None else float(energy_cap)

        # Sweep mode
        if isinstance(lift_va, bool) and lift_va:
            best_solution, best_obj, best_delta, hist, sweep_info = run_va_floor_sweep(
                case=self.case,
                obj_net=obj_net,
                cfg=self._cfg,
                debug_mode=debug_mode,
            )
            # Filter by energy cap (no-op if cap = +Inf)
            hist_filtered = [h for h in hist if h[2] <= cap]

            if not hist_filtered:
                # TODO: Fallback: try to optimize further with current best solution, or ones that are very close to the cap, see if they can be further opted to provide "energy <= cap" solution
                log_info(f"[SWEEP] No valid solution found to be better than cap ({cap:.1f}), but found best solution with objVal {best_obj:.1f}.")
                set_warm_start_sol(self, solution_dict=best_solution, debug_mode=debug_mode)
                # If really all fails, then return this.
                return sweep_info + "\nNo valid solution found in sweep mode."

            self.NumStart = len(hist_filtered)
            self.update()
            for i, (sol_i, delta_i, obj_i) in enumerate(hist_filtered):
                self.Params.StartNumber = i
                self.update()
                set_warm_start_sol(self, solution_dict=sol_i, debug_mode=debug_mode)
            sweep_info += log_info(f">>> best delta={best_delta:.3f}, objVal={best_obj:.6f}")

            return sweep_info

        # Direct modes: lowest VA or fixed lift value
        if isinstance(lift_va, (float, int)):
            ws = gen_warm_start_sol(
                case=self.case,
                use_running_time_exhaustive=True,
                debug_mode=debug_mode,
                lift_va=True,
                delta_e_lift=float(lift_va),
                obj_net=obj_net,
                force_quiet=False,
                tighten_energy=True,
            )
            set_warm_start_sol(self, solution_dict=ws, debug_mode=debug_mode)
            return "Warm start solution generated by lifted va of {}.".format(lift_va)

        if isinstance(lift_va, bool) and not lift_va:
            ws = gen_warm_start_sol(
                case=self.case,
                use_running_time_exhaustive=True,
                debug_mode=debug_mode,
                obj_net=obj_net,
                tighten_energy=True,
            )
            set_warm_start_sol(self, solution_dict=ws, debug_mode=debug_mode)
            return "Warm start solution generated by lowest va."

        raise TypeError("lift_va should be either a bool or a float/int")

    
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
        obj_net: bool = False,
        with_nonlinear_t: bool = False,
        use_warm_start: bool = True,
        lift_va: bool | float | int = True,
        debug_mode: bool = False,
        opt_depth: bool = False,
        energy_cap: float | None = None,
        pareto_alpha: float | None = None,
        
    ):
        """
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
        :param bool obj_net: Optional. If True, sets the objective to minimize net energy; if False, minimizes total energy. Default is False.
        :param bool with_nonlinear_t: Optional. If True, includes nonlinear terms in the regenerative braking train control model. Default is False.
        :param bool use_warm_start: Optional. If True, uses a warm start solution for initialization. Default is True.
        :param bool | float | int lift_va: Optional. If True, iteratively lifts the VA curve before setting the warm start solution for better initialization.
                                           If a number is provided, directly uses that value as the lifting parameter. Default is True.
        :param bool debug_mode: Optional. If True, enables debug mode, which sets hard constraints with warm start solutions to help generate .ilp files for debugging violated constraints. Default is False.
        :param bool opt_depth: Optional. If True, optimizes (MIN) the depth of the VA curve (relative to lower platform elevation). Default is False.
                               If True, either `energy_cap` or `pareto_alpha` must be provided.
                               Enable it with `energy_cap` provided will make the model optimize *only* the depth of the VA curve with at most `energy_cap` energy consumption.
                               Enable it with `pareto_alpha` provided will make the model optimize *both* the depth of the VA curve and energy consumption.
        :param float | None energy_cap: Optional. If provided, sets an upper bound on the total energy consumption. Default is None.
        :param float | None pareto_alpha: Optional. If provided, sets the weight for the energy consumption term in the objective function. Default is None.

        :return: None
        """
        
        # ===== build core models =====
        model_va(m=self, va_solution=va_solution)
        if self.case['direction'][0]:
            model_rbtc(self, l2r=True, nonlinear_t=with_nonlinear_t)
            model_oesd(self, varpi_0=self.case["varpi_0"], l2r=True)
        if self.case['direction'][1]:
            model_rbtc(self, l2r=False, nonlinear_t=with_nonlinear_t)
            model_oesd(self, varpi_0=self.case["varpi_0"], l2r=False)
        
        # ===== add logic cuts =====
        if EC_on:
            model_EC(self)
        
        if TC_on:
            model_TC(self)
        
        if SC_on:
            model_SC(self)

        if FC_on:
            model_FC(self)
        
        if VC_on:
            model_VC(self)
            
        if MS_on:
            model_MS(self)
        
        # ==== Energy expression ====
        energy_exp = gp.LinExpr(0)
        # net energy (must be optimized)
        if self.case['direction'][0]:
            energy_exp += _energy_expr_for_direction_tupledict(self.rbtc_variables, self.oesd_variables, prefix="L2R_", obj_net=obj_net)
        if self.case['direction'][1]:
            energy_exp += _energy_expr_for_direction_tupledict(self.rbtc_variables, self.oesd_variables, prefix="R2L_", obj_net=obj_net)
        
        if not opt_depth:  # only optimize energy
            
            # ===== set objective =====
            self.setObjective(energy_exp, GRB.MINIMIZE)
            
            # ===== warm-start =====
            self._ws_info = self._warm_start_pipeline(
                use_warm_start=use_warm_start,
                lift_va=lift_va,
                obj_net=obj_net,
                debug_mode=debug_mode,
                energy_cap=None,  # no cap in energy-only mode
            )

            return
        
        # optimize depth
        lower_platform_ele = min(self.case['P1'][1], self.case['P2'][1])
        e_min = self.addVar(lb=0, ub=lower_platform_ele, vtype=GRB.CONTINUOUS, name="e_min")
        self.addConstrs((self.va_variables['e'][i] >= e_min for i in self.va_variables['e'].keys()), name="e_min_constr")
        depth = lower_platform_ele - e_min  # lowest track elevation relative to platforms
        
        if pareto_alpha is None and energy_cap is not None:  # optimize *only* depth
            self.addConstr(energy_exp <= energy_cap, name="energy_cap_constr")  # consume at most energy_cap 
            self.setObjective(depth, GRB.MINIMIZE)
            
            # ===== warm-start =====
            self._ws_info = self._warm_start_pipeline(
                use_warm_start=use_warm_start,
                lift_va=lift_va,
                obj_net=obj_net,
                debug_mode=debug_mode,
                energy_cap=energy_cap,
            )

            return
        
        if pareto_alpha is not None and energy_cap is not None:  # optimize both
            assert 0 <= pareto_alpha <= 1, "pareto_alpha should be in [0, 1]"
            
            _tmp_mod, e_lowest_dict = solve_edge_va(self.case)
            e_lowest = min(e_lowest_dict.values())
            depth_max = lower_platform_ele - e_lowest
            
            self.setObjectiveN(energy_exp/energy_cap, index=0, weight=pareto_alpha)
            self.setObjectiveN(depth/depth_max, index=1, weight=1-pareto_alpha)
            
            self._ws_info = self._warm_start_pipeline(
                use_warm_start=use_warm_start,
                lift_va=lift_va,
                obj_net=obj_net,
                debug_mode=debug_mode,
                energy_cap=energy_cap,  # None -> +Inf, so filtering is unified
            )

            
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
                f.write(self.case["quick_text"])
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
            log_info(f">>> Model is infeasible for case {self._case['case_id']}. Saved conflict constraints file at {self._log_file_path[:-1]}.ilp")
        elif (
            self.Status == GRB.TIME_LIMIT and self.SolCount == 0
        ):  # Time limit reached without finding feasible solution
            print(
                f"\033[91m"
                f"Can't find feasible solutions within {kwargs['TimeLimit']}s for case {self._case['case_id']}."
                f"\033[0m"
            )
        elif (
            self.Status == GRB.OPTIMAL or self.SolCount >= 1
        ):  # TimeLimit reached OR Gap < MIPGap
            self.write(f"{self._log_file_path[:-4]}.json")  # solution file
            # self.write(f"{self._log_file_path[:-4]}.sol")
            # append into log files
            with open(self._log_file_path, "a") as f:
                f.write(get_detail_results_decorated_txt(self))
        else:
            raise Exception(f">>>\t\tModel status is {self.Status}!\t\t<<<")
        return




def set_warm_start_sol(
    mod: VA_RBTC_OESD,
    solution_dict: dict[str, Any],
    debug_mode: bool = False,
):
    """Apply warm start solution to the optimization model.

    Steps:
    1. **Parse Variables**: Extract variable names and indices from `solution_dict`.
    2. **Validate Model**: Ensure model is properly built with all variable dictionaries
    3. **Apply Values**: Set the `start` attribute for each model variable
    4. **Debug Ranges** (optional): Add absolute bounds equal to warm start values

    :param VA_RBTC_OESD mod: The built optimization model containing VA (Vertical Alignment),
                             RBTC (Regenerative-Braking Train Control), and OESD (energy storage) variables
    :param dict[str, Any] solution_dict: The solution to be assigned to `mod`, which is 
                                         typically generated by `gen_warm_start_sol()`.
    :param bool, optional debug_mode: Enable debug logging and add absolute ranges to
                                      fix variables at warm start values (to examine IIS and generate .ilp file).
                                      Defaults to False.

    :raise RuntimeError: If the model is not built (missing variable dictionaries)

    Note:

    - Unrecognized variable names are logged but do not cause failures

    """
    # log_info( "Solution application: Setting warm start values...", debug_mode=debug_mode )

    # ===== Parse variable names and indices =====
    parsed_solution = {}
    for var_name_str, var_value in solution_dict.items():
        var_name, var_index = parse_var_name(var_name_str)
        parsed_solution[var_name_str] = (var_name, var_index, var_value)

    # ===== Validate model state =====
    if not hasattr(mod, "_va_variables") or not mod._va_variables:
        raise RuntimeError(
            "Model must be built (call mod.build()) before setting warm start solution"
        )

    def apply_variable_start(
        var_dict, var_name: str, var_index, var_value: float
    ) -> bool:
        """
        Helper function to apply warm start value to a variable

        :param var_dict: Variable dictionary or single variable
        :param var_name: Variable name for debugging
        :param var_index: Variable index (if applicable)
        :param var_value: Warm start value
        :return: True if successful, False otherwise
        """
        eps = 0  # 1e-6
        try:
            if (
                isinstance(var_dict, dict)
                and var_index is not None
                and var_index in var_dict
            ):
                var_dict[var_index].start = var_value
                if debug_mode:
                    # mod.addRange(var_dict[var_index], var_value-eps, var_value+eps)
                    mod.addConstr(var_dict[var_index] == var_value, name=f"DEBUG_{var_name}[{var_index}]")
                return True
            elif not isinstance(var_dict, dict):
                var_dict.start = var_value
                if debug_mode:
                    # mod.addRange(var_dict, var_value-eps, var_value+eps)
                    mod.addConstr(var_dict == var_value, name=f"DEBUG_{var_name}")
                return True
            return False
        except Exception as e:
            log_info(
                f"Solution application: Failed to set {var_name}: {str(e)}",
                is_debug=True,
                debug_mode=debug_mode,
            )
            return False

    # ===== Apply warm start values =====
    va_vars_set = 0
    tc_vars_set = 0
    sd_vars_set = 0
    debug_ranges_added = 0

    for var_name_str, (var_name, var_index, var_value) in parsed_solution.items():
        success = False

        if var_name in mod.va_variables:
            success = apply_variable_start(
                mod.va_variables[var_name], var_name, var_index, var_value
            )
            if success:
                va_vars_set += 1
        elif var_name in mod.rbtc_variables:
            if var_name not in ["f_max", "b_max"]:
                success = apply_variable_start(
                    mod.rbtc_variables[var_name], var_name, var_index, var_value
                )
                if success:
                    tc_vars_set += 1
        elif var_name in mod.oesd_variables:
            success = apply_variable_start(
                mod.oesd_variables[var_name], var_name, var_index, var_value
            )
            if success:
                sd_vars_set += 1
        elif var_name.startswith("__") or "_sw_" in var_name:
            pass
        else:
            log_info(
                f"Solution application: Setting unknown variable {var_name} with value {var_value} failed!",
                is_debug=True,
                debug_mode=debug_mode,
            )

        if success and debug_mode:
            debug_ranges_added += 1

    # ===== Summary =====
    total_vars_set = va_vars_set + tc_vars_set + sd_vars_set

    log_info(f"Warm start solution applied successfully: VA={va_vars_set}, RBTC={tc_vars_set}, OESD={sd_vars_set}, Total={total_vars_set}", debug_mode=debug_mode)
    log_info( f"  - Debug ranges: {debug_ranges_added}", is_debug=True, debug_mode=debug_mode )
    # log_info("Warm start setup completed", debug_mode=debug_mode)
    
    return


def main():
    # md = VA_RBTC_OESD(case_id=10030222, save_dir="testing")
    md = VA_RBTC_OESD(case_id=10030222, save_dir="testing1")
    # md.build(
    #     EC_on=True,
    #     TC_on=True,
    #     SC_on=True,
    #     obj_net=True,
    #     with_nonlinear_t=False,
    #     use_warm_start=True,
    #     lift_va=6.0,
    #     debug_mode=True,
    #     opt_depth=False,
    # )
    md.build(
        with_nonlinear_t=False,
        EC_on=True, TC_on=True, SC_on=True, FC_on=True, obj_net=False,
        use_warm_start=True,
        # lift_va=True,
        lift_va = 4.678,
        debug_mode=True,
        opt_depth=False, energy_cap=1.480957318671e+08, pareto_alpha=None,
    )
    md.optimize_(save_on=True, MIPGap=0, TimeLimit=20, NumericFocus=0, FeasibilityTol=1e-4)
    return


if __name__ == "__main__":
    main()
