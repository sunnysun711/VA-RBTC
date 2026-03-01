import re
import gurobipy as gp


def get_detail_results_decorated_txt(model: gp.Model) -> str:
    """Get detailed results from the Gurobi model, including variable values and constraints equations.

    :param gp.Model model: Gurobi model
    :return str: Detailed results
    """
    txt = ""
    txt += ">>" * 20 + f" DETAIL RESULT " + "<<" * 20 + "\n"
    txt += ">>" * 10 + f" VARIABLES " + "<<" * 10 + "\n"
    for var in model.getVars():
        txt += (
            f"{var.VarName}, VType: {var.VType}, LB: {var.LB}, UB: "
            f"{var.UB}, ObjCoefficient: {var.Obj}, value: {var.X}\n"
        )
    txt += ">>" * 10 + f" CONSTRAINTS " + "<<" * 10 + "\n"
    for constr in model.getConstrs():
        expr = model.getRow(constr)
        lhs_value = 0

        for i in range(expr.size()):
            var = expr.getVar(i)
            coeff = expr.getCoeff(i)
            lhs_value += var.X * coeff

        txt += (
            f"{constr.ConstrName} with SLACK {constr.Slack:.4f}: "
            f"{model.getRow(constr)} = {lhs_value} ||{constr.Sense}=|| {constr.rhs}\n"  # type: ignore
        )
    txt += ">>" * 20 + f" DETAIL RESULT " + "<<" * 20 + "\n"
    return txt


# ===== Helper functions for parsing variables and logging =====
def parse_var_name(var_name_str):
    """
    parse variable names:
        'e[1]' -> ('e', 1)
        'L2R_phi_n[5]' -> ('L2R_phi_n', 5)
    """
    match = re.match(r"([^[]+)\[(\d+)\]", var_name_str)
    if match:
        var_name = match.group(1)
        var_index = int(match.group(2))
        return var_name, var_index
    else:
        return var_name_str, None


def log_info(message: str, is_debug: bool = False, debug_mode: bool = False, force_quiet: bool = False) -> str:
    """Log information message"""
    info_txt = ""
    if force_quiet:
        return info_txt
    if is_debug and debug_mode:
        info_txt = f"[DEBUG] {message}"
        print(info_txt)
    elif not is_debug:
        info_txt = f"[INFO] {message}"
        print(info_txt)
    return info_txt


def log_timing(
    operation: str, duration: float, is_debug: bool = False, debug_mode: bool = False, force_quiet: bool = False
) -> str:
    """Log timing information"""
    info_txt = ""
    if force_quiet:
        return info_txt
    if is_debug and debug_mode:
        info_txt = f"[DEBUG] {operation} completed in {duration:.4f} seconds"
        print(info_txt)
    elif not is_debug:
        info_txt = f"[INFO] {operation} completed in {duration:.3f} seconds"
        print(info_txt)
    return info_txt



def anl_variables(var_name: str, opt_results: dict) -> dict:
    """
    Extract the values of a specific variable from the optimization results.

    This function extracts the values of a specified variable from the optimization results
    and returns them as a dictionary indexed by the variable indices.

    Args:
        var_name (str): The name of the variable to extract.
        opt_results (dict): The dictionary read from the result json file.

    Returns:
        dict: A dictionary containing the values of the specified variable, indexed by their indices.
    """
    values: dict = {}
    for var in opt_results["Vars"]:
        var_name = var["VarName"]
        if var_name.startswith(var_name + "["):
            index = int(var_name.split("[")[1].split("]")[0])
            values[index] = var["X"]
    return values



def set_times_style():
    import matplotlib
    params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],  # 回退字体
        "mathtext.fontset": "stix",   # 数学符号用 STIX（Times 风格）
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
    matplotlib.rcParams.update(params)