import json
from src.prepare import get_case


def anl_results(case_id: int, folder: str = "result/0617") -> dict | None:
    """
    Load the optimization results from a JSON file.

    This function reads the optimization results for a given case ID from a specified directory
    and returns the results as a dictionary.

    Args:
        case_id (int): The ID of the case for which the results are to be loaded.
        folder (str, optional): The directory where the results file is located. Defaults to "result/0617".

    Returns:
        dict: A dictionary containing the optimization results.
    """
    file = f"{folder}/{case_id}.json"
    try:
        res: dict = json.load(open(file, "r", encoding="utf-8"))
        return res
    except FileNotFoundError:
        print(f"case: {case_id} not found!")
        return None


def anl_variables(variable: str, opt_results: dict) -> dict:
    """
    Extract the values of a specific variable from the optimization results.

    This function extracts the values of a specified variable from the optimization results
    and returns them as a dictionary indexed by the variable indices.

    Args:
        variable (str): The name of the variable to extract.
        opt_results (dict): The dictionary containing the optimization results.

    Returns:
        dict: A dictionary containing the values of the specified variable, indexed by their indices.
    """
    values: dict = {}
    for var in opt_results["Vars"]:
        var_name = var["VarName"]
        if var_name.startswith(variable + "["):
            index = int(var_name.split("[")[1].split("]")[0])
            values[index] = var["X"]
    return values


def anl_solution_info(opt_results: dict) -> dict:
    return opt_results["SolutionInfo"]


def anl_one_case(case_id: int, folder: str = "result/0617") -> dict:
    # Status, Runtime, ObjVal, MIPGap,
    # total_net_energy, total_oesd_energy, total_oesd_energy_charged,
    # varpi_fnished, travel_time
    case_result = anl_results(case_id, folder)

    if case_result is None:
        return {}

    case_info = get_case(case_id)
    case_result_info = anl_solution_info(case_result)

    if case_result_info["SolCount"] == 0:
        return {}

    # phi_n, phi_b, xi, varpi, T
    phi_n_l2r = anl_variables("L2R_phi_n", case_result)
    phi_b_l2r = anl_variables("L2R_phi_b", case_result)
    xi_l2r = anl_variables("L2R_xi", case_result)
    varpi_l2r = anl_variables("L2R_varpi", case_result)
    t_l2r = anl_variables("L2R_t", case_result)

    phi_n_r2l = anl_variables("R2L_phi_n", case_result)
    phi_b_r2l = anl_variables("R2L_phi_b", case_result)
    xi_r2l = anl_variables("R2L_xi", case_result)
    varpi_r2l = anl_variables("R2L_varpi", case_result)
    t_r2l = anl_variables("R2L_t", case_result)
    
    e = anl_variables("e", case_result)

    full_info = {
        "case_id": case_id,
        "train": case_info["train"].type,
        "oesd": case_info["oesd"].type,
        "varpi_0": case_info["varpi_0"],
        "S_": case_info["S_"],
        "section_time": case_info["T_range"],
        "status": case_result_info['Status'],
        "runtime": case_result_info['Runtime'],
        "objval": case_result_info['ObjVal'],
        "mipgap": case_result_info['MIPGap'],
        "net_l2r": sum(phi_n_l2r.values()),
        "net_r2l": sum(phi_n_r2l.values()),
        "oesd_l2r": sum(phi_b_l2r.values()),
        "oesd_r2l": sum(phi_b_r2l.values()),
        "oesd_charge_l2r": sum(xi_l2r.values()),
        "oesd_charge_r2l": sum(xi_r2l.values()),
        "T_l2r": sum(t_l2r.values()),
        "T_r2l": sum(t_r2l.values()),
        "varpi_left_l2r": 0,
        "varpi_left_r2l": 0,
        "e_min": min(e.values()),
    }
    if case_info['oesd'].type is not None:
        if case_info['direction'][0]:
            full_info.update({"varpi_left_l2r": list(varpi_l2r.values())[-1]})
        if case_info['direction'][1]:
            full_info.update({"varpi_left_r2l": list(varpi_r2l.values())[0]})

    return full_info
