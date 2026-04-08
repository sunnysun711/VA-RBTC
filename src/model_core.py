from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np


V_MIN = 1e-1  # consistent with RBTC start/stop speed

def model_va(
    m: gp.Model,
    case: dict,
    va_solution: Optional[dict] = None,
) -> dict[str, gp.tupledict]:
    """
    Build the vertical alignment (VA) model for the given case.

    :param gp.Model m: The model instance to which the VA constraints and variables will be added. 
    :param dict case: A dictionary containing case-specific parameters, used to create LC.
    :param dict, optional va_solution: A dictionary containing solution variables 'e' with keys from 1 to S_+1.
                                       If provided, only e will be modeled.

    :return: A dictionary of variables of VA model. 
             The keys are variable names, and the values are tupledicts of variables.
    :rtype: dict[str, gp.tupledict]
    """
    # >>>>>>>>>>>>>>>> parameters <<<<<<<<<<<<<<<<<<
    S_ = case["S_"]
    g_min, g_max = case["g_min"], case["g_max"]
    e_min = min(case["P1"][1], case["P2"][1]) - g_max * S_ / 2
    e_max = max(case["P1"][1], case["P2"][1]) + g_min * S_ / 2
    M1, S, M2, SIGMA = case["M1"], case["S"], case["M2"], case["SIGMA"]
    dg_max = case["dg_max"]
    ALEPH1, ALEPH2 = case["ALEPH1"], case["ALEPH2"]

    # >>>>>>>>>>>>>>>> variables <<<<<<<<<<<<<<<<<<
    e           = m.addVars(range(1, S_ + 2), vtype=GRB.CONTINUOUS, lb=e_min, ub=e_max, name="e")
    pi          = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name="pi")
    e_bar       = m.addVar(lb=e_min, ub=e_max, name="e_bar")  # lowest elevation
    B_minus_s   = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_minus_s")
    B_add_s     = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_add_s")
    g_s         = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=g_min, ub=g_max, name="g_s")
    eth_minus_s = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0.0, ub=g_max, name="eth_minus_s")
    eth_add_s   = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0.0, ub=g_max, name="eth_add_s")

    # >>>>>>>>> constraints <<<<<<<<<<
    # extra index for e_s
    m.addConstr(e[S_ + 1] == ALEPH2[S_] + ALEPH2[S_] - ALEPH2[S_ - 1], name="elevation_[S_+1]")
    # slope length
    for i in range(M1, S + M1 - SIGMA + 3):
        m.addConstr(
            gp.quicksum(pi[i] for i in range(i, i + SIGMA)) <= 1,
            name=f"slope_length[{i}]",
        )
    # slope gradient
    for s in range(M1, S + M1 + 2):
        # separate_grad_binary
        m.addConstr(B_minus_s[s] + B_add_s[s] == 1, name=f"separate_grad_binary[{s}]")
        # linearize: eth_add_s = B_add_s * g_s  (McCormick)
        m.addConstr(eth_add_s[s]/g_min >= B_add_s[s], name=f"mc_add_1[{s}]")
        m.addConstr(eth_add_s[s]/g_max <= B_add_s[s], name=f"mc_add_2[{s}]")
        m.addConstr(eth_add_s[s]/g_max >= g_s[s]/g_max - (1 - B_add_s[s]), name=f"mc_add_3[{s}]")
        m.addConstr(eth_add_s[s]/g_min <= g_s[s]/g_min - (1 - B_add_s[s]), name=f"mc_add_4[{s}]")
        # gradient: e_{s+1} - e_s = eth_add - eth_minus = 2*eth_add - g_s
        m.addConstr(e[s + 1] - e[s] == 2 * eth_add_s[s] - g_s[s], name=f"gradient_balance[{s}]")
        # gradient_linear
        m.addConstr((e[s + 1] + e[s - 1] - 2 * e[s])/dg_max <= pi[s], name=f"gradient_linear1[{s}]")
        m.addConstr((e[s + 1] + e[s - 1] - 2 * e[s])/dg_max >= -pi[s], name=f"gradient_linear2[{s}]")
    # platform
    for s in range(1, M1 + 1):
        m.addConstr(e[s] == ALEPH1[s], name=f"platform1[{s}]")
    for s in range(S + M1 + 1, S_ + 1):
        m.addConstr(e[s] == ALEPH2[s], name=f"platform2[{s}]")
    # platform_vpi
    for s in range(1, M1):
        m.addConstr(pi[s] == 0, name=f"platform_vpi1[{s}]")
    for s in range(S + M1 + 2, S_ + 1):
        m.addConstr(pi[s] == 0, name=f"platform_vpi2[{s}]")
    # e_bar
    m.addConstrs((e[s] >= e_bar for s in range(1, S_ + 2)), name="e_bar")

    # va solution mode
    if va_solution is not None:
        m.addConstrs(
            (e[s] == va_solution[s] for s in range(1, S_ + 2)), name="va_solution:e"
        )

    # save them in _va_variables
    _va_variables = {}
    _va_variables["e"] = e
    _va_variables["pi"] = pi
    _va_variables["e_bar"] = gp.tupledict({None: e_bar})
    _va_variables["B_minus_s"] = B_minus_s
    _va_variables["B_add_s"] = B_add_s
    _va_variables["g_s"] = g_s
    _va_variables["eth_minus_s"] = eth_minus_s
    _va_variables["eth_add_s"] = eth_add_s
    
    return _va_variables



def model_rbtc(
    m: gp.Model,
    case: dict,
    e: dict | gp.tupledict,
    l2r: bool = True,
) -> dict[str, gp.tupledict]:
    """
    :param gp.Model m: the model instance
    :param dict e: elevation variables
    :param dict case: A dictionary containing case-specific parameters, used to create LC.
    :param bool, optional l2r: short for "left to right"
    :return: A dictionary containing the variables of the model, including energy, velocity, acceleration, force, and time.

            access with `f"L2R_{varname}"` or `f"R2L_{varname}"`, acceptable `varname` as below:

            [E_hat, v, a, f, b, f_max, b_max, gamma, t, E_s, R_s, Psi_t, phi_n, phi_b, w0, wi, wtn, T]
    :rtype: dict[str, gp.tupledict]

    """
    # parameters
    S_ = case["S_"]
    g = case["g"]
    train_mass, oesd_mass = (
        case["train"].data["mass"],
        case["oesd"].data["mass"],
    )
    total_mass = train_mass + oesd_mass
    rho = case["train"].data["rho"]
    M_f, M_b = (
        case["train"].data["max_force"],
        case["train"].data["max_force"],
    )
    r0, r1, r2 = (
        case["train"].data["davis_a"],
        case["train"].data["davis_b"],
        case["train"].data["davis_c"],
    )
    rtn = case["train"].data["r_tn"]
    a_max = case["train"].data["a_max"]
    v_max = case["train"].data["v_max"]
    v_min = V_MIN
    eta_b = case["train"].data["eta_b"]
    eta_n = case["train"].data["eta_n"]
    force_ek_pwa = case["train"].data["pwa"]
    v_ek_pwa = case["train"].data["pwa_v_E"]
    one_over_v_ek_pwa = case["train"].data["pwa_1/v_E"]
    T_range = case["T_range"]
    ds = case["ds"]
    mu = case["train"].data["mu"] / 1000
    # resistance upper bounds in kN
    _w0_ub = (r0 + r1 * v_max + r2 * v_max**2)
    _wi_ub = case["g_max"] / ds * total_mass * g + 1e-2  # for precision issues
    _wtn_ub = rtn * r2 * v_max**2
    _phi_n_ub, _phi_b_ub = (M_f * ds + ds/v_min * mu) / eta_n, (M_f * ds + ds/v_min * mu) / eta_b
    
    dir_prefix: str = "L2R_" if l2r else "R2L_"

    # variables
    v     = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=v_min, ub=v_max, name=f"{dir_prefix}v")  # m/s
    v2    = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=v_min**2, ub=v_max**2, name=f"{dir_prefix}v2")  # (m/s)^2
    E_hat = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=v_min**2/2, ub=v_max**2 / 2, name=f"{dir_prefix}E_hat")
    a     = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=-a_max, ub=a_max, name=f"{dir_prefix}a")
    f     = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f")  # kN
    b     = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b")  # kN
    f_max = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f_max")  # kN
    b_max = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b_max")  # kN
    gamma = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name=f"{dir_prefix}gamma")
    t     = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=ds/v_max, ub=ds/v_min, name=f"{dir_prefix}t") 
    Psi_t = m.addVars(range(1, S_), vtype=GRB.CONTINUOUS, lb=1/v_max, ub=1/v_min, name=f"{dir_prefix}Psi_t")
    phi_n = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_phi_n_ub, name=f"{dir_prefix}phi_n")  # kJ
    phi_b = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_phi_b_ub, name=f"{dir_prefix}phi_b")  # kJ
    w0    = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_w0_ub, name=f"{dir_prefix}w0")  # kN
    wi    = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=-_wi_ub, ub=_wi_ub, name=f"{dir_prefix}wi")  #kN
    wtn   = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_wtn_ub, name=f"{dir_prefix}wtn")  # kN
    T     = m.addVar(vtype=GRB.CONTINUOUS, lb=T_range[0], ub=T_range[1], name=f"{dir_prefix}T")  # s
    
    # constraints
    if l2r:
        # newton's law
        m.addConstrs(
            (2 * a[s] * ds == v2[s] - v2[s - 1] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}2as",
        )
        m.addConstrs(
            (
                total_mass * (1 + rho) * a[s] == f[s] - b[s] - w0[s] - wi[s] - wtn[s]
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}f=ma",
        )

        # resistances
        m.addConstrs(
            (
                wi[s] == (e[s + 1] - e[s]) / ds * total_mass * g
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}wi",
        )
        m.addConstrs(
            (
                w0[s] == (r0 + r1 * v[s - 1] + r2 * v2[s - 1])
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}w0",
        )
        m.addConstrs(
            (wtn[s] == rtn * r2 * v2[s - 1] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}wtn",
        )

        # velocity to v2 (relaxed version)
        for s in range(0, S_+1):
            m.addQConstr(v2[s] == v[s]**2)

        m.addConstrs((E_hat[s] == v2[s] / 2 for s in range(0, S_ + 1)), name=f"{dir_prefix}E_hat")
        
        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(
                E_hat[s - 1],
                f_max[s],
                force_ek_pwa[0],
                [i/1000 for i in force_ek_pwa[1]],
                name=f"{dir_prefix}force_ek_pwa[{s}]",
            )
            m.addGenConstrPWL(
                E_hat[s - 1],
                b_max[s],
                force_ek_pwa[0],
                [i/1000 for i in force_ek_pwa[1]],
                name=f"{dir_prefix}brake_ek_pwa[{s}]",
            )
        m.addConstrs(
            (f[s] <= f_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_f"
        )
        m.addConstrs(
            (b[s] <= b_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_b"
        )
        m.addConstrs(
            (f[s] <= M_f * gamma[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}f_s"
        )
        m.addConstrs(
            (b[s] <= M_b * (1 - gamma[s]) for s in range(1, S_ + 1)),
            name=f"{dir_prefix}b_s",
        )

        # energy
        m.addConstrs(
            (f[s] * ds + mu * t[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}eta",
        )

        # time related constraints
        # PWA expressions or gurobi embedded methods (might incur infeasibility)
        for s in range(1, S_):
            # m.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
            # Refer to https://www.gurobi.com/documentation/11.0/refman/funcpieces.html for option attributes.
            # m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
            m.addQConstr(v[s] * Psi_t[s] == 1, name=f"{dir_prefix}Psi_t[{s}]")
        m.addConstr(t[1] == 2 * ds * Psi_t[1], name=f"{dir_prefix}t[1]")
        m.addConstr(t[S_] == 2 * ds * Psi_t[S_ - 1], name=f"{dir_prefix}t[{S_}]")
        for s in range(2, S_):
            m.addConstr(
                t[s] == ds / 2 * (Psi_t[s - 1] + Psi_t[s]),
                name=f"{dir_prefix}t[{s}]",
                )
        m.addConstr(T == t.sum(), name=f"{dir_prefix}T")

        # start and stop velocity
        for s in [0, S_]:
            m.addConstr(v[s] == v_min, name=f"{dir_prefix}speed_limit[{s}]")

    else:  # right to left
        # newton's law
        m.addConstrs(
            (2 * a[s] * ds == v2[s - 1] - v2[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}2as",
        )
        m.addConstrs(
            (
                total_mass * (1 + rho) * a[s] == f[s] - b[s] - w0[s] - wi[s] - wtn[s]
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}f=ma",
        )

        # resistances
        m.addConstrs(
            (
                wi[s] == (e[s] - e[s + 1]) / ds * total_mass * g
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}wi",
        )
        m.addConstrs(
            (
                w0[s] == (r0 + r1 * v[s] + r2 * v2[s])
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}w0",
        )
        m.addConstrs(
            (wtn[s] == rtn * r2 * v2[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}wtn",
        )

        # velocity to v2 (relaxed version)
        for s in range(0, S_+1):
            m.addQConstr(v2[s] == v[s]**2)

        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(
                E_hat[s],
                f_max[s],
                force_ek_pwa[0],
                [i/1000 for i in force_ek_pwa[1]],
                name=f"{dir_prefix}force_ek_pwa[{s}]",
            )
            m.addGenConstrPWL(
                E_hat[s],
                b_max[s],
                force_ek_pwa[0],
                [i/1000 for i in force_ek_pwa[1]],
                name=f"{dir_prefix}brake_ek_pwa[{s}]",
            )
        m.addConstrs(
            (f[s] <= f_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_f"
        )
        m.addConstrs(
            (b[s] <= b_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_b"
        )
        m.addConstrs(
            (f[s] <= M_f * gamma[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}f_s"
        )
        m.addConstrs(
            (b[s] <= M_b * (1 - gamma[s]) for s in range(1, S_ + 1)),
            name=f"{dir_prefix}b_s",
        )

        # energy
        m.addConstrs(
            (f[s] * ds + mu * t[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}eta",
        )

        # time related constraints
        for s in range(1, S_):
            # model.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
            # m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
            m.addQConstr(v[s] * Psi_t[s] == 1, name=f"{dir_prefix}Psi_t[{s}]")
        m.addConstr(t[1] == 2 * ds * Psi_t[1], name=f"{dir_prefix}t[1]")
        m.addConstr(t[S_] == 2 * ds * Psi_t[S_ - 1], name=f"{dir_prefix}t[{S_}]")
        for s in range(2, S_):
            m.addConstr(
                t[s] == ds / 2 * (Psi_t[s - 1] + Psi_t[s]), name=f"{dir_prefix}t[{s}]"
            )
        m.addConstr(T == t.sum(), name=f"{dir_prefix}T")

        # start and stop velocity
        for s in [0, S_]:
            m.addConstr(v[s] == v_min, name=f"{dir_prefix}speed_limit_v[{s}]")

    rbtc_variables: dict[str, gp.tupledict] = {}
    rbtc_variables[f"{dir_prefix}E_hat"] = E_hat
    rbtc_variables[f"{dir_prefix}v"] = v
    rbtc_variables[f"{dir_prefix}v2"] = v2
    rbtc_variables[f"{dir_prefix}a"] = a
    rbtc_variables[f"{dir_prefix}f"] = f
    rbtc_variables[f"{dir_prefix}b"] = b
    rbtc_variables[f"{dir_prefix}f_max"] = f_max
    rbtc_variables[f"{dir_prefix}b_max"] = b_max
    rbtc_variables[f"{dir_prefix}gamma"] = gamma
    rbtc_variables[f"{dir_prefix}t"] = t
    rbtc_variables[f"{dir_prefix}Psi_t"] = Psi_t
    rbtc_variables[f"{dir_prefix}phi_n"] = phi_n
    rbtc_variables[f"{dir_prefix}phi_b"] = phi_b
    rbtc_variables[f"{dir_prefix}w0"] = w0
    rbtc_variables[f"{dir_prefix}wi"] = wi
    rbtc_variables[f"{dir_prefix}wtn"] = wtn
    rbtc_variables[f"{dir_prefix}t"] = t
    rbtc_variables[f"{dir_prefix}T"] = gp.tupledict({None: T})
    return rbtc_variables


def model_oesd(
    m: gp.Model,
    case: dict,
    rbtc_variables: dict,
    varpi_0: float = 1,
    l2r: bool = True,
    cyclic: bool | float | int = False,
    convexify : bool = True,
    power_time_trapezoid: bool = False,
) -> dict[str, gp.tupledict]:
    r"""model OESD

    :param gp.Model m: gurobi model
    :param dict case: case data dictionary, is used when m is not a VA_RBTC_OESD model, defaults to None
    :param dict rbtc_variables: RBTC variables dictionary, is used when m is not a VA_RBTC_OESD model, defaults to None
    :param float varpi_0: initial State-of-Energy, defaults to 1
    :param bool l2r: direction, short for left-to-right, defaults to True
    :param bool | float | int cyclic: whether to apply cyclic soe condition, defaults to False
        if True: varpi_end = varpi_start = varpi_0
        if False: no constraint
        if float or int: varpi_end \in [varpi_0-cyclic, varpi_0 + cyclic] (the result range must be between 0 and 1)
    :param bool convexify: whether to apply convexification to the soe-power function, defaults to True.
    :param bool power_time_trapezoid: whether to apply power time trapezoid constraint, defaults to False
        if False: phi_b[s] <= kappa_minus[s] * t[s]  # Rectangle
                  xi[s] <= kappa_plus[s] * t[s]  # Rectangle
        if True: phi_b[s] <= (kappa_minus[s] + kappa_minus[s-1]) * t[s] / 2  # Trapezoid
                 xi[s] <= (kappa_plus[s] + kappa_plus[s-1]) * t[s] / 2  # Trapezoid
        
    :return dict[str, gp.tupledict]: A dictionary containing the variables of the OESD model

            access with `f"L2R_{varname}"` or `f"R2L_{varname}"`, acceptable `varname` as below:

            [varpi, kappa_plus, kappa_minus, xi], when OESD is applied.

            [xi], when OESD type is None.
    """
    dir_prefix: str = "L2R_" if l2r else "R2L_"
    # previously defined variables
    phi_b = rbtc_variables[f"{dir_prefix}phi_b"]  # kJ
    t = rbtc_variables[f"{dir_prefix}t"]
    b = rbtc_variables[f"{dir_prefix}b"]  # kN
    
    S_ = case["S_"]
    if case["oesd"].type is None:
        xi = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"{dir_prefix}xi")  # kJ
        m.addConstrs((phi_b[s] == 0 for s in range(1, S_ + 1)), name=f"{dir_prefix}phi_b")
        return {f"{dir_prefix}xi": xi}
    
    # parameters
    XI = case["oesd"].data["capacity"] * 3600  # kJ
    ds = case["ds"]
    eta_b = case["train"].data["eta_b"]
    _max_charge_power = max(case["oesd"].data["charge"]["y"]) / 1000  # kW
    _max_discharge_power = max(case["oesd"].data["discharge"]["y"]) / 1000  # kW
    _t_ub = case["T_range"][1] - ds * (S_ - 1) / case["train"].data["v_max"]

    # variables
    varpi = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=100, name=f"{dir_prefix}varpi")  # (%)
    kappa_plus = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_charge_power, name=f"{dir_prefix}kappa_plus")  # kW
    kappa_minus = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_discharge_power, name=f"{dir_prefix}kappa_minus")  # kW
    xi = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_charge_power * _t_ub, name=f"{dir_prefix}xi")  # kJ
        
    # contraints
    # (dis)charging power curve pwa
    
    if not convexify:  # PWL
        for s in range(0, S_ + 1):
            m.addGenConstrPWL(
                varpi[s],
                kappa_plus[s],
                [i*100 for i in case["oesd"].data["charge"]["x"]],  # (%)
                [i/1000 for i in case["oesd"].data["charge"]["y"]],  # kW
                name=f"{dir_prefix}kappa_plus[{s}]",
            )
            m.addGenConstrPWL(
                varpi[s],
                kappa_minus[s],
                [i*100 for i in case["oesd"].data["discharge"]["x"]],  # (%)
                [i/1000 for i in case["oesd"].data["discharge"]["y"]],  # kW
                name=f"{dir_prefix}kappa_minus[{s}]",
            )
    else:  # Convex reformulation
        # NOTE: 需要保证pwl曲线是开口向下的concave的，才能保证可松弛。
        # Replace the entire PWL loop with:
        for s in range(0, S_ + 1):
            # Charge power (concave) → linear upper envelope
            xc = [i * 100 for i in case["oesd"].data["charge"]["x"]]
            yc = [i / 1000 for i in case["oesd"].data["charge"]["y"]]
            for k in range(len(xc) - 1):
                slope = (yc[k+1] - yc[k]) / (xc[k+1] - xc[k])
                m.addConstr(
                    kappa_plus[s] <= yc[k] + slope * (varpi[s] - xc[k]),
                    name=f"{dir_prefix}kappa_plus_seg{k}[{s}]",
                )

            # Discharge power (concave) → linear upper envelope
            xd = [i * 100 for i in case["oesd"].data["discharge"]["x"]]
            yd = [i / 1000 for i in case["oesd"].data["discharge"]["y"]]
            for k in range(len(xd) - 1):
                slope = (yd[k+1] - yd[k]) / (xd[k+1] - xd[k])
                m.addConstr(
                    kappa_minus[s] <= yd[k] + slope * (varpi[s] - xd[k]),
                    name=f"{dir_prefix}kappa_minus_seg{k}[{s}]",
                )

    if l2r:
        for s in range(1, S_ + 1):
            if power_time_trapezoid:
                m.addQConstr(2 * phi_b[s] <= (kappa_minus[s-1] + kappa_minus[s]) * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
                m.addQConstr(2 * xi[s] <= (kappa_plus[s-1] + kappa_plus[s]) * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
            else:
                m.addQConstr(phi_b[s] <= kappa_minus[s-1] * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
                m.addQConstr(xi[s] <= kappa_plus[s-1] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
        m.addConstrs(
            (xi[s] <= b[s] * ds * eta_b for s in range(1, S_ + 1)), name=f"{dir_prefix}xi"
        )
        m.addConstrs(
            (
                varpi[s] == varpi[s - 1] + (xi[s] - phi_b[s]) / XI * 100
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}SOE_change",
        )
        m.addConstr(
            varpi[0] == varpi_0*100, name=f"{dir_prefix}varpi_0"
        )  # SOE starting state
        
        # optional for cyclic soe condition
        if isinstance(cyclic, bool):
            if cyclic:
                m.addConstr(varpi[S_] == varpi_0*100, name=f"{dir_prefix}varpi_S_1")
        elif isinstance(cyclic, (float, int)):
            assert varpi_0 * 100 - cyclic >= 0, f"cyclic={cyclic} is too large for varpi_0={varpi_0}"
            assert varpi_0 * 100 + cyclic <= 100, f"cyclic={cyclic} is too large for varpi_0={varpi_0}"
            m.addConstr(
                varpi[S_] >= varpi_0 * 100 - cyclic, 
                name=f"{dir_prefix}varpi_S_1"
            )
            m.addConstr(
                varpi[S_] <= varpi_0 * 100 + cyclic, 
                name=f"{dir_prefix}varpi_S_2"
            )
    else:
        for s in range(1, S_ + 1):
            if power_time_trapezoid:
                m.addQConstr(2 * phi_b[s] <= (kappa_minus[s] + kappa_minus[s-1]) * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
                m.addQConstr(2 * xi[s] <= (kappa_plus[s] + kappa_plus[s-1]) * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
            else:
                m.addQConstr(phi_b[s] <= kappa_minus[s] * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
                m.addQConstr(xi[s] <= kappa_plus[s] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
        m.addConstrs(
            (xi[s] <= b[s] * ds * eta_b for s in range(1, S_ + 1)), name=f"{dir_prefix}xi"
        )
        m.addConstrs(
            (
                varpi[s - 1] == varpi[s] + (xi[s] - phi_b[s]) / XI * 100
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}SOE_change",
        )
        m.addConstr(
            varpi[S_] == varpi_0*100, name=f"{dir_prefix}varpi_0"
        )  # SOE starting state
        
        # optional for cyclic soe condition
        if isinstance(cyclic, bool):
            if cyclic:
                m.addConstr(varpi[0] == varpi_0*100, name=f"{dir_prefix}varpi_S_1")
        elif isinstance(cyclic, (float, int)):
            assert varpi_0 * 100 - cyclic >= 0, f"cyclic={cyclic} is too large for varpi_0={varpi_0}"
            assert varpi_0 * 100 + cyclic <= 100, f"cyclic={cyclic} is too large for varpi_0={varpi_0}"
            m.addConstr(
                varpi[0] >= varpi_0*100 - cyclic, 
                name=f"{dir_prefix}varpi_S_1"
            )
            m.addConstr(
                varpi[0] <= varpi_0*100 + cyclic, 
                name=f"{dir_prefix}varpi_S_2"
            )

    oesd_variables = {}
    oesd_variables[f"{dir_prefix}varpi"] = varpi
    oesd_variables[f"{dir_prefix}kappa_plus"] = kappa_plus
    oesd_variables[f"{dir_prefix}kappa_minus"] = kappa_minus
    oesd_variables[f"{dir_prefix}xi"] = xi
    return oesd_variables



def solve_edge_va(case: dict, output_flag: int = 0, lowest_edge:bool=True) -> tuple[gp.Model, dict[int, float]]:
    """Get lowest/highest possible track elevation solutions and the absolute track edge value
    """
    S_ = case["S_"]
    m = gp.Model("temp_va_edge")
    va_variables = model_va(m, case=case)
    e = va_variables['e']
    if lowest_edge:
        m.setObjective(e.sum(), GRB.MINIMIZE)
    else:
        m.setObjective(e.sum(), GRB.MAXIMIZE)
    m.setParam("OutputFlag", output_flag)
    m.setParam("MIPGap", 0.0)
    # m.setParam("TimeLimit", 2) 
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError("Lowest VA infeasible")
    e_lowest_dict = {s: e[s].X for s in range(1, S_ + 2)}
    return m, e_lowest_dict


def get_energy_expr_one_direction(tc_vars: dict, oesd_vars: dict, prefix: str, is_oesd_included: bool) -> gp.LinExpr:
    """Build energy objective. phi_n.sum() if not `is_oesd_included` else sum(phi_n + phi_b - xi)"""
    expr = gp.LinExpr(0)
    expr += tc_vars[f"{prefix}phi_n"].sum()
    if is_oesd_included:
        expr += tc_vars[f"{prefix}phi_b"].sum()
        expr -= oesd_vars[f"{prefix}xi"].sum()
    return expr


def model_EC(m: gp.Model, case: dict, va_variables: dict, e_lowest_dict:dict|None=None):
    """Add Elevation-based Cuts to the model. (with LB, UB)"""
    # Elevation-based cuts: upper bound, connecting directly two platforms
    M1, S, M2 = case["M1"], case["S"], case["M2"]
    vi_e: dict = {i: 0 for i in range(M1 + 1, S + M1 + 1)}  # to be updated
    left_e, right_e = case["ALEPH1"][M1], case["ALEPH2"][S + M1 + 1]
    grad = (right_e - left_e) / (case["ds"] * (S + 1))
    for i in range(M1 + 1, S + M1 + 1):
        vi_e[i] = grad * case["ds"] * (i - M1) + left_e
        va_variables["e"][i].setAttr("UB", vi_e[i])  # section elevation-based cuts
    return

def _bang_bang_sweep(S_: int, ds: float, v_min: float, v_max: float, a_max: float):
    """Forward/backward bang-bang time-optimal sweep in O(S_).
 
    Returns (t_fwd, t_bwd, v_fwd, v_bwd), each np.ndarray of shape (S_+1,).
      - t_fwd[i]: min time from node 0 (v_min) to node i
      - t_bwd[i]: min time from node i to node S_ (v_min)
      - v_fwd[i]: max reachable speed at node i from the left
      - v_bwd[i]: max reachable speed at node i from the right
    """
    n = S_ + 1
    t_fwd = np.zeros(n)
    v_fwd = np.zeros(n)
    v_fwd[0] = v_min
 
    for i in range(S_):
        v_fwd[i + 1] = min((v_fwd[i] ** 2 + 2 * a_max * ds) ** 0.5, v_max)
        t_fwd[i + 1] = t_fwd[i] + 2 * ds / (v_fwd[i] + v_fwd[i + 1])
 
    t_bwd = np.zeros(n)
    v_bwd = np.zeros(n)
    v_bwd[S_] = v_min
 
    for i in range(S_ - 1, -1, -1):
        v_bwd[i] = min((v_bwd[i + 1] ** 2 + 2 * a_max * ds) ** 0.5, v_max)
        t_bwd[i] = t_bwd[i + 1] + 2 * ds / (v_bwd[i] + v_bwd[i + 1])
 
    return t_fwd, t_bwd, v_fwd, v_bwd

def model_TC(m: gp.Model, case: dict, rbtc_variables: dict, directions: tuple[bool, bool] | None = None):
    """Add Time-based Cuts (LB & UB). UB tightened per-interval via
    bang-bang sweep: t_ub[s] = T_max - t_fwd[s-1] - t_bwd[s].
    """
    ds    = case["ds"]
    v_max = case["train"].data["v_max"]
    v_min = V_MIN
    a_max = case["train"].data["a_max"]
    Tm    = case["T_range"][1]
    S_    = case["S_"]
 
    _t_lb = ds / v_max
    t_fwd, t_bwd, _, _ = _bang_bang_sweep(S_, ds, v_min, v_max, a_max)
    t_ubs = np.maximum(Tm - t_fwd[:-1] - t_bwd[1:], _t_lb)
 
    directions = case['direction'] if directions is None else directions
    if directions[0]:
        for s in range(1, S_ + 1):
            rbtc_variables["L2R_t"][s].setAttr("LB", _t_lb)
            rbtc_variables["L2R_t"][s].setAttr("UB", t_ubs[s - 1])
    if directions[1]:
        for s in range(1, S_ + 1):
            rbtc_variables["R2L_t"][s].setAttr("LB", _t_lb)
            rbtc_variables["R2L_t"][s].setAttr("UB", t_ubs[s - 1])
    
    window_lens = [i for i in range(2, S_//2 + 1)]
    for w in window_lens:
        if w < 2 or w > S_:
            continue
        for s in range(1, S_ - w + 2):          # s = 1 … S_-w+1
            # segments in window: s, s+1, …, s+w-1
            # outside-window minimum time:
            #   t_fwd[s-1]   = min time for segments 1 … s-1
            #   t_bwd[s+w-1] = min time for segments s+w … S_
            win_ub = max(Tm - t_fwd[s - 1] - t_bwd[s + w - 1],
                         w * _t_lb)

            if directions[0]:
                lhs = gp.quicksum(rbtc_variables["L2R_t"][s + k]
                                  for k in range(w))
                m.addConstr(lhs <= win_ub,
                            name=f"TC_win_L2R_w{w}_s{s}")
            if directions[1]:
                lhs = gp.quicksum(rbtc_variables["R2L_t"][s + k]
                                  for k in range(w))
                m.addConstr(lhs <= win_ub,
                            name=f"TC_win_R2L_w{w}_s{s}")
    return

def model_VC(m: gp.Model, case: dict, rbtc_variables: dict,
             directions: tuple[bool, bool] | None = None):
    """Add Velocity-based Cuts (VC) to the model. (with addConstr). `directions` is used when doing one-direction models

    Uses bang-bang sweep for time-budget computation and Cauchy-Schwarz
    for the RHS lower bound.  Window widths grow geometrically to keep
    cut count at O(S log S).
    """
    directions = case['direction'] if directions is None else directions

    S_    = case["S_"]
    ds    = case["ds"]
    v_max = case["train"].data["v_max"]
    v_min = V_MIN
    a_max = case["train"].data["a_max"]
    T_max = case["T_range"][1]

    # ---- precompute bang-bang reachability (O(S)) ----
    t_fwd, t_bwd, v_fwd, v_bwd = _bang_bang_sweep(S_, ds, v_min, v_max, a_max)
    v_ub = np.minimum(v_fwd, v_bwd)  # per-node velocity UB from bang-bang reachability

    # ---- geometric window widths: 1, 2, 4, 8, ... < S/2 ----
    window_lens: list[int] = []
    w = 1
    while w < S_ // 2:
        window_lens.append(w)
        w *= 2

    def _add_for_dir(prefix: str):
        v = rbtc_variables[f"{prefix}v"]
        
        for s in range(S_ + 1):
            v[s].setAttr("UB", v_ub[s])
            # m.addConstr(v[s] <= v_ub[s], name=f"{prefix}vc_ub[{s}]")

        for W in window_lens:
            for s in range(1, S_ - W + 2):
                # Window covers intervals [s, s+W-1], boundary nodes s-1 and s+W-1.
                # Time budget = T_max - min_time(0→node s-1) - min_time(node s+W-1→S_)
                t_bar_win = T_max - t_fwd[s - 1] - t_bwd[s + W - 1]
                if t_bar_win <= 0:
                    continue

                # Cauchy-Schwarz: Σ(v_{j-1}+v_j) ≥ 2·W²·Δs / T_W
                rhs = 2.0 * W * W * ds / t_bar_win

                expr = gp.LinExpr()
                for j in range(s, s + W):
                    expr += v[j - 1] + v[j]
                m.addConstr(expr >= rhs, name=f"{prefix}vc_W{W}[{s}]")

    if directions[0]:
        _add_for_dir("L2R_")
    if directions[1]:
        _add_for_dir("R2L_")
    return

def model_MS(
    m: gp.Model,
    case: dict,
    rbtc_variables: dict,
    directions: tuple[bool, bool] | None = None,
):
    """
    Add Mode-Switching Cuts (MS) that enforce monotonic traction-to-braking
    transition within each travel direction:

        L2R:  gamma_s <= gamma_{s-1}   (decreases along travel direction)
        R2L:  gamma_{s-1} <= gamma_s   (increases along travel direction)

    :param gp.Model m: gurobipy Model
    :param dict case: problem data dict
    :param dict rbtc_variables: dict containing gamma binaries for each direction
    :param tuple[bool, bool] | None directions: (include L2R, include R2L)
    """
    directions = case["direction"] if directions is None else directions
    S_ = case["S"]

    for dir_prefix, l2r in [("L2R_", True), ("R2L_", False)]:
        if not directions[0 if l2r else 1]:
            continue

        gamma = rbtc_variables[f"{dir_prefix}gamma"]

        for s in range(2, S_ + 1):
            if l2r:
                # L2R: train travels s=1->S, traction first then braking
                m.addConstr(
                    gamma[s] <= gamma[s - 1],
                    name=f"{dir_prefix}MS_{s}",
                )
            else:
                # R2L: train travels s=S->1, traction at high s, braking at low s
                m.addConstr(
                    gamma[s - 1] <= gamma[s],
                    name=f"{dir_prefix}MS_{s}",
                )

    m.update()


def model_SC(m: gp.Model, case: dict, va_variables: dict):
    """Add Slope-based Cuts to the model. (with LB, UB)
    """
    # the track should decends for first third and ascends for the last third, to fully utilize gravity
    M1, S, M2 = case["M1"], case["S"], case["M2"]
    _third = S // 3
    # decending for first third
    for s in range(M1, M1 + _third + 1):
        va_variables["B_minus_s"][s].setAttr("LB", 1)
        va_variables["B_minus_s"][s].setAttr("UB", 1)
        va_variables["B_add_s"][s].setAttr("LB", 0)
        va_variables["B_add_s"][s].setAttr("UB", 0)
    # ascending for the last third
    for s in range(S + M1 + 2 - _third, S + M1 + 2):
        va_variables["B_minus_s"][s].setAttr("LB", 0)
        va_variables["B_minus_s"][s].setAttr("UB", 0)
        va_variables["B_add_s"][s].setAttr("LB", 1)
        va_variables["B_add_s"][s].setAttr("UB", 1)
    
    return


def model_FC(m: gp.Model, case: dict, rbtc_variables: dict, directions: tuple[bool, bool] | None = None):
    """Add Force-based Cuts to the model. (with LB, UB). `directions` is used when doing one-direction models
    """
    S_ = case["S_"]
    am = case['train'].data['a_max']
    ds = case['ds']
    _v_cruise_min = 15  # minimal cruise speed, you should at least max-accelerate to this
    directions = case['direction'] if directions is None else directions
    num_intervals_max_force = int(np.ceil(_v_cruise_min**2 / (2 * (am) / 2 ) / ds))
    
    for s in range(1, num_intervals_max_force):
        if directions[0]:  # l2r
            rbtc_variables["L2R_gamma"][s].setAttr("LB", 1)  # fixed to 1, no deceleration
            rbtc_variables["L2R_gamma"][s].setAttr("UB", 1)  # fixed to 1, no deceleration
        # no guarantee for braking phases
        if directions[1]:  # r2l
            rbtc_variables["R2L_gamma"][s].setAttr("UB", 0)  # fixed to 0, no acceleration
            rbtc_variables["R2L_gamma"][s].setAttr("LB", 0)  # fixed to 0, no acceleration
    for s in range(S_ - num_intervals_max_force + 1, S_ + 1):
        # no guarantee for braking phases
        if directions[0]:  # l2r
            rbtc_variables["L2R_gamma"][s].setAttr("UB", 0)  # fixed to 0, no acceleration
            rbtc_variables["L2R_gamma"][s].setAttr("LB", 0)  # fixed to 0, no acceleration
        if directions[1]:  # r2l
            rbtc_variables["R2L_gamma"][s].setAttr("LB", 1)  # fixed to 1, no deceleration
            rbtc_variables["R2L_gamma"][s].setAttr("UB", 1)  # fixed to 1, no deceleration
    
    return


if __name__ == "__main__":
    from src.prepare import get_case
    import matplotlib.pyplot as plt
    
    m1 = gp.Model()
    case = get_case(10030122)
    S_ = case["S_"]
    va_vars = model_va(m1, case=case, va_solution=None)
    e = va_vars['e']
    tc_vars = model_rbtc(m1, case=case, e=e)
    es_vars = model_oesd(m1, case=case, rbtc_variables=tc_vars)
    tc_vars2 = model_rbtc(m1, l2r=False, case=case, e=e)
    es_vars2 = model_oesd(m1, l2r=False, case=case, rbtc_variables=tc_vars2)
    
    expr = gp.LinExpr(0)
    expr += get_energy_expr_one_direction(tc_vars, es_vars, "L2R_", True)
    expr += get_energy_expr_one_direction(tc_vars2, es_vars2, "R2L_", False)
    m1.setObjective(expr, GRB.MINIMIZE)
    
    # m1.write("testing.lp")
    
    # m1.setParam(gp.GRB.Param.MIPGap, 0.01)
    m1.setParam(gp.GRB.Param.TimeLimit, 60)
    m1.optimize()
    
    l2r = True
    dir_prefix = "L2R_"
    e = np.array([va_vars['e'][i].x for i in va_vars['e'].keys()])
    v = np.array([tc_vars[f'{dir_prefix}v'][i].x for i in tc_vars[f'{dir_prefix}v'].keys()])
    f = np.array([tc_vars[f'{dir_prefix}f'][i].x for i in tc_vars[f'{dir_prefix}f'].keys()])
    b = np.array([tc_vars[f'{dir_prefix}b'][i].x for i in tc_vars[f'{dir_prefix}b'].keys()])
    t = np.array([tc_vars[f'{dir_prefix}t'][i].x for i in tc_vars[f'{dir_prefix}t'].keys()])
    cumsum_t = np.cumsum(t)
    phi_n = np.array(
        [tc_vars[f'{dir_prefix}phi_n'][i].x for i in tc_vars[f'{dir_prefix}phi_n'].keys()])
    phi_b = np.array(
        [tc_vars[f'{dir_prefix}phi_b'][i].x for i in tc_vars[f'{dir_prefix}phi_b'].keys()])
    xi = np.array([es_vars[f'{dir_prefix}xi'][i].x for i in es_vars[f'{dir_prefix}xi'].keys()])
    varpi = np.array(
        [es_vars[f'{dir_prefix}varpi'][i].x for i in es_vars[f'{dir_prefix}varpi'].keys()])
    R_s = np.array([tc_vars[f'{dir_prefix}R_s'][i].x for i in tc_vars[f'{dir_prefix}R_s'].keys()])


    fig, axes = plt.subplots(4, 1, figsize=(5, 10))
    # first subplot: elevation (1,S_) and speed (0,S_)
    ax_e = axes[0]
    ax_v = ax_e.twinx()
    ele = ax_e.plot(range(1, S_ + 1), e[:-1], "ro-", label="Track elevation")
    vel = ax_v.plot([i + 0.5 for i in range(0, S_ + 1)], v, "b-", label="Speed")
    ax_e.set_ylabel("Track Elevation (m)")
    ax_e.set_ylim((e[0] - 30, e[0] + 5))
    ax_v.set_ylabel("Speed (m/s)")
    lines = ele + vel
    ax_e.legend(lines, [ll.get_label() for ll in lines])

    # second subplot: control forces and times
    ax_f = axes[1]
    ax_t = ax_f.twinx()
    tra = ax_f.plot(range(1, S_ + 1), f, "rx-", label="Traction force")
    bra = ax_f.plot(range(1, S_ + 1), -b, "bx-", label="Braking force")
    # control = ax_f.plot(range(1, S_ + 1), (f - b) / 1000, "rx-", label="Control force")
    ax_f.axhline(0, color="grey", alpha=0.3, linestyle=":")
    ax_f.set_ylabel("Force (kN)")
    cumsum_t = cumsum_t if l2r else cumsum_t[::-1]
    cst = ax_t.plot(range(1, S_ + 1), cumsum_t, color="grey", ls="-", label="Cumulative time")
    ax_t.set_ylabel("Time (s)")
    lines = tra + bra + cst
    ax_f.legend(lines, [ll.get_label() for ll in lines])

    # third subplot: (dis)charge
    ax_energy = axes[2]
    # ax_soe = ax_energy.twinx()
    ax_energy.set_ylabel("Energy consumption (kWh)")
    ln = ax_energy.plot(range(1, S_ + 1), phi_n / 3600, color="#8ECFC9", marker="", ls="-",
                        label="Energy discharged from Net")
    lo = ax_energy.plot(range(1, S_ + 1), phi_b / 3600, color="#FFBF7A", marker="", ls="-",
                        label="Energy discharged from OESD")
    lx = ax_energy.plot(range(1, S_ + 1), xi / 3600, color="#FA7F6F", ls="-", marker="",
                        label="Energy charged to OESD")
    lines = ln + lo + lx
    ax_energy.legend(lines, [ll.get_label() for ll in lines])

    # fourth subplot: SOE
    ax_soe = axes[3]
    ls = ax_soe.plot([i + 0.5 for i in range(0, S_ + 1)], varpi, color="#82B0D2", marker="x", ls="-",
                     label="State of Energy (SOE)")
    ax_soe.set_ylim([-5, 105])
    ax_soe.set_ylabel("State of Energy (%)")
    ax_soe.set_xlabel("Intervals")
    ax_soe.legend()

    fig = plt.gcf()
    plt.tight_layout()
    plt.show()