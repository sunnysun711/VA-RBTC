import gurobipy as gp
from gurobipy import GRB
import numpy as np


def model_va(
    m: gp.Model,
    case: dict | None = None,
    va_solution: dict | None = None,
    e_floor: float | None = None,
) -> dict[str, gp.tupledict]:
    """
    Build the vertical alignment (VA) model for the given case.

    :param gp.Model m: The model instance to which the VA constraints and variables will be added. 
                       Should be a `VA_RBTC_OESD` object.
    :param dict, optional case: A dictionary containing case-specific parameters, used to create LC. If None, the
                                case details from the model instance `m` will be used.
    :param dict, optional va_solution: A dictionary containing solution variables 'e' with keys from 1 to S_+1.
                                       If provided, only e will be modeled.
    :param float | None, optional e_floor: Additional lower bounds for all elevations. 

    :return: A dictionary of variables of VA model. 
             The keys are variable names, and the values are tupledicts of variables.
             If va_solution is provided, only 'e' will be modeled and returned.
    :rtype: dict[str, gp.tupledict]
    
    Parameters from `case` dictionary:
        S_ (int): The number of sections in the vertical alignment.
        g_min (float): The minimum gradient value.
        g_max (float): The maximum gradient value.
        M1 (int): The starting section index for slope constraints.
        S (int): The total number of sections.
        M2 (int): The ending section index for slope constraints.
        SIGMA (int): The maximum number of consecutive sections that can have a slope.
        dg_max (float): The maximum allowable change in gradient between sections.
        ALEPH1 (list): The elevation values at the beginning of the section.
        ALEPH2 (list): The elevation values at the end of the section.

    Variables created:
        e (dict): Continuous variables representing elevation at each section index.
        pi (dict): Binary variables indicating whether a section is part of a slope.
        B_minus_s (dict): Binary variables for negative gradient range.
        B_add_s (dict): Binary variables for positive gradient range.
        g_minus_s (dict): Continuous variables for negative gradient values.
        g_add_s (dict): Continuous variables for positive gradient values.
        eth_minus_s (dict): Continuous variables for negative gradient auxiliary values.
        eth_add_s (dict): Continuous variables for positive gradient auxiliary values.

    Constraints added:
        - Elevation constraint at the last section.
        - Slope length constraints ensuring only one slope within a specified range.
        - Slope gradient constraints separating positive and negative gradients.
        - Gradient linear constraints limiting changes in gradient between sections.
        - Platform constraints fixing elevation values at the beginning and end sections.
        - Platform slope constraints setting slope indicators to zero for platform sections.

    """
    # specify parameters
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) is dict, f"Input a 'VA_RBTC_OESD' model or provide 'case' parameter."
    va_variables: dict | None = getattr(m, "va_variables", None)
    
    S_ = case["S_"]
    g_min, g_max = case["g_min"], case["g_max"]
    M1, S, M2, SIGMA = case["M1"], case["S"], case["M2"], case["SIGMA"]
    dg_max = case["dg_max"]
    ALEPH1, ALEPH2 = case["ALEPH1"], case["ALEPH2"]

    # _va_variables
    e = m.addVars(
        range(1, S_ + 2), vtype=GRB.CONTINUOUS, name="e"
    )  # extra index (S_+1) is to calculate w_{i, S_}
    pi = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name="pi")

    # va solution mode
    if va_solution is not None:
        m.addConstrs(
            (e[s] == va_solution[s] for s in range(1, S_ + 2)), name="va_solution:e"
        )
        if va_variables is not None:
            va_variables["e"] = e
        return {"e": e}

    if e_floor:  # manually lift track elevations (lowest intervals)
        m.addConstrs(
            (e[s] >= e_floor for s in range(1, S_ + 2)), name=f"floor_lift")

    # aux _va_variables for gradient range
    B_minus_s = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_minus_s")
    B_add_s = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_add_s")
    g_minus_s = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=-g_max, ub=-g_min, name="g_minus_s"
    )
    g_add_s = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=g_min, ub=g_max, name="g_add_s"
    )
    eth_minus_s = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=-g_max, ub=0.0, name="eth_minus_s"
    )
    eth_add_s = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0.0, ub=g_max, name="eth_add_s"
    )

    # >>>>>>>>> constraints <<<<<<<<<<
    # extra index for e_s
    m.addConstr(
        e[S_ + 1] == ALEPH2[S_] + ALEPH2[S_] - ALEPH2[S_ - 1], name="elevation_[S_+1]"
    )
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

        # linearize: B*g -> eth (use indicator constraint)
        m.addConstr(
            (B_minus_s[s] == 1) >> (eth_minus_s[s] == g_minus_s[s]),
            name=f"ind_minus1[{s}]",
        )
        m.addConstr(
            (B_minus_s[s] == 0) >> (eth_minus_s[s] == 0.0), name=f"ind_minus0[{s}]"
        )

        m.addConstr(
            (B_add_s[s] == 1) >> (eth_add_s[s] == g_add_s[s]), name=f"ind_add1[{s}]"
        )
        m.addConstr((B_add_s[s] == 0) >> (eth_add_s[s] == 0.0), name=f"ind_add0[{s}]")

        # gradient = eth_minus_s + eth_add_s
        m.addConstr(
            e[s + 1] - e[s] == eth_minus_s[s] + eth_add_s[s],
            name=f"gradient_balance[{s}]",
        )

    # gradient_linear
    for s in range(M1, S + M1 + 2):
        m.addConstr(
            e[s + 1] + e[s - 1] - 2 * e[s] <= dg_max * pi[s],
            name=f"gradient_linear1[{s}]",
        )
        m.addConstr(
            e[s + 1] + e[s - 1] - 2 * e[s] >= -dg_max * pi[s],
            name=f"gradient_linear2[{s}]",
        )
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

    # save them in _va_variables
    _va_variables = {}
    _va_variables["e"] = e
    _va_variables["pi"] = pi
    _va_variables["B_minus_s"] = B_minus_s
    _va_variables["B_add_s"] = B_add_s
    _va_variables["g_minus_s"] = g_minus_s
    _va_variables["g_add_s"] = g_add_s
    _va_variables["eth_minus_s"] = eth_minus_s
    _va_variables["eth_add_s"] = eth_add_s
    if va_variables is not None:
        va_variables.update(_va_variables)
    
    return _va_variables


def model_rbtc(
    m: gp.Model,
    l2r: bool = True,
    nonlinear_t: bool = False,
    case: dict | None = None,
    e: dict | None = None,
    # set_phi_b: bool = False
) -> dict[str, gp.Var | gp.tupledict]:
    """

    :param gp.Model m: the model instance
    :param bool, optional l2r: short for "left to right"
    :param bool, optional nonlinear_t: use nonlinear expressions for variable t[s] with longer solving time;
                                       if not, the model may be infeasible, but it could also solve faster.
                                       default is False.       
    :param dict, optional case: A dictionary containing case-specific parameters, used to create LC. If None, the
                                case details from the model instance `m` will be used.
    :param dict, optional e: elevation variables, default is None, if not provided, use `m.va_variables["e"]`
    :param bool, optional set_phi_b: default False, if True, set phi_b to always be zero.
    :return: A dictionary containing the variables of the model, including energy, velocity, acceleration, force, and time.

            access with `f"L2R_{varname}"` or `f"R2L_{varname}"`, acceptable `varname` as below:

            [E_hat, v, a, f, b, f_max, b_max, gamma, t, E_s, R_s, Psi_t, phi_n, phi_b, w0, wi, wtn, T]
    :rtype: dict[str, gp.Var | gp.tupledict]

    """
    if case is None:
        case = getattr(m, "case", None)
        assert case is not None, "case parameter is None, and m.case is also None."
    va_variables: dict | None = getattr(m, "va_variables", None)

    # define all the parameters
    S_ = case["S_"]
    g = case["g"]
    train_mass, oesd_mass = (
        case["train"].data["mass"] * 1000,
        case["oesd"].data["mass"] * 1000,
    )
    total_mass = train_mass + oesd_mass
    rho = case["train"].data["rho"]
    M_f, M_b = (
        case["train"].data["max_force"] * 1000,
        case["train"].data["max_force"] * 1000,
    )
    r0, r1, r2 = (
        case["train"].data["davis_a"],
        case["train"].data["davis_b"],
        case["train"].data["davis_c"],
    )
    rtn = case["train"].data["r_tn"]
    a_max = case["train"].data["a_max"]
    v_max = case["train"].data["v_max"]
    eta_b = case["train"].data["eta_b"]
    eta_n = case["train"].data["eta_n"]
    force_ek_pwa = case["train"].data["pwa"]
    v_ek_pwa = case["train"].data["pwa_v_E"]
    one_over_v_ek_pwa = case["train"].data["pwa_1/v_E"]
    T_range = case["T_range"]
    ds = case["ds"]
    mu = case["train"].data["mu"]

    # previous va_variables
    if va_variables is not None and e is None:
        e = va_variables["e"]
    assert isinstance(
        e, (dict, gp.tupledict)
    ), f"e should be either dict or tupledict, now is {type(e)}."

    # prefix added to the names
    dir_prefix: str = "L2R_" if l2r else "R2L_"

    # rbtc_variables
    E_hat = m.addVars(
        range(0, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=v_max**2 / 2,
        name=f"{dir_prefix}E_hat",
    )
    v = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=v_max, name=f"{dir_prefix}v"
    )
    for s in range(1, S_):  # for running intervals, add a small epsilon to lowerbound to avoid numerical issues
        setattr(v[s], "LB", 1e-1)
    
    E_hat_times_2 = m.addVars(
        range(0, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=v_max**2,
        name=f"{dir_prefix}E_hat_times_2",
    )

    a = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=-a_max,
        ub=a_max,
        name=f"{dir_prefix}a",
    )
    f = m.addVars(
        range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f"
    )
    b = m.addVars(
        range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b"
    )
    f_max = m.addVars(
        range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f_max"
    )
    b_max = m.addVars(
        range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b_max"
    )
    gamma = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name=f"{dir_prefix}gamma")

    _t_lb = 0
    _t_ub = T_range[1]

    t = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=_t_lb,
        ub=_t_ub,
        name=f"{dir_prefix}t",
    )

    E_s = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=M_f * ds,
        name=f"{dir_prefix}E_s",
    )
    R_s = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=M_b * ds / eta_b,
        name=f"{dir_prefix}R_s",
    )
    _Psi_t_ub = _t_ub / (
        2 * ds
    ) 
    _Psi_t_lb = 1 / v_max
    Psi_t = m.addVars(
        range(1, S_),
        vtype=GRB.CONTINUOUS,
        lb=_Psi_t_lb,
        ub=_Psi_t_ub,
        name=f"{dir_prefix}Psi_t",
    )
    _phi_n_ub, _phi_b_ub = (M_f * ds + _t_ub * mu) / eta_n, (
        M_f * ds + _t_ub * mu
    ) / eta_b
    phi_n = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_phi_n_ub,
        name=f"{dir_prefix}phi_n",
    )
    phi_b = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_phi_b_ub,
        name=f"{dir_prefix}phi_b",
    )
    _w0_ub = (r0 + r1 * v_max + r2 * v_max**2) * 1000
    w0 = m.addVars(
        range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_w0_ub, name=f"{dir_prefix}w0"
    )
    _wi_ub = case["g_max"] / ds * total_mass * g + 1e-2  # for precision issues
    wi = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=-_wi_ub,
        ub=_wi_ub,
        name=f"{dir_prefix}wi",
    )
    _wtn_ub = rtn * r2 * v_max**2 * 1000
    wtn = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_wtn_ub,
        name=f"{dir_prefix}wtn",
    )
    T = m.addVar(
        vtype=GRB.CONTINUOUS, lb=T_range[0], ub=T_range[1], name=f"{dir_prefix}T"
    )

    # rbtc_constraints
    if l2r:
        # newton's law
        m.addConstrs(
            (a[s] * ds == E_hat[s] - E_hat[s - 1] for s in range(1, S_ + 1)),
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
                w0[s] == 1000 * (r0 + r1 * v[s - 1] + 4 * r2 * E_hat[s - 1])
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}w0",
        )
        m.addConstrs(
            (wtn[s] == 1000 * 4 * rtn * r2 * E_hat[s - 1] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}wtn",
        )

        # velocity to Ek
        m.addConstrs(
            (E_hat_times_2[s] == 2 * E_hat[s] for s in range(0, S_ + 1)),
            name=f"{dir_prefix}E_hat_times2",
        )
        for s in range(0, S_ + 1):
            m.addGenConstrPow(
                E_hat_times_2[s], v[s], a=0.5, name=f"{dir_prefix}v=sqrt(2E_hat)[{s}]"
            )

        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(
                E_hat[s - 1],
                f_max[s],
                force_ek_pwa[0],
                force_ek_pwa[1],
                name=f"{dir_prefix}force_ek_pwa[{s}]",
            )
            m.addGenConstrPWL(
                E_hat[s - 1],
                b_max[s],
                force_ek_pwa[0],
                force_ek_pwa[1],
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
            (E_s[s] == f[s] * ds + mu * t[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}E_s",
        )
        m.addConstrs(
            (E_s[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}eta",
        )
        m.addConstrs(
            (R_s[s] == b[s] * ds / eta_b for s in range(1, S_ + 1)),
            name=f"{dir_prefix}R_s",
        )

        # time related constraints
        if nonlinear_t:
            # nonlinear expressions (longer solving time)
            for s in range(1, S_ + 1):
                m.addQConstr(
                    t[s] * v[s - 1] + t[s] * v[s] == 2 * ds,
                    name=f"{dir_prefix}nonlinear_t[{s}]",
                )
        else:
            # PWA expressions or gurobi embedded methods (might incur infeasibility)
            for s in range(1, S_):
                # m.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
                # Refer to https://www.gurobi.com/documentation/11.0/refman/funcpieces.html for option attributes.
                m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
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
            m.addConstr(E_hat[s] == 0, name=f"{dir_prefix}speed_limit[{s}]")

    else:  # right to left
        # newton's law
        m.addConstrs(
            (a[s] * ds == E_hat[s - 1] - E_hat[s] for s in range(1, S_ + 1)),
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
                w0[s] == 1000 * (r0 + r1 * v[s] + 4 * r2 * E_hat[s])
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}w0",
        )
        m.addConstrs(
            (wtn[s] == 1000 * 4 * rtn * r2 * E_hat[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}wtn",
        )

        # velocity to Ek
        m.addConstrs(
            (E_hat_times_2[s] == 2 * E_hat[s] for s in range(0, S_ + 1)),
            name=f"{dir_prefix}E_hat_times2",
        )
        for s in range(0, S_ + 1):
            m.addGenConstrPow(
                E_hat_times_2[s], v[s], a=0.5, name=f"{dir_prefix}v=sqrt(2E_hat)[{s}]"
            )

        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(
                E_hat[s],
                f_max[s],
                force_ek_pwa[0],
                force_ek_pwa[1],
                name=f"{dir_prefix}force_ek_pwa[{s}]",
            )
            m.addGenConstrPWL(
                E_hat[s],
                b_max[s],
                force_ek_pwa[0],
                force_ek_pwa[1],
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
            (E_s[s] == f[s] * ds + mu * t[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}E_s",
        )
        m.addConstrs(
            (E_s[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
            name=f"{dir_prefix}eta",
        )
        m.addConstrs(
            (R_s[s] == b[s] * ds / eta_b for s in range(1, S_ + 1)),
            name=f"{dir_prefix}R_s",
        )

        # time related constraints
        for s in range(1, S_):
            # model.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
            m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
        m.addConstr(t[1] == 2 * ds * Psi_t[1], name=f"{dir_prefix}t[1]")
        m.addConstr(t[S_] == 2 * ds * Psi_t[S_ - 1], name=f"{dir_prefix}t[{S_}]")
        for s in range(2, S_):
            m.addConstr(
                t[s] == ds / 2 * (Psi_t[s - 1] + Psi_t[s]), name=f"{dir_prefix}t[{s}]"
            )
        m.addConstr(T == t.sum(), name=f"{dir_prefix}T")

        # start and stop velocity
        for s in [0, S_]:
            m.addConstr(E_hat[s] == 0, name=f"{dir_prefix}speed_limit_E_hat[{s}]")
            m.addConstr(v[s] == 0, name=f"{dir_prefix}speed_limit_v[{s}]")

    rbtc_variables: dict[str, gp.Var | gp.tupledict] = {}
    rbtc_variables[f"{dir_prefix}E_hat"] = E_hat
    rbtc_variables[f"{dir_prefix}v"] = v
    rbtc_variables[f"{dir_prefix}E_hat_times_2"] = E_hat_times_2
    rbtc_variables[f"{dir_prefix}a"] = a
    rbtc_variables[f"{dir_prefix}f"] = f
    rbtc_variables[f"{dir_prefix}b"] = b
    rbtc_variables[f"{dir_prefix}f_max"] = f_max
    rbtc_variables[f"{dir_prefix}b_max"] = b_max
    rbtc_variables[f"{dir_prefix}gamma"] = gamma
    rbtc_variables[f"{dir_prefix}t"] = t
    rbtc_variables[f"{dir_prefix}E_s"] = E_s
    rbtc_variables[f"{dir_prefix}R_s"] = R_s
    rbtc_variables[f"{dir_prefix}Psi_t"] = Psi_t
    rbtc_variables[f"{dir_prefix}phi_n"] = phi_n
    rbtc_variables[f"{dir_prefix}phi_b"] = phi_b
    rbtc_variables[f"{dir_prefix}w0"] = w0
    rbtc_variables[f"{dir_prefix}wi"] = wi
    rbtc_variables[f"{dir_prefix}wtn"] = wtn
    rbtc_variables[f"{dir_prefix}T"] = T

    model_rbtc_vars = getattr(m, "rbtc_variables", None)
    if model_rbtc_vars is not None:
        model_rbtc_vars.update(rbtc_variables)

    return rbtc_variables


def model_oesd(
    m: gp.Model,
    varpi_0: float = 1,
    l2r: bool = True,
    case: dict | None = None,
    rbtc_variables: dict | None = None,
) -> dict[str, gp.tupledict]:
    """model OESD

    :param gp.Model m: gurobi model
    :param float varpi_0: initial State-of-Energy, defaults to 1
    :param bool l2r: direction, short for left-to-right, defaults to True
    :param dict | None case: case data dictionary, is used when m is not a VA_RBTC_OESD model, defaults to None
    :param dict | None rbtc_variables: RBTC variables dictionary, is used when m is not a VA_RBTC_OESD model, defaults to None
    :return dict[str, gp.tupledict]: A dictionary containing the variables of the OESD model

            access with `f"L2R_{varname}"` or `f"R2L_{varname}"`, acceptable `varname` as below:

            [varpi, kappa_plus, kappa_minus, xi], when OESD is applied.

            [xi], when OESD type is None.
    """
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) == dict, f"case should be a dict, now {type(case)}."

    if rbtc_variables is None:
        rbtc_variables = getattr(m, "rbtc_variables", None)
    assert (
        rbtc_variables is not None and type(rbtc_variables) == dict
    ), f"rbtc_variables should be a dict, now {type(rbtc_variables)}."

    # define all the parameters
    S_ = case["S_"]

    # prefix added to the names
    dir_prefix: str = "L2R_" if l2r else "R2L_"

    oesd_variables: dict[str, gp.tupledict] = {}

    if case["oesd"].type is None:
        # set xi to always be zero
        xi = m.addVars(
            range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"{dir_prefix}xi"
        )
        # set phi_b to always be zero
        phi_b = rbtc_variables[f"{dir_prefix}phi_b"]
        m.addConstrs(
            (phi_b[s] == 0 for s in range(1, S_ + 1)), name=f"{dir_prefix}phi_b"
        )

        oesd_variables[f"{dir_prefix}xi"] = xi
        if hasattr(m, "oesd_variables"):
            m.oesd_variables.update(oesd_variables)
        return oesd_variables

    XI = case["oesd"].data["capacity"] * 1000 * 3600  # unit is J
    ds = case["ds"]
    a_max = case["train"].data["a_max"]

    # previously defined variables
    phi_b = rbtc_variables[f"{dir_prefix}phi_b"]
    t = rbtc_variables[f"{dir_prefix}t"]
    R_s = rbtc_variables[f"{dir_prefix}R_s"]

    # oesd variables
    varpi = m.addVars(
        range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"{dir_prefix}varpi"
    )
    _max_charge_power, _max_discharge_power = max(
        case["oesd"].data["charge"]["y"]
    ), max(case["oesd"].data["discharge"]["y"])
    kappa_plus = m.addVars(
        range(0, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_max_charge_power,
        name=f"{dir_prefix}kappa_plus",
    )
    kappa_minus = m.addVars(
        range(0, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_max_discharge_power,
        name=f"{dir_prefix}kappa_minus",
    )
    _t_ub = case["T_range"][1] - ds * (S_ - 1) / case["train"].data["v_max"]
    xi = m.addVars(
        range(1, S_ + 1),
        vtype=GRB.CONTINUOUS,
        lb=0,
        ub=_max_charge_power * _t_ub,
        name=f"{dir_prefix}xi",
    )

    # oesd contraints
    # (dis)charging power curve pwa
    for s in range(0, S_ + 1):
        m.addGenConstrPWL(
            varpi[s],
            kappa_plus[s],
            case["oesd"].data["charge"]["x"],
            case["oesd"].data["charge"]["y"],
            name=f"{dir_prefix}kappa_plus[{s}]",
        )
        m.addGenConstrPWL(
            varpi[s],
            kappa_minus[s],
            case["oesd"].data["discharge"]["x"],
            case["oesd"].data["discharge"]["y"],
            name=f"{dir_prefix}kappa_minus[{s}]",
        )

    if l2r:
        for s in range(1, S_ + 1):
            m.addQConstr(
                phi_b[s] <= kappa_minus[s - 1] * t[s],
                name=f"{dir_prefix}QConstr_phi_b[{s}]",
            )
            m.addQConstr(
                xi[s] <= kappa_plus[s - 1] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]"
            )
        m.addConstrs(
            (xi[s] <= R_s[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}xi"
        )
        m.addConstrs(
            (
                varpi[s] == varpi[s - 1] + (xi[s] - phi_b[s]) / XI
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}SOE_change",
        )
        m.addConstr(
            varpi[0] == varpi_0, name=f"{dir_prefix}varpi_0"
        )  # SOE starting state
    else:
        for s in range(1, S_ + 1):
            m.addQConstr(
                phi_b[s] <= kappa_minus[s] * t[s],
                name=f"{dir_prefix}QConstr_phi_b[{s}]",
            )
            m.addQConstr(
                xi[s] <= kappa_plus[s] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]"
            )
        m.addConstrs(
            (xi[s] <= R_s[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}xi"
        )
        m.addConstrs(
            (
                varpi[s - 1] == varpi[s] + (xi[s] - phi_b[s]) / XI
                for s in range(1, S_ + 1)
            ),
            name=f"{dir_prefix}SOE_change",
        )
        m.addConstr(
            varpi[S_] == varpi_0, name=f"{dir_prefix}varpi_0"
        )  # SOE starting state

    # store them in oesd_variables
    oesd_variables[f"{dir_prefix}varpi"] = varpi
    oesd_variables[f"{dir_prefix}kappa_plus"] = kappa_plus
    oesd_variables[f"{dir_prefix}kappa_minus"] = kappa_minus
    oesd_variables[f"{dir_prefix}xi"] = xi

    model_oesd_vars = getattr(m, "oesd_variables", None)
    if type(model_oesd_vars) == dict:
        model_oesd_vars.update(oesd_variables)

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


def model_EC(m: gp.Model, case: dict | None = None, va_variables: dict | None = None, e_lowest_dict:dict|None=None):
    """Add Elevation-based Cuts to the model. (with LB, UB)
    
    BUG: will cut out feasible solutions, leading to infeasible warm start solutions.
    TEMP: add a big threshold 0.1m
    
    """
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) == dict, "case must be a dictionary"
    if va_variables is None:
        va_variables = getattr(m, "va_variables", None)
        assert type(va_variables) == dict, "va_variables must be a dictionary"
    
    # Elevation-based cuts: upper bound, connecting directly two platforms
    M1, S, M2 = case["M1"], case["S"], case["M2"]
    vi_e: dict = {i: 0 for i in range(M1 + 1, S + M1 + 1)}  # to be updated
    left_e, right_e = case["ALEPH1"][M1], case["ALEPH2"][S + M1 + 1]
    grad = (right_e - left_e) / (case["ds"] * (S + 1))
    for i in range(M1 + 1, S + M1 + 1):
        vi_e[i] = grad * case["ds"] * (i - M1) + left_e
        va_variables["e"][i].setAttr("UB", vi_e[i])  # section elevation-based cuts
        
    # Elevation-based cuts: lower bound, solve a quick model to minimize elevations
    LC_threshold = 0.1  # TEMP FIX
    
    if e_lowest_dict is None:
        temp_model, lower_e_LC = solve_edge_va(case, output_flag=0, lowest_edge=True)
        # temp_model, upper_e_LC = solve_edge_va(m.case, output_flag=0, lowest_edge=False)
        temp_model.dispose()  # clear cache
    else:
        lower_e_LC = e_lowest_dict
    
    S_ = case["S_"]
    for i in range(1, S_ + 1):
        # va_variables["e"][i].setAttr("LB", lower_e_LC[i] - LC_threshold)  # will override va_lb_pairs in model_va.
        m.addConstr(va_variables["e"][i] >= lower_e_LC[i] - LC_threshold, name=f"EC_LB_{i}")
        # va_variables["e"][i].setAttr("UB", upper_e_LC[i])
    
    return


def model_SC(m: gp.Model, case: dict | None = None, va_variables: dict | None = None):
    """Add Slope-based Cuts to the model. (with LB, UB)
    """
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) == dict, "case must be a dictionary"
    
    if va_variables is None:
        va_variables = getattr(m, "va_variables", None)
        assert type(va_variables) == dict, "va_variables must be a dictionary"
    
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


def model_TC(m: gp.Model, case: dict | None = None, rbtc_variables: dict | None = None, directions: tuple[bool, bool] | None = None):
    """Add Time-based Cuts to the model. (with LB, UB). `directions` is used when doing one-direction models
    """
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) == dict, "case must be a dictionary"
    if rbtc_variables is None:
        rbtc_variables = getattr(m, "rbtc_variables", None)
        assert type(rbtc_variables) == dict, "rbtc_variables must be a dictionary"
    
    case = case
    ds = case["ds"]
    v_max = case["train"].data["v_max"]
    Tm = case["T_range"][1]
    S_ = case["S_"]
    
    # >>> interval time-based cuts implemented here
    # >>> the time to pass the current interval with v_max.
    _t_lb = ds / v_max  # otherwise would be zero
    # >>> the time to pass the current interval when passing all other intervals with v_max.
    _t_ub = Tm - ds * (S_ - 1) / v_max  # otherwise would be a very large number (T_max).
    
    directions = case['direction'] if directions is None else directions
    if directions[0]:
        dir_prefix = "L2R_"
        for s in range(1, S_ + 1):
            rbtc_variables[f"{dir_prefix}t"][s].setAttr("LB", _t_lb)
            rbtc_variables[f"{dir_prefix}t"][s].setAttr("UB", _t_ub)
    if directions[1]:
        dir_prefix = "R2L_"
        for s in range(1, S_ + 1):
            rbtc_variables[f"{dir_prefix}t"][s].setAttr("LB", _t_lb)
            rbtc_variables[f"{dir_prefix}t"][s].setAttr("UB", _t_ub)
    return


def model_FC(m: gp.Model, case: dict | None = None, rbtc_variables: dict | None = None, directions: tuple[bool, bool] | None = None):
    """Add Force-based Cuts to the model. (with LB, UB). `directions` is used when doing one-direction models
    """
    if case is None:
        case = getattr(m, "case", None)
        assert type(case) == dict, "case must be a dictionary"
    if rbtc_variables is None:
        rbtc_variables = getattr(m, "rbtc_variables", None)
        assert type(rbtc_variables) == dict, "rbtc_variables must be a dictionary"
    
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


def model_VC(m: gp.Model, case:dict | None = None, rbtc_variables: dict | None = None, directions: tuple[bool, bool] | None = None):
    """Add Velocity-based Cuts to the model. (with LB, UB). `directions` is used when doing one-direction models
    """
    if case is None:
        case = getattr(m, "case", None)
        assert isinstance(case, dict), "case must be a dictionary"
    if rbtc_variables is None:
        rbtc_variables = getattr(m, "rbtc_variables", None)
        assert isinstance(rbtc_variables, dict), "rbtc_variables must be a dictionary"
    directions = case['direction'] if directions is None else directions

    S_     = case["S_"]
    ds     = case["ds"]
    v_max  = case["train"].data["v_max"]
    a_max  = case["train"].data["a_max"]
    T_max  = case["T_range"][1]
    L      = S_ * ds
    window_lens: list[int] = [1,2,4,8]  # number of intervals to consider together for a mean speed calculation
    
    # ---- helper: minimal time to traverse a segment of length d with endpoints (v0 -> v1),
    #      under |a|<=a_max, v<=v_max, time-optimal bang-bang (accelerate/cruise/decel).
    #      Closed-form, no decision vars involved.
    def t_min_segment(d: float, v0: float, v1: float) -> float:
        # try to reach v_max; distances for accel/decel
        if v0 < v_max:
            da = (v_max**2 - v0**2) / (2 * a_max)
        else:
            da = 0.0
        if v1 < v_max:
            dd = (v_max**2 - v1**2) / (2 * a_max)
        else:
            dd = 0.0

        if d >= da + dd:
            # accel to v_max, cruise, decel
            t = (max(v_max - v0, 0.0)) / a_max \
                + (max(v_max - v1, 0.0)) / a_max \
                + (d - da - dd) / v_max
            return t
        else:
            # triangular (no cruise): peak speed v_peak <= v_max
            # distance = (v_peak^2 - v0^2)/(2 a_max) + (v_peak^2 - v1^2)/(2 a_max) = d
            v_peak_sq = a_max * d + 0.5 * (v0**2 + v1**2)
            v_peak = (v_peak_sq ** 0.5)
            # ensure not exceeding v_max due to numerics; if > v_max, fallback to the previous branch (shouldn't happen)
            if v_peak > v_max + 1e-9:
                # tiny safeguard
                return (max(v_max - v0, 0.0)) / a_max + (max(v_max - v1, 0.0)) / a_max + max(d - da - dd, 0.0) / v_max
            return max(v_peak - v0, 0.0) / a_max + max(v_peak - v1, 0.0) / a_max

    # ---- helper: compute tight window time upper bound as a constant
    # window [s, s+W-1] uses nodes [s-1 .. s+W-1]
    def t_bar_window_const(s: int, W: int) -> float:
        xL = (s - 1) * ds                 # left boundary position
        xR = (s + W - 1) * ds             # right boundary position
        d_left  = xL                      # distance from start (0) to left boundary
        d_right = L - xR                  # distance from right boundary to end (L)

        # max feasible boundary speeds given distances to ends (start/end at rest)
        v_left_max  = min(v_max, (2 * a_max * d_left) ** 0.5)
        v_right_max = min(v_max, (2 * a_max * d_right) ** 0.5)

        # minimal time outside the window:
        #   left outside: 0 -> v_left_max over distance d_left
        #   right outside: v_right_max -> 0 over distance d_right
        t_left  = t_min_segment(d_left,  0.0,          v_left_max)
        t_right = t_min_segment(d_right, v_right_max,  0.0)

        # the remaining time budget for the window
        t_bar = T_max - (t_left + t_right)
        # it should be > 0; if not, this means T_max itself is too tight for such a window (shouldn't happen in normal setups)
        return t_bar

    def _add_for_dir(prefix: str):
        v = rbtc_variables[f"{prefix}v"]

        # tight sliding-window cuts using t_bar_window
        for W in window_lens:
            for s in range(1, S_ - W + 2):
                t_bar_win = t_bar_window_const(s, W)
                if t_bar_win <= 0:
                    # skip degenerate windows
                    continue
                rhsW = 2.0 * W * ds / t_bar_win
                expr = gp.LinExpr()
                for j in range(s, s + W):
                    expr += v[j - 1] + v[j]
                m.addConstr(expr >= rhsW, name=f"{prefix}vc_tight_W{W}[{s}]")
    if directions[0]:
        _add_for_dir("L2R_")
    if directions[1]:
        _add_for_dir("R2L_")
    return

def model_MS(
    m: gp.Model,
    case: dict | None = None,
    rbtc_variables: dict | None = None,
    directions: tuple[bool, bool] | None = None,
    max_switches: int = 1,
    window_interval: int = 4,
):
    """
    Add Mode-Switching Cuts to reduce unrealistic frequent switching 
    between traction and braking modes.
    Each window of consecutive intervals is limited by `max_switches`.

    :param gp.Model m: gurobipy Model
    :param dict | None case: problem data dict
    :param dict | None rbtc_variables: dict containing gamma binaries for each direction
    :param tuple[bool, bool] | None directions: tuple of booleans (include L2R, include R2L)
    :param int max_switches: maximum allowed switches in a window
    :param int window_interval: window length for switching cut
    """
    assert window_interval >= 3, f"window_interval must be >= 3, but got {window_interval}"
    assert 0<max_switches<=window_interval-2, f"max_switches must be in (0, window_interval-2], but got {max_switches}"
    if case is None:
        case = getattr(m, "case", None)
        assert isinstance(case, dict)
    if rbtc_variables is None:
        rbtc_variables = getattr(m, "rbtc_variables", None)
        assert isinstance(rbtc_variables, dict)
    directions = case['direction'] if directions is None else directions

    S_ = case["S"]

    for dir_prefix, l2r in [("L2R_", True), ("R2L_", False)]:
        if not directions[(0 if l2r else 1)]:
            continue

        gamma = rbtc_variables[f"{dir_prefix}gamma"]

        # # switching indicators: sw_s = |gamma_s - gamma_{s-1}|
        # sw = {}
        # for s in range(2, S_ + 1):
        #     sw[s] = m.addVar(
        #         vtype=GRB.BINARY, name=f"{dir_prefix}sw_{s}"
        #     )
        #     m.addConstr(sw[s] >= gamma[s] - gamma[s - 1])
        #     m.addConstr(sw[s] >= gamma[s - 1] - gamma[s])

        # # add cuts on switching within a sliding window
        # for start in range(2, S_ - window_interval + 2):
        #     m.addConstr(
        #         gp.quicksum(sw[s] for s in range(start, start + window_interval - 1))
        #         <= max_switches,
        #         name=f"{dir_prefix}sw_limit_{start}"
        #     )
        
        # z_s = min(gamma_s, gamma_{s-1})  -> this method does not introduce extra integer variables
        z = {}
        for s in range(2, S_ + 1):
            z[s] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"{dir_prefix}z_{s}")
            m.addConstr(z[s] <= gamma[s], name=f"{dir_prefix}z_le_g_{s}")
            m.addConstr(z[s] <= gamma[s - 1], name=f"{dir_prefix}z_le_gm1_{s}")
            m.addConstr(z[s] >= gamma[s] + gamma[s - 1] - 1.0, name=f"{dir_prefix}z_ge_sum-1_{s}")

        # add cuts on switching within a sliding window
        for start in range(2, S_ - window_interval + 2):
            # sum_{s=start}^{start+W-2} |gamma_s - gamma_{s-1}| <= max_switches
            m.addConstr(
                gp.quicksum((gamma[s] + gamma[s - 1] - 2.0 * z[s])
                            for s in range(start, start + window_interval - 1))
                <= max_switches,
                name=f"{dir_prefix}sw_limit_{start}"
            )

    m.update()

