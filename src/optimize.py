import os

import gurobipy as gp
from gurobipy import GRB
from matplotlib import pyplot as plt

from src import get_case


class VA_RBTC_OESD(gp.Model):
    """
    Class for Vertical Alignment - Regenerative Braking Train Control - Onboard Energy Storage Devices (VA-RBTC-OESD).

    This class is designed to model and optimize the vertical alignment of a railway section, incorporating regenerative
    braking train control and onboard energy storage devices.

    Args:
        case_id (int): The ID of the selected case, currently available options are [0, 1, 2].
        save_dir (str, optional): The parent directory for saving results. If None, results are saved in the root folder
         ("result").

    Attributes:
        _name (str): The name of the case, including the save directory if provided.
        _log_file_path (str): The file path for the log file.
        _case (dict): The case details obtained from the `get_case` function.
        _info (str): Quick description text for the case.
        _train (dict): Details of the train configuration for the case.
        _oesd (dict): Details of the onboard energy storage devices configuration for the case.
        _va_variables (dict): Dictionary to store vertical alignment variables.
        _rbtc_variables (dict): Dictionary to store regenerative braking train control variables.
        _oesd_variables (dict): Dictionary to store onboard energy storage devices variables.
    """

    def __init__(self, case_id: int, save_dir: str = None):
        """

        :param case_id:
        :param save_dir: only the parent directory of the saved model.
        """
        super().__init__()
        na = f"{case_id}"
        if save_dir is None:
            self._name: str = na
        else:
            self._name: str = f"{save_dir}\\{na}"
        self._log_file_path = f"..\\result\\{self._name}.log"
        self._case: dict = get_case(case_id)
        self._info: str = self._case["quick_text"]
        self._train = self._case["train"]
        self._oesd = self._case["oesd"]
        self._va_variables: dict = {}
        self._rbtc_variables: dict = {}
        self._oesd_variables: dict = {}
        print(self._info)
        print(self._case['INFO_TEXT'])

    def build(self,
              va_solution: dict = None,
              LC_on: bool = False,
              obj_net: bool = True,
              with_nonlinear_t: bool = False,
              VI_on: bool = False):
        """

        Build the model for vertical alignment, regenerative braking train control, and onboard energy storage devices.

        This method sets up the objective function and constraints for the optimization model, incorporating optional settings
        for linear constraints (LC), objective net (obj_net), nonlinear terms (with_nonlinear_t), and vertical intervention (VI).

        Args:
            va_solution (dict, optional): Predefined solution for vertical alignment variables. Defaults to None.
            LC_on (bool, optional): Flag to enable linear constraints to find the absolute range of elevation. Defaults to False.
            obj_net (bool, optional): Flag to set the objective to minimize the net energy. If False, minimizes total energy. Defaults to True.
            with_nonlinear_t (bool, optional): Flag to include nonlinear terms in the regenerative braking train control model. Defaults to False.
            VI_on (bool, optional): Flag to enable vertical intervention to cut high elevations in middle sections. Defaults to False.

        Returns:
            None
        """
        S_ = self.case['S_']
        model_va(m=self, va_solution=va_solution)
        objExp = gp.LinExpr()

        def _model_one_direction(l2r: bool):
            model_rbtc(self, l2r=l2r, nonlinear_t=with_nonlinear_t)
            model_oesd(self, varpi_0=self.case['varpi_0'], l2r=l2r)

            _pref = "L2R_" if l2r else "R2L_"
            phi_n = self.rbtc_variables[f"{_pref}phi_n"]
            phi_b = self.rbtc_variables[f"{_pref}phi_b"]
            xi = self.oesd_variables[f"{_pref}xi"]
            if obj_net:
                return gp.quicksum(phi_n[s] for s in range(1, S_ + 1))
            else:
                return gp.quicksum(phi_n[s] + phi_b[s] - xi[s] for s in range(1, S_ + 1))

        if self.case['direction'][0]:
            objExp += _model_one_direction(True)
        if self.case['direction'][1]:
            objExp += _model_one_direction(False)

        self.setObjective(objExp, GRB.MINIMIZE)

        if LC_on:
            # the LC is to find absolute range of e. (not very efficient in small cases)
            temp_model = gp.Model("temp model to generate VA-LC")
            e = model_va(temp_model, case=self.case)  # returns the elevation variable
            temp_model.setObjective(gp.quicksum(e), GRB.MINIMIZE)
            temp_model.setParam("OutputFlag", 0)  # no output
            temp_model.setParam("MIPGap", 0)
            temp_model.optimize()
            lower_e_LC = {i: e[i].x for i in e.keys()}

            temp_model.setObjective(gp.quicksum(e), GRB.MAXIMIZE)
            temp_model.setParam("OutputFlag", 0)  # no output
            temp_model.setParam("MIPGap", 0)
            temp_model.optimize()
            upper_e_LC = {i: e[i].x for i in e.keys()}

            self.addConstrs((self.va_variables['e'][i] >= lower_e_LC[i] for i in range(1, S_ + 1)), name="lower_e_LC")
            self.addConstrs((self.va_variables['e'][i] <= upper_e_LC[i] for i in range(1, S_ + 1)), name="upper_e_LC")

        if VI_on:

            # cut too high elevations in middle sections.
            M1, S, M2 = self.case["M1"], self.case['S'], self.case["M2"]
            vi_e: dict = {i: 0 for i in range(M1 + 1, S + M1 + 1)}  # to be updated
            left_e, right_e = self.case["ALEPH1"][M1], self.case["ALEPH2"][S + M1 + 1]
            grad = (right_e - left_e) / (self.case['ds'] * (S + 1))
            for i in range(M1 + 1, S + M1 + 1):
                vi_e[i] = grad * self.case['ds'] * (i - M1) + left_e

            # section elevation-based cuts
            self.addConstrs((self.va_variables['e'][s] <= vi_e[s] for s in range(M1 + 1, S + M1 + 1)), name="VI_e")

            # add strict time constraint
            # max_section_time = self.case['T_range'][1]
            # self.addConstr(self.rbtc_variables['L2R_T'] >= max_section_time - 2, name="VI_L2R_T")
            # self.addConstr(self.rbtc_variables['R2L_T'] >= max_section_time - 2, name="VI_R2L_T")

            # interval time-based cuts
            # already implemented in variable lower and upper bounds

        return

    def build_l2r(self, with_nonlinear_t: bool = False):
        model_va(self)
        model_rbtc(self, nonlinear_t=with_nonlinear_t)
        model_oesd(self, varpi_0=self.case["varpi_0"])

        # parameters
        S_ = self.case['S_']
        phi_n = self.rbtc_variables['L2R_phi_n']
        phi_b = self.rbtc_variables['L2R_phi_b']
        xi = self.oesd_variables['L2R_xi']

        linExp = gp.LinExpr()
        linExp += gp.quicksum(phi_n[s] + phi_b[s] - xi[s] for s in range(1, S_ + 1))
        self.setObjective(linExp, GRB.MINIMIZE)
        return

    def optimize_(self, save_on=True, **kwargs):
        for paramname, param in kwargs.items():
            self.setParam(paramname, param)
        if save_on:
            self.write(f"{self._log_file_path[:-4]}.mps")
            self.write(f"{self._log_file_path[:-4]}.lp")
            self.setParam("LogFile", self._log_file_path)
            with open(self._log_file_path, "w", encoding='utf-8') as f:
                f.write(self.case["quick_text"])
                f.write("\n")
                f.write(self.case["INFO_TEXT"])
        self.optimize()
        if self.Status == GRB.INFEASIBLE:
            if save_on:
                self.setParam("TimeLimit", 5)
                self.computeIIS()
                self.write(f"{self._log_file_path[:-4]}.ilp")
        else:
            if save_on:
                self.write(f"{self._log_file_path[:-4]}.json")  # solution file
                # append into log files
                with open(self._log_file_path, "a") as f:
                    f.write(self.get_detail_results_decorated_txt())
        return

    def get_detail_results_decorated_txt(self):
        txt = ""
        txt += '>>' * 20 + f' DETAIL RESULT ' + '<<' * 20 + '\n'
        txt += '>>' * 10 + f' VARIABLES ' + '<<' * 10 + '\n'
        for var in self.getVars():
            txt += (f"{var.VarName}, VType: {var.VType}, LB: {var.LB}, UB: "
                    f"{var.UB}, ObjCoefficient: {var.Obj}, value: {var.X}\n")
        txt += ">>" * 10 + f' CONSTRAINTS ' + '<<' * 10 + '\n'
        for constr in self.getConstrs():
            expr = self.getRow(constr)  # 获取约束的左侧表达式
            lhs_value = 0  # 初始化左侧表达式的值

            # 遍历表达式中的所有项
            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                lhs_value += var.X * coeff

            txt += (f"{constr.ConstrName} with SLACK {constr.Slack:.4f}: "
                    f"{self.getRow(constr)} = {lhs_value} ||{constr.Sense}=|| {constr.rhs}\n")
        txt += '>>' * 20 + f' DETAIL RESULT ' + '<<' * 20 + '\n'
        return txt

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


def model_va(m: VA_RBTC_OESD | gp.Model, case: dict = None, va_solution: dict = None):
    """
    Build the vertical alignment (VA) model for the given case.

    This function specifies the parameters, variables, and constraints necessary to model the vertical alignment
    of a railway section within the VA-RBTC-OESD framework.

    Args:
        m (VA_RBTC_OESD | gp.Model): The model instance to which the VA constraints and variables will be added.
        case (dict, optional): A dictionary containing case-specific parameters, used to create LC. If None, the
                               case details from the model instance `m` will be used.
        va_solution (dict, optional): A dictionary containing solution variables 'e' with keys from 1 to S_+1.
                                      If provided, only e will be modeled.

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

    Returns:
        If `case` is provided, returns the elevation variables `e`.
        If `case` is None, saves the variables in the model's `_va_variables` attribute.
    """
    # specify parameters
    case = m.case if case is None else case
    S_ = case["S_"]
    g_min, g_max = case["g_min"], case["g_max"]
    M1, S, M2, SIGMA = case["M1"], case["S"], case["M2"], case["SIGMA"]
    dg_max = case["dg_max"]
    ALEPH1, ALEPH2 = case["ALEPH1"], case["ALEPH2"]

    # _va_variables
    e = m.addVars(range(1, S_ + 2), vtype=GRB.CONTINUOUS, name='e')  # extra index (S_+1) is to calculate w_{i, S_}
    pi = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name='pi')

    # va solution mode
    if va_solution is not None:
        m.addConstrs((e[s] == va_solution[s] for s in range(1, S_ + 2)), name="va_solution:e")
        m.va_variables['e'] = e
        return

    # aux _va_variables for gradient range
    B_minus_s = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_minus_s")
    B_add_s = m.addVars(range(0, S_ + 1), vtype=GRB.BINARY, name="B_add_s")
    g_minus_s = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=-g_max, ub=-g_min, name="g_minus_s")
    g_add_s = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=g_min, ub=g_max, name="g_add_s")
    eth_minus_s = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=-g_max, ub=-g_min, name="eth_minus_s")
    eth_add_s = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=g_min, ub=g_max, name="eth_add_s")

    # >>>>>>>>> constraints <<<<<<<<<<
    # extra index for e_s
    m.addConstr(e[S_ + 1] == ALEPH2[S_] + ALEPH2[S_] - ALEPH2[S_ - 1], name="elevation_[S_+1]")
    # slope length
    for i in range(M1, S + M1 - SIGMA + 3):
        m.addConstr(gp.quicksum(pi[i] for i in range(i, i + SIGMA)) <= 1, name=f"slope_length[{i}]")
    # slope gradient
    for s in range(M1, S + M1 + 2):
        # raw
        m.addQConstr(e[s + 1] - e[s] == B_minus_s[s] * g_minus_s[s] + B_add_s[s] * g_add_s[s],
                     name=f"separate_grad_range[{s}]")
        # separate_grad_binary
        m.addConstr(B_minus_s[s] + B_add_s[s] == 1, name=f"separate_grad_binary[{s}]")
    # gradient_linear
    for s in range(M1, S + M1 + 2):
        m.addConstr(e[s + 1] + e[s - 1] - 2 * e[s] <= dg_max * pi[s], name=f"gradient_linear1[{s}]")
        m.addConstr(e[s + 1] + e[s - 1] - 2 * e[s] >= -dg_max * pi[s], name=f"gradient_linear2[{s}]")
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
    if isinstance(m, VA_RBTC_OESD):
        m.va_variables['e'] = e
        m.va_variables['pi'] = pi
        m.va_variables['B_minus_s'] = B_minus_s
        m.va_variables['B_add_s'] = B_add_s
        m.va_variables['g_minus_s'] = g_minus_s
        m.va_variables['g_add_s'] = g_add_s
        m.va_variables['eth_minus_s'] = eth_minus_s
        m.va_variables['eth_add_s'] = eth_add_s
        m.va_variables['eth_minus_s'] = eth_minus_s
        m.va_variables['eth_add_s'] = eth_add_s
        return
    else:
        return e


def model_va_with_solution(m: VA_RBTC_OESD, va_solution: list):
    # specify parameters
    S_ = m.case["S_"]
    g_min, g_max = m.case["g_min"], m.case["g_max"]
    M1, S, M2, SIGMA = m.case["M1"], m.case["S"], m.case["M2"], m.case["SIGMA"]
    dg_max = m.case["dg_max"]
    ALEPH1, ALEPH2 = m.case["ALEPH1"], m.case["ALEPH2"]

    e = m.addVars(range(1, S_ + 2), vtype=GRB.CONTINUOUS, name='e')  # extra index (S_+1) is to calculate w_{i, S_}
    m.addConstrs((e[s] == va_solution[s] for s in range(1, S_ + 2)), name="va_solution:e")
    return


def model_rbtc(m: VA_RBTC_OESD, l2r: bool = True, nonlinear_t: bool = False):
    """

    :param m:
    :param l2r: short for "left to right"
    :param nonlinear_t: use nonlinear expressions for variable t[s] with longer solving time; if not, the model may
        be infeasible, but
    :return:
    """
    # define all the parameters
    S_ = m.case["S_"]
    g = m.case["g"]
    train_mass, oesd_mass = m.train.data['mass'] * 1000, m.oesd.data['mass'] * 1000
    total_mass = train_mass + oesd_mass
    rho = m.train.data['rho']
    M_f, M_b = m.train.data["max_force"] * 1000, m.train.data["max_force"] * 1000
    r0, r1, r2 = m.train.data["davis_a"], m.train.data["davis_b"], m.train.data["davis_c"]
    rtn = m.train.data["r_tn"]
    a_max = m.train.data["a_max"]
    v_max = m.train.data["v_max"]
    eta_b = m.train.data["eta_b"]
    eta_n = m.train.data["eta_n"]
    force_ek_pwa = m.train.data["pwa"]
    v_ek_pwa = m.train.data["pwa_v_E"]
    one_over_v_ek_pwa = m.train.data["pwa_1/v_E"]
    T_range = m.case["T_range"]
    ds = m.case["ds"]
    mu = m.train.data['mu']

    # previous va_variables
    e = m.va_variables['e']

    # prefix added to the names
    dir_prefix: str = "L2R_" if l2r else "R2L_"

    # rbtc_variables
    E_hat = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=v_max ** 2 / 2, name=f"{dir_prefix}E_hat")
    v = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=v_max, name=f"{dir_prefix}v")
    E_hat_times2 = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=v_max ** 2,
                             name=f"{dir_prefix}E_hat_times_2")

    a = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=-a_max, ub=a_max, name=f"{dir_prefix}a")
    f = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f")
    b = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b")
    f_max = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f, name=f"{dir_prefix}f_max")
    b_max = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b, name=f"{dir_prefix}b_max")
    gamma = m.addVars(range(1, S_ + 1), vtype=GRB.BINARY, name=f"{dir_prefix}gamma")

    # >>> interval time-based cuts implemented here
    # >>> the time to pass the current interval with v_max.
    _t_lb = ds / v_max  # otherwise would be zero
    # >>> the time to pass the current interval when passing all other intervals with v_max.
    _t_ub = T_range[1] - ds * (S_ - 1) / v_max  # otherwise would be a very large number (T_max).
    t = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=_t_lb, ub=_t_ub, name=f"{dir_prefix}t")

    E_s = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_f * ds, name=f"{dir_prefix}E_s")
    R_s = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=M_b * ds / eta_b, name=f"{dir_prefix}R_s")
    _Psi_t_ub = _t_ub / (2 * ds)  # 速度从0开始、按照最大加速度一半运行完1个interval时，\frac{1}{\sqrt{2\hat{E}}}的取值。
    _Psi_t_lb = 1 / v_max
    Psi_t = m.addVars(range(1, S_), vtype=GRB.CONTINUOUS, lb=_Psi_t_lb, ub=_Psi_t_ub, name=f"{dir_prefix}Psi_t")
    _phi_n_ub, _phi_b_ub = (M_f * ds + _t_ub * mu) / eta_n, (M_f * ds + _t_ub * mu) / eta_b
    phi_n = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_phi_n_ub, name=f"{dir_prefix}phi_n")
    phi_b = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_phi_b_ub, name=f"{dir_prefix}phi_b")
    _w0_ub = (r0 + r1 * v_max + r2 * v_max ** 2) * 1000
    w0 = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_w0_ub, name=f"{dir_prefix}w0")
    _wi_ub = m.case["g_max"] / ds * total_mass * g
    wi = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=-_wi_ub, ub=_wi_ub, name=f"{dir_prefix}wi")
    _wtn_ub = rtn * r2 * v_max ** 2 * 1000
    wtn = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_wtn_ub, name=f"{dir_prefix}wtn")
    T = m.addVar(vtype=GRB.CONTINUOUS, lb=T_range[0], ub=T_range[1], name=f"{dir_prefix}T")

    # save rbtc_variables
    m.rbtc_variables[f'{dir_prefix}E_hat'] = E_hat
    m.rbtc_variables[f'{dir_prefix}v'] = v
    m.rbtc_variables[f'{dir_prefix}a'] = a
    m.rbtc_variables[f'{dir_prefix}f'] = f
    m.rbtc_variables[f'{dir_prefix}b'] = b
    m.rbtc_variables[f'{dir_prefix}f_max'] = f_max
    m.rbtc_variables[f'{dir_prefix}b_max'] = b_max
    m.rbtc_variables[f'{dir_prefix}gamma'] = gamma
    m.rbtc_variables[f'{dir_prefix}t'] = t
    m.rbtc_variables[f'{dir_prefix}E_s'] = E_s
    m.rbtc_variables[f'{dir_prefix}R_s'] = R_s
    m.rbtc_variables[f'{dir_prefix}Psi_t'] = Psi_t
    m.rbtc_variables[f'{dir_prefix}phi_n'] = phi_n
    m.rbtc_variables[f'{dir_prefix}phi_b'] = phi_b
    m.rbtc_variables[f'{dir_prefix}w0'] = w0
    m.rbtc_variables[f'{dir_prefix}wi'] = wi
    m.rbtc_variables[f'{dir_prefix}wtn'] = wtn
    m.rbtc_variables[f'{dir_prefix}T'] = T

    # rbtc_constraints
    if l2r:
        # newton's law
        m.addConstrs((a[s] * ds == E_hat[s] - E_hat[s - 1] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}2as")
        m.addConstrs((total_mass * (1 + rho) * a[s] == f[s] - b[s] - w0[s] - wi[s] - wtn[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}f=ma")

        # resistances
        m.addConstrs((wi[s] == (e[s + 1] - e[s]) / ds * total_mass * g for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}wi")
        m.addConstrs((w0[s] == 1000 * (r0 + r1 * v[s - 1] + 4 * r2 * E_hat[s - 1]) for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}w0")
        m.addConstrs((wtn[s] == 1000 * 4 * rtn * r2 * E_hat[s - 1] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}wtn")

        # velocity to Ek
        m.addConstrs((E_hat_times2[s] == 2 * E_hat[s] for s in range(0, S_ + 1)),
                     name=f"{dir_prefix}E_hat_times2")
        for s in range(0, S_ + 1):
            m.addGenConstrPow(E_hat_times2[s], v[s], a=0.5, name=f"{dir_prefix}v=sqrt(2E_hat)[{s}]")

        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(E_hat[s - 1], f_max[s], force_ek_pwa[0], force_ek_pwa[1],
                              name=f"{dir_prefix}force_ek_pwa[{s}]")
            m.addGenConstrPWL(E_hat[s - 1], b_max[s], force_ek_pwa[0], force_ek_pwa[1],
                              name=f"{dir_prefix}brake_ek_pwa[{s}]")
        m.addConstrs((f[s] <= f_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_f")
        m.addConstrs((b[s] <= b_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_b")
        m.addConstrs((f[s] <= M_f * gamma[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}f_s")
        m.addConstrs((b[s] <= M_b * (1 - gamma[s]) for s in range(1, S_ + 1)), name=f"{dir_prefix}b_s")

        # energy
        m.addConstrs((E_s[s] == f[s] * ds + mu * t[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}E_s")
        m.addConstrs((E_s[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}eta")
        m.addConstrs((R_s[s] == b[s] * ds / eta_b for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}R_s")

        # time related constraints
        if nonlinear_t:
            # nonlinear expressions (longer solving time)
            for s in range(1, S_ + 1):
                m.addQConstr(t[s] * v[s - 1] + t[s] * v[s] == 2 * ds, name=f"{dir_prefix}nonlinear_t[{s}]")
        else:
            # PWA expressions or gurobi embedded methods (might incur infeasibility)
            for s in range(1, S_):
                # m.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
                # Refer to https://www.gurobi.com/documentation/11.0/refman/funcpieces.html for option attributes.
                m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
            m.addConstr(t[1] == 2 * ds * Psi_t[1], name=f"{dir_prefix}t[1]")
            m.addConstr(t[S_] == 2 * ds * Psi_t[S_ - 1], name=f"{dir_prefix}t[{S_}]")
            for s in range(2, S_):
                m.addConstr(t[s] == ds / 2 * (Psi_t[s - 1] + Psi_t[s]), name=f"{dir_prefix}t[{s}]")
        m.addConstr(T == gp.quicksum(t), name=f"{dir_prefix}T")

        # start and stop velocity
        for s in [0, S_]:
            m.addConstr(E_hat[s] == 0, name=f"{dir_prefix}speed_limit[{s}]")

    else:  # right to left
        # newton's law
        m.addConstrs((a[s] * ds == E_hat[s - 1] - E_hat[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}2as")
        m.addConstrs((total_mass * (1 + rho) * a[s] == f[s] - b[s] - w0[s] - wi[s] - wtn[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}f=ma")

        # resistances
        m.addConstrs((wi[s] == (e[s] - e[s + 1]) / ds * total_mass * g for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}wi")
        m.addConstrs((w0[s] == 1000 * (r0 + r1 * v[s] + 4 * r2 * E_hat[s]) for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}w0")
        m.addConstrs((wtn[s] == 1000 * 4 * rtn * r2 * E_hat[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}wtn")

        # velocity to Ek
        m.addConstrs((E_hat_times2[s] == 2 * E_hat[s] for s in range(0, S_ + 1)),
                     name=f"{dir_prefix}E_hat_times2")
        for s in range(0, S_ + 1):
            m.addGenConstrPow(E_hat_times2[s], v[s], a=0.5, name=f"{dir_prefix}v=sqrt(2E_hat)[{s}]")

        # control forces ranges
        for s in range(1, S_ + 1):
            m.addGenConstrPWL(E_hat[s], f_max[s], force_ek_pwa[0], force_ek_pwa[1],
                              name=f"{dir_prefix}force_ek_pwa[{s}]")
            m.addGenConstrPWL(E_hat[s], b_max[s], force_ek_pwa[0], force_ek_pwa[1],
                              name=f"{dir_prefix}brake_ek_pwa[{s}]")
        m.addConstrs((f[s] <= f_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_f")
        m.addConstrs((b[s] <= b_max[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}Psi_b")
        m.addConstrs((f[s] <= M_f * gamma[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}f_s")
        m.addConstrs((b[s] <= M_b * (1 - gamma[s]) for s in range(1, S_ + 1)), name=f"{dir_prefix}b_s")

        # energy
        m.addConstrs((E_s[s] == f[s] * ds + mu * t[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}E_s")
        m.addConstrs((E_s[s] == eta_b * phi_b[s] + eta_n * phi_n[s] for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}eta")
        m.addConstrs((R_s[s] == b[s] * ds / eta_b for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}R_s")

        # time related constraints
        for s in range(1, S_):
            # model.addGenConstrPWL(E_hat[s], Psi_t[s], one_over_v_ek_pwa[0], one_over_v_ek_pwa[1], name=f"Psi_t[{s}]")
            m.addGenConstrPow(v[s], Psi_t[s], a=-1, name=f"{dir_prefix}Psi_t[{s}]")
        m.addConstr(t[1] == 2 * ds * Psi_t[1], name=f"{dir_prefix}t[1]")
        m.addConstr(t[S_] == 2 * ds * Psi_t[S_ - 1], name=f"{dir_prefix}t[{S_}]")
        for s in range(2, S_):
            m.addConstr(t[s] == ds / 2 * (Psi_t[s - 1] + Psi_t[s]), name=f"{dir_prefix}t[{s}]")
        m.addConstr(T == gp.quicksum(t), name=f"{dir_prefix}T")

        # start and stop velocity
        for s in [0, S_]:
            m.addConstr(E_hat[s] == 0, name=f"{dir_prefix}speed_limit[{s}]")
    return


def model_oesd(m: VA_RBTC_OESD, varpi_0: float = 1, l2r: bool = True):
    # define all the parameters
    S_ = m.case["S_"]

    # prefix added to the names
    dir_prefix: str = "L2R_" if l2r else "R2L_"

    if m.oesd.type is None:
        # set xi to always be zero
        xi = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"{dir_prefix}xi")
        # set phi_b to always be zero
        phi_b = m.rbtc_variables[f'{dir_prefix}phi_b']
        m.addConstrs((phi_b[s] == 0 for s in range(1, S_ + 1)), name=f"{dir_prefix}phi_b")

        m.oesd_variables[f'{dir_prefix}xi'] = xi
        return

    XI = m.oesd.data["capacity"] * 1000 * 3600  # unit is J
    ds = m.case["ds"]
    a_max = m.train.data["a_max"]

    # previously defined variables
    phi_b = m.rbtc_variables[f'{dir_prefix}phi_b']
    t = m.rbtc_variables[f'{dir_prefix}t']
    R_s = m.rbtc_variables[f'{dir_prefix}R_s']

    # oesd variables
    varpi = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"{dir_prefix}varpi")
    _max_charge_power, _max_discharge_power = max(m.oesd.data['charge']['y']), max(
        m.oesd.data['discharge']['y'])
    kappa_plus = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_charge_power,
                           name=f"{dir_prefix}kappa_plus")
    kappa_minus = m.addVars(range(0, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_discharge_power,
                            name=f"{dir_prefix}kappa_minus")
    _t_ub = (2 * ds / (a_max / 2)) ** 0.5  # 速度从0开始、按照最大加速度的一半运行完1个子区间的时间
    xi = m.addVars(range(1, S_ + 1), vtype=GRB.CONTINUOUS, lb=0, ub=_max_charge_power * _t_ub,
                   name=f"{dir_prefix}xi")

    # store them in oesd_variables
    m.oesd_variables[f"{dir_prefix}varpi"] = varpi
    m.oesd_variables[f"{dir_prefix}kappa_plus"] = kappa_plus
    m.oesd_variables[f"{dir_prefix}kappa_minus"] = kappa_minus
    m.oesd_variables[f"{dir_prefix}xi"] = xi

    # oesd contraints
    # (dis)charging power curve pwa
    for s in range(0, S_ + 1):
        m.addGenConstrPWL(
            varpi[s], kappa_plus[s], m.oesd.data['charge']['x'], m.oesd.data['charge']['y'],
            name=f"{dir_prefix}kappa_plus[{s}]")
        m.addGenConstrPWL(
            varpi[s], kappa_minus[s], m.oesd.data['discharge']['x'], m.oesd.data['discharge']['y'],
            name=f"{dir_prefix}kappa_minus[{s}]")

    if l2r:
        for s in range(1, S_ + 1):
            m.addQConstr(phi_b[s] <= kappa_minus[s - 1] * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
            m.addQConstr(xi[s] <= kappa_plus[s - 1] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
        m.addConstrs((xi[s] <= R_s[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}xi")
        m.addConstrs((varpi[s] == varpi[s - 1] + (xi[s] - phi_b[s]) / XI for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}SOE_change")
        m.addConstr(varpi[0] == varpi_0, name=f"{dir_prefix}varpi_0")  # SOE starting state
    else:
        for s in range(1, S_ + 1):
            m.addQConstr(phi_b[s] <= kappa_minus[s] * t[s], name=f"{dir_prefix}QConstr_phi_b[{s}]")
            m.addQConstr(xi[s] <= kappa_plus[s] * t[s], name=f"{dir_prefix}QConstr_xi[{s}]")
        m.addConstrs((xi[s] <= R_s[s] for s in range(1, S_ + 1)), name=f"{dir_prefix}xi")
        m.addConstrs((varpi[s - 1] == varpi[s] + (xi[s] - phi_b[s]) / XI for s in range(1, S_ + 1)),
                     name=f"{dir_prefix}SOE_change")
        m.addConstr(varpi[S_] == varpi_0, name=f"{dir_prefix}varpi_0")  # SOE starting state

    return


def main():
    pass


if __name__ == '__main__':
    main()
