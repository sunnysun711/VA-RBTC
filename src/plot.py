import numpy as np
from matplotlib import pyplot as plt

from src import VA_RBTC_OESD, anl_results, anl_variables


def plot_track_elevation(model: VA_RBTC_OESD):
    e = np.array([model.va_variables['e'][i].x for i in model.va_variables['e'].keys()])
    print(e)
    plt.plot(e, "ro-")
    plt.xlabel("Intervals")
    plt.ylabel("Track Elevation (m)")
    plt.ylim((e[0] - 30, e[0] + 5))
    plt.show()


def plot_track_elevation_compares_two_level(
        case_id_groups: list[list[int]],
        label_groups: list[list[str]],
        marker_groups: list[list[str]],
        ls_groups: list[list[str]],
        save_name: str = "track_ele_compare.pdf",
        folder: str = "..\\result\\0617",
):
    plt.figure(figsize=(6, 4), facecolor="white")
    for case_group, label_group, marker_group, ls_group in zip(case_id_groups, label_groups, marker_groups, ls_groups):
        for case_id, label, marker, ls in zip(case_group, label_group, marker_group, ls_group):
            case_results = anl_results(case_id, folder)
            print(case_id, "MIPGap:\t", case_results['SolutionInfo']["MIPGap"])
            e: dict = anl_variables(variable='e', opt_results=case_results)
            plt.plot(
                list(e.keys()), list(e.values()),
                marker=marker,
                linestyle=ls,
                label=label
            )
    plt.legend()
    plt.xlabel("Intervals")
    plt.ylabel("Track Elevation (m)")
    plt.ylim((e[1] - 30, e[1] + 5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if input(f"save plots in {folder}\\{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"\\{save_name}", dpi=600)
    return


def plot_track_elevation_compares(
        case_ids: list, labels: list, save_name: str = "track_ele_compare.pdf",
        folder: str = "..\\result\\0617", marker="x", ls="-"
):
    plt.figure(figsize=(6, 4), facecolor='white')
    for case_id, label in zip(case_ids, labels):
        case_results = anl_results(case_id, folder)
        print(case_id, "MIPGap:\t", case_results["SolutionInfo"]["MIPGap"])
        e: dict = anl_variables(variable="e", opt_results=case_results)
        plt.plot(
            list(e.keys()), list(e.values()),
            marker=marker,
            linestyle=ls,
            label=label
        )
    plt.legend()
    plt.xlabel("Intervals")
    plt.ylabel("Track Elevation (m)")
    plt.ylim((e[1] - 30, e[1] + 5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if input(f"save plots in {folder}\\{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"\\{save_name}", dpi=600)
    return


def plot_track_elevation_compares_with_speed(
        case_ids: list, labels: list, save_name: str = "track_ele_compares_with_speed.pdf",
        folder: str = "..\\result\\0617", marker="x", ls="-"
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white")
    ax_v = ax.twinx()
    ax.set_ylabel("Track Elevation (m)")
    ax.set_xlabel("Intervals")
    ax_v.set_ylabel("Velocity (m/s)")
    lines = []
    for case_id, label in zip(case_ids, labels):
        case_results = anl_results(case_id, folder)
        print(case_id, "MIPGap:\t", case_results['SolutionInfo']['MIPGap'])
        e: dict = anl_variables(variable="e", opt_results=case_results)
        v_l2r: dict = anl_variables(variable="L2R_v", opt_results=case_results)
        v_r2l: dict = anl_variables(variable="R2L_v", opt_results=case_results)
        v_indices = [i + 0.5 for i in v_l2r.keys()]

        ele = ax.plot(
            list(e.keys()), list(e.values()),
            marker=marker,
            linestyle=ls,
            label=label
        )
        v1 = ax_v.plot(
            v_indices, list(v_l2r.values()),
            marker=">",
            markersize=4,
            linestyle="-.",
            label=label + "_v_L2R",
            color=ele[0].get_color()
        )
        v2 = ax_v.plot(
            v_indices, list(v_r2l.values()),
            marker="<",
            markersize=4,
            linestyle="-.",
            label=label + "_v_R2L",
            color=ele[0].get_color()
        )
        lines.append(ele[0])
        lines.append(v1[0])
        lines.append(v2[0])
    ax.legend(lines, [ll.get_label() for ll in lines], fontsize="small", loc="center", bbox_to_anchor=(0.5, 0.55))
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if input(f"save plots in {folder}\\{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"\\{save_name}", dpi=600)
    return


def plot_compact_no_oesd(mo: VA_RBTC_OESD, l2r: bool = True, show_plot: bool = False) -> plt.Figure:
    S_ = mo.case['S_']

    dir_prefix = "L2R_" if l2r else "R2L_"
    e = np.array([mo.va_variables['e'][i].x for i in mo.va_variables['e'].keys()])
    v = np.array([mo.rbtc_variables[f'{dir_prefix}v'][i].x for i in mo.rbtc_variables[f'{dir_prefix}v'].keys()])
    f = np.array([mo.rbtc_variables[f'{dir_prefix}f'][i].x for i in mo.rbtc_variables[f'{dir_prefix}f'].keys()])
    b = np.array([mo.rbtc_variables[f'{dir_prefix}b'][i].x for i in mo.rbtc_variables[f'{dir_prefix}b'].keys()])
    t = np.array([mo.rbtc_variables[f'{dir_prefix}t'][i].x for i in mo.rbtc_variables[f'{dir_prefix}t'].keys()])
    cumsum_t = np.cumsum(t)
    phi_n = np.array(
        [mo.rbtc_variables[f'{dir_prefix}phi_n'][i].x for i in mo.rbtc_variables[f'{dir_prefix}phi_n'].keys()])

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
    tra = ax_f.plot(range(1, S_ + 1), f / 1000, "rx-", label="Traction force")
    bra = ax_f.plot(range(1, S_ + 1), -b / 1000, "bx-", label="Braking force")
    # control = ax_f.plot(range(1, S_ + 1), (f - b) / 1000, "rx-", label="Control force")
    ax_f.set_ylabel("Force (kN)")
    cumsum_t = cumsum_t if l2r else cumsum_t[::-1]
    cst = ax_t.plot(range(1, S_ + 1), cumsum_t, color="grey", ls="-", label="Cumulative time")
    ax_t.set_ylabel("Time (s)")
    lines = tra + bra + cst
    ax_f.legend(lines, [ll.get_label() for ll in lines])

    # third subplot: (dis)charge
    ax_energy = axes[2]
    ax_energy.set_ylabel("Energy consumption (kWh)")
    ln = ax_energy.plot(range(1, S_ + 1), phi_n / 1000 / 3600, color="#8ECFC9", marker="", ls="-",
                        label="Energy discharged from Net")
    lines = ln
    ax_energy.legend(lines, [ll.get_label() for ll in lines])

    # fourth subplot: SOE
    ax_soe = axes[3]
    ls = ax_soe.plot([i + 0.5 for i in range(0, S_ + 1)], [0 for i in range(0, S_ + 1)], color="#82B0D2", marker="x",
                     ls="-", label="State of Energy (SOE)")
    ax_soe.set_ylim([-0.05, 1.05])
    ax_soe.set_ylabel("State of Energy")
    ax_soe.set_xlabel("Intervals")
    ax_soe.legend()

    fig = plt.gcf()
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


def plot_compact(mo: VA_RBTC_OESD, l2r: bool = True, show_plot: bool = False) -> plt.Figure:
    S_ = mo.case['S_']

    dir_prefix = "L2R_" if l2r else "R2L_"
    e = np.array([mo.va_variables['e'][i].x for i in mo.va_variables['e'].keys()])
    v = np.array([mo.rbtc_variables[f'{dir_prefix}v'][i].x for i in mo.rbtc_variables[f'{dir_prefix}v'].keys()])
    f = np.array([mo.rbtc_variables[f'{dir_prefix}f'][i].x for i in mo.rbtc_variables[f'{dir_prefix}f'].keys()])
    b = np.array([mo.rbtc_variables[f'{dir_prefix}b'][i].x for i in mo.rbtc_variables[f'{dir_prefix}b'].keys()])
    t = np.array([mo.rbtc_variables[f'{dir_prefix}t'][i].x for i in mo.rbtc_variables[f'{dir_prefix}t'].keys()])
    cumsum_t = np.cumsum(t)
    phi_n = np.array(
        [mo.rbtc_variables[f'{dir_prefix}phi_n'][i].x for i in mo.rbtc_variables[f'{dir_prefix}phi_n'].keys()])
    phi_b = np.array(
        [mo.rbtc_variables[f'{dir_prefix}phi_b'][i].x for i in mo.rbtc_variables[f'{dir_prefix}phi_b'].keys()])
    xi = np.array([mo.oesd_variables[f'{dir_prefix}xi'][i].x for i in mo.oesd_variables[f'{dir_prefix}xi'].keys()])
    varpi = np.array(
        [mo.oesd_variables[f'{dir_prefix}varpi'][i].x for i in mo.oesd_variables[f'{dir_prefix}varpi'].keys()])
    R_s = np.array([mo.rbtc_variables[f'{dir_prefix}R_s'][i].x for i in mo.rbtc_variables[f'{dir_prefix}R_s'].keys()])

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
    tra = ax_f.plot(range(1, S_ + 1), f / 1000, "rx-", label="Traction force")
    bra = ax_f.plot(range(1, S_ + 1), -b / 1000, "bx-", label="Braking force")
    # control = ax_f.plot(range(1, S_ + 1), (f - b) / 1000, "rx-", label="Control force")
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
    ln = ax_energy.plot(range(1, S_ + 1), phi_n / 1000 / 3600, color="#8ECFC9", marker="", ls="-",
                        label="Energy discharged from Net")
    lo = ax_energy.plot(range(1, S_ + 1), phi_b / 1000 / 3600, color="#FFBE7A", marker="", ls="-",
                        label="Energy discharged from OESD")
    lx = ax_energy.plot(range(1, S_ + 1), xi / 1000 / 3600, color="#FA7F6F", ls="-", marker="",
                        label="Energy charged to OESD")
    lines = ln + lo + lx
    ax_energy.legend(lines, [ll.get_label() for ll in lines])

    # fourth subplot: SOE
    ax_soe = axes[3]
    ls = ax_soe.plot([i + 0.5 for i in range(0, S_ + 1)], varpi, color="#82B0D2", marker="x", ls="-",
                     label="State of Energy (SOE)")
    ax_soe.set_ylim([-0.05, 1.05])
    ax_soe.set_ylabel("State of Energy")
    ax_soe.set_xlabel("Intervals")
    ax_soe.legend()

    fig = plt.gcf()
    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig
