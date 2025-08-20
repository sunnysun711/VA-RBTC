import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.analyze import anl_results, anl_variables
from src.optimize import VA_RBTC_OESD



def plot_track_elevation(model: VA_RBTC_OESD):
    """
    Plot the track elevation for the given model.

    This function extracts the track elevation values from the model's
    'va_variables' dictionary (using key 'e') and plots them with red markers
    connected by lines.

    The y-axis limits are set relative to the initial elevation.

    :param model: An instance of the VA_RBTC_OESD model containing
            the variable 'e' representing track elevations.
    :return:
    """
    e = np.array([model.va_variables['e'][i].x for i in model.va_variables['e'].keys()])
    print(e)
    plt.plot(e, "ro-")
    plt.xlabel("Intervals")
    plt.ylabel("Track Elevation (m)")
    plt.ylim((e[0] - 30, e[0] + 5))
    plt.show()
    return


def plot_track_elevation_compares_two_level(
        case_id_groups: list[list[int]],
        label_groups: list[list[str]],
        marker_groups: list[list[str]],
        ls_groups: list[list[str]],
        save_name: str = "track_ele_compare.pdf",
        folder: str = "result/0617",
):
    """
    Plot track elevation comparisons for multiple groups of cases.

    This function creates a comparative plot of track elevation profiles for different cases organized in groups.
    For each case, it retrieves the analysis results and extracts the track elevation data.
    Each case is plotted using the specified markers, line styles, and labels.
    After plotting, the figure is displayed and the user is prompted to save the figure.

    :param case_id_groups: A list where each element is a list of case IDs to be compared within a group.
    :param label_groups: A list where each element is a list of labels corresponding to the case IDs in
        `case_id_groups`.
    :param marker_groups: A list where each element is a list of marker styles for plotting each case.
    :param ls_groups: A list where each element is a list of line styles for plotting each case.
    :param save_name: Optional. The filename for saving the plot.
        Default to "track_ele_compare.pdf".
    :param folder: Optional. The directory where result files are stored and where the plot will be saved.
        Default to "result/0617".
    :return:
    """
    plt.figure(figsize=(6, 4), facecolor="white")
    for case_group, label_group, marker_group, ls_group in zip(case_id_groups, label_groups, marker_groups, ls_groups):
        for case_id, label, marker, ls in zip(case_group, label_group, marker_group, ls_group):
            case_results: dict = anl_results(case_id, folder)
            print(case_id, "MIPGap:\t", case_results['SolutionInfo']["MIPGap"])
            e: dict[int, float] = anl_variables(variable='e', opt_results=case_results)
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
    if input(f"save plots in {folder}/{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"/{save_name}", dpi=600)
    return


def plot_track_elevation_compares(
        case_ids: list, labels: list, save_name: str = "track_ele_compare.pdf",
        folder: str = "result/0617", marker="x", ls="-"
):
    """
    Plot track elevation comparisons for multiple cases.

    This function creates a plot comparing track elevation profiles for a list of cases.
    For each case, the analysis results are retrieved and the track elevation data is extracted
    and plotted using the specified marker and line style. The y-axis limits are set relative
    to the second elevation value. After plotting, the figure is displayed and the user is prompted
    to save the plot.

    :param case_ids: List of case IDs to compare.
    :param labels: List of labels for each case.
    :param save_name: The filename for saving the plot.
        Defaults to "track_ele_compare.pdf".
    :param folder: The directory where result files are stored and where the plot will be saved.
        Defaults to "result/0617".
    :param marker: Marker style for the plot. Defaults to "x".
    :param ls: Line style for the plot. Defaults to "-".

    :return:
    """
    plt.figure(figsize=(6, 4), facecolor='white')
    for case_id, label in zip(case_ids, labels):
        case_results = anl_results(case_id, folder)
        print(case_id, "MIPGap:\t", case_results["SolutionInfo"]["MIPGap"])
        e: dict = anl_variables(variable="e", opt_results=case_results)
        plt.plot(
            list(e.keys()), list(e.values()),
            marker=marker,
            linestyle=ls,
            label=label,
            alpha=0.8,
        )
    plt.legend()
    plt.xlabel("Intervals")
    plt.ylabel("Track Elevation (m)")
    plt.ylim((e[1] - 30, e[1] + 5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if input(f"save plots in {folder}/{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"/{save_name}", dpi=600)
    return


def plot_track_elevation_compares_with_speed(
        case_ids: list, labels: list, save_name: str = "track_ele_compares_with_speed.pdf",
        folder: str = "result/0617", marker="x", ls="-"
):
    """
    Plot track elevation and speed comparisons for multiple cases.

    This function creates a dual-axis plot where the primary y-axis shows the track
    elevation and the secondary y-axis displays the corresponding speeds (for both left-to-right
    and right-to-left directions). For each case, the analysis results are retrieved and both
    the elevation and speed data are extracted and plotted. Speed data is offset by 0.5 to align
    with the intervals. The figure is displayed and the user is prompted to save it.

    :param case_ids: List of case IDs to compare.
    :param labels: List of labels for each case.
    :param save_name: The filename for saving the plot.
        Defaults to "track_ele_compares_with_speed.pdf".
    :param folder: The directory where result files are stored and where the plot will be saved.
        Defaults to "result/0617".
    :param marker: Marker style for the track elevation plot.
            Defaults to "x".
    :param ls: Line style for the track elevation plot.
        Defaults to "-".
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor="white")
    ax_v = ax.twinx()
    ax.set_ylabel("Track Elevation (m)")
    ax.set_xlabel("Intervals")
    ax_v.set_ylabel("Velocity (m/s)")
    lines = []
    for case_id, label in zip(case_ids, labels):
        case_results: dict = anl_results(case_id, folder)
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
            markersize=1,
            linestyle="-.",
            label=label + "_v_L2R",
            color=ele[0].get_color()
        )
        v2 = ax_v.plot(
            v_indices, list(v_r2l.values()),
            marker="<",
            markersize=1,
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
    if input(f"save plots in {folder}/{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"/{save_name}", dpi=600)
    return


def plot_track_elevation_compares_with_speed2(
        case_ids: list, labels: list, save_name: str = "track_ele_compares_with_speed.pdf",
        folder: str = "result/0617", marker="x", ls="-"
):
    """
    Plot track elevation and speed comparisons for multiple cases.

    This function creates a dual-axis plot where the primary y-axis shows the track
    elevation and the secondary y-axis displays the corresponding speeds (for both left-to-right
    and right-to-left directions). For each case, the analysis results are retrieved and both
    the elevation and speed data are extracted and plotted. Speed data is offset by 0.5 to align
    with the intervals. The figure is displayed and the user is prompted to save it.

    :param case_ids: List of case IDs to compare.
    :param labels: List of labels for each case.
    :param save_name: The filename for saving the plot.
        Defaults to "track_ele_compares_with_speed.pdf".
    :param folder: The directory where result files are stored and where the plot will be saved.
        Defaults to "result/0617".
    :param marker: Marker style for the track elevation plot.
            Defaults to "x".
    :param ls: Line style for the track elevation plot.
        Defaults to "-".
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), facecolor="white", sharex=True)
    ax = axes[0]
    ax_v = axes[1]
    ax.set_ylabel("Track Elevation (m)")
    ax_v.set_xlabel("Intervals")
    ax_v.set_ylabel("Velocity (m/s)")
    lines = []
    for case_id, label in zip(case_ids, labels):
        case_results: dict = anl_results(case_id, folder)
        print(case_id, "MIPGap:\t", case_results['SolutionInfo']['MIPGap'])
        e: dict = anl_variables(variable="e", opt_results=case_results)
        v_l2r: dict = anl_variables(variable="L2R_v", opt_results=case_results)
        v_r2l: dict = anl_variables(variable="R2L_v", opt_results=case_results)
        v_indices = [i + 0.5 for i in v_l2r.keys()]

        ele = ax.plot(
            list(e.keys()), list(e.values()),
            marker=marker,
            linestyle=ls,
            label=label,
            alpha=0.5
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
    # ax.legend(lines, [ll.get_label() for ll in lines], fontsize="small", loc="center", bbox_to_anchor=(0.5, 0.55))
    ax.legend()
    ax_v.legend(ncols=2)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    if input(f"save plots in {folder}/{save_name}? (y/n)\n") == "y":
        fig.savefig(folder + f"/{save_name}", dpi=600)
    return


def plot_compact_no_oesd(mo: VA_RBTC_OESD, l2r: bool = True, show_plot: bool = False) -> plt.Figure:
    """
    Generate a compact multi-panel plot for the given model (excluding OESD variables).

    This function creates a figure with four subplots that display:
      1. Track elevation and speed.
      2. Control forces (traction and braking) along with cumulative time.
      3. Energy consumption from the network.
      4. A placeholder State of Energy (SOE) plot (all zeros).

    The plotting direction (left-to-right or right-to-left) is determined by the 'l2r' flag.
    After constructing the plots, the figure is returned and optionally displayed.

    :param mo: An instance of the VA_RBTC_OESD model containing the necessary variables.
    :param l2r: If True, use left-to-right variables; otherwise, use right-to-left.
        Defaults to True.
    :param show_plot: If True, display the plot interactively.
        Defaults to False.
    :return: matplotlib.figure.Figure: The figure object containing the multi-panel plot.
    """
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
    """
    Generate a compact multi-panel plot for the given model including OESD variables.

    This function creates a figure with four subplots that display:
      1. Track elevation and speed.
      2. Control forces (traction and braking) along with cumulative time.
      3. Energy flows: energy discharged from the network, energy discharged from OESD,
         and energy charged to OESD.
      4. State of Energy (SOE) from OESD.

    The plotting direction (left-to-right or right-to-left) is determined by the 'l2r' flag.
    After constructing the plots, the figure is returned and optionally displayed.

    :param mo: An instance of the VA_RBTC_OESD model containing the necessary variables.
    :param l2r: If True, use left-to-right variables; otherwise, use right-to-left.
        Defaults to True.
    :param show_plot: If True, display the plot interactively.
        Defaults to False.
    :return: matplotlib.figure.Figure: The figure object containing the multi-panel plot.
    """
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


def plot_converge(log_file: str, legend_label: str, show_plot: bool = False):
    """plot the convergence process of one case"""
    pass


def plot_scatter_runtimes(all_case_info_file: str = "result/0617/all_case_info.csv", ):
    """scatter the runtimes of all cases inside the csv file"""
    cols = ["case_id", "train", "oesd", "varpi_0", "S_", "section_time", "status", "runtime", "mipgap"]
    df = pd.read_csv(all_case_info_file, index_col=0)[cols]
    print(df.shape)
    df = df[df['train'] == "Wu2021Train"]
    print(df.shape)
    df = df[~((df['oesd'] is None) & (df['varpi_0'] == 0.6))]
    print(df[df['oesd'] is None])
    print(df.shape)
    pass


def main():
    plot_scatter_runtimes()
    pass


if __name__ == '__main__':
    main()
