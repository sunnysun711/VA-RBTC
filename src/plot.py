# no oesd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.model_full import Solution


def plot_compact(case: dict, sol: Solution, l2r: bool = True) -> Figure:
    dir_prefix = "L2R_" if l2r else "R2L_"
    has_oesd = case['oesd'].type is not None

    S_ = case['S_']
    e = np.array([sol.va_sol[f'e[{i}]'] for i in range(1, S_+2)])
    v = np.array([sol.rbtc_sol[f'{dir_prefix}v[{i}]'] for i in range(0, S_+1)])
    f = np.array([sol.rbtc_sol[f'{dir_prefix}f[{i}]'] for i in range(1, S_+1)])
    b = np.array([sol.rbtc_sol[f'{dir_prefix}b[{i}]'] for i in range(1, S_+1)])
    t = np.array([sol.rbtc_sol[f'{dir_prefix}t[{i}]'] for i in range(1, S_+1)])
    t = t[::-1] if dir_prefix == "R2L_" else t
    phi_n = np.array([sol.rbtc_sol[f'{dir_prefix}phi_n[{i}]'] / 3600 for i in range(1, S_+1)])
    phi_n = phi_n[::-1] if dir_prefix == "R2L_" else phi_n
    phi_b = np.array([sol.rbtc_sol[f'{dir_prefix}phi_b[{i}]'] / 3600 for i in range(1, S_+1)])
    phi_b = phi_b[::-1] if dir_prefix == "R2L_" else phi_b
    xi = np.array([sol.oesd_sol[f'{dir_prefix}xi[{i}]'] / 3600 for i in range(1, S_+1)])
    xi = xi[::-1] if dir_prefix == "R2L_" else xi

    cs_t = np.cumsum(t)
    cs_t = cs_t[::-1] if dir_prefix == "R2L_" else cs_t
    cs_phi_n = np.cumsum(phi_n)
    cs_phi_n = cs_phi_n[::-1] if dir_prefix == "R2L_" else cs_phi_n
    cs_phi_b = np.cumsum(phi_b)
    cs_phi_b = cs_phi_b[::-1] if dir_prefix == "R2L_" else cs_phi_b
    cs_xi = np.cumsum(xi)
    cs_xi = cs_xi[::-1] if dir_prefix == "R2L_" else cs_xi
    
    n_rows = 4 if has_oesd else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(5, 2 * n_rows + 2))

    # ---- 1. Elevation & Speed ----
    ax_e = axes[0]
    ax_v = ax_e.twinx()
    ele = ax_e.plot(range(1, S_+1), e[:-1], "ro-", label="Track elevation")
    vel = ax_v.plot([i+0.5 for i in range(S_+1)], v, "b-", label="Speed")
    ax_e.set_ylabel("Track Elevation (m)")
    ax_e.set_ylim((e[0] - 30, e[0] + 5))
    ax_v.set_ylabel("Speed (m/s)")
    lines = ele + vel
    ax_e.legend(lines, [l.get_label() for l in lines], loc="lower center", ncols=2)

    # ---- 2. Forces & Time ----
    ax_f = axes[1]
    ax_t = ax_f.twinx()
    ax_f.plot(range(1, S_+1), f, "rx-", label="Traction force")
    ax_f.plot(range(1, S_+1), -b, "bx-", label="Braking force")
    ax_f.set_ylabel("Force (kN)")
    ax_t.plot(range(1, S_+1), cs_t, color="grey", ls="-", label="Run time")
    ax_t.set_ylabel("Time (s)")
    ax_f.legend(loc="upper center")
    ax_t.legend(loc="lower center")

    # ---- 3. Energy (dis)charge ----
    ax_energy = axes[2]
    ax_energy.set_ylabel("Energy consumption (kWh)")
    ln = ax_energy.plot(range(1, S_+1), cs_phi_n, color="#8ECFC9", ls="-", label="From Net")
    lo = ax_energy.plot(range(1, S_+1), cs_phi_b, color="#FFBE7A", ls="-", label="From OESD")
    lx = ax_energy.plot(range(1, S_+1), cs_xi, color="#FA7F6F", ls="-", label="Charged to OESD")
    ax_energy.legend([*ln, *lo, *lx], [l.get_label() for l in [*ln, *lo, *lx]], loc="center")

    # ---- 4. SOE (only if OESD) ----
    if has_oesd:
        varpi = np.array([sol.oesd_sol[f'{dir_prefix}varpi[{i}]'] for i in range(0, S_+1)])
        ax_soe = axes[3]
        ax_soe.plot([i+0.5 for i in range(0, S_+1)], varpi, color="#82B0D2", marker="x", ls="-", label="SOE")
        ax_soe.set_ylim([-5, 105])
        ax_soe.set_ylabel("State of Energy (%)")
        ax_soe.legend()

    axes[-1].set_xlabel("Intervals")
    plt.tight_layout()
    return fig



if __name__ == "__main__":
    from src.prepare import get_case
    case_id = 10030122
    case = get_case(case_id)
    print(case['quick_text'])
    # sol = Solution.from_file(f"result/pareto/10010022/step019_depth18.31/{case_id}.sol")
    # sol = Solution.from_file(f"result/{case_id}/{case_id}.sol")
    sol = Solution.from_file("result/pareto/10030122/step029_depth19.22/10030122.sol")
    plot_compact(case=case, sol=sol, l2r=True)
    # plot_compact_no_oesd(case=case, sol=sol, l2r=True)
    plt.show()
    pass