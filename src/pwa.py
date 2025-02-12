# this file uses the package from:
# https://jekel.me/piecewise_linear_fit_py/examples.html#force-a-fit-through-data-points

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pwlf
# from GPyOpt.methods import BayesianOptimization

from src import Train, OESD

matplotlib.use("TkAgg")


def piecewise_linear_approx(x, y, n_segments: int, fix_x: list = None, fix_y: list = None, bounds: list = None,
                            fitfast: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a continuous piecewise linear function for a specified number of line segments. 
    Uses differential evolution to finds the optimum location of breakpoints for a given number of line segments by minimizing the sum of the square error.

    Args:
        x (list | np.ndarray): original data x points
        y (list | np.ndarray): original data y points
        n_segments (int): number of segments. if n_segments = 4, there will be 5 points to return.
        fix_x (list, optional): must-go-through x points. Defaults to None.
        fix_y (list, optional): must-go-through y points. Defaults to None.
        bounds (list, optional): must-go-through bounds. Defaults to None.
        fitfast (bool, optional): whether to fit fast. Defaults to False.
    Returns:
        tuple[np.ndarray, np.ndarray]: x_hat points, y_hat points.
    """
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    if fitfast:
        x_hat = my_pwlf.fitfast(n_segments=n_segments, bounds=bounds)
    else:
        x_hat = my_pwlf.fit(n_segments=n_segments, x_c=fix_x, y_c=fix_y, bounds=bounds)
    y_hat = my_pwlf.predict(x_hat)

    return x_hat, y_hat


def plot_approximation_with_previous_studies(
        x: list | np.ndarray, y: list | np.ndarray, n_segments: int,
        fix_x: list = None, fix_y: list = None,
        bounds: list = None, fitfast: bool = False,
        prev_x: list | np.ndarray = None, prev_y: list | np.ndarray = None,
        xlabel: str = r"$\hat{E} = 0.5 v^2$ (m$^2$/s$^2$)", ylabel: str = "Maximum forces (N)",
        plot_name: str = "test.pdf"
):
    if n_segments != 0:
        # approximate
        x_hat, y_hat = piecewise_linear_approx(
            x=x, y=y, n_segments=n_segments, fix_x=fix_x, fix_y=fix_y, bounds=bounds, fitfast=fitfast)
        print("Approximation\n", x_hat.tolist(), "\n", y_hat.tolist())
    if prev_x is not None:
        print("Previous studies\n", prev_x, "\n", prev_y)

    # 绘制结果
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Original Function')
    if n_segments != 0:
        plt.plot(x_hat, y_hat, 'ro-', label='Piecewise Linear Approximation')
    if prev_x is not None:
        # plt.plot(prev_x, prev_y, 'bx-', label='Approximation from previous studies')
        plt.plot(prev_x, prev_y, 'ro-', label='Piecewise Linear Approximation')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    fig = plt.gcf()

    plt.show()
    if input("Save plot? (y/n):\n") == "y":
        fig.savefig(fr"..\\pic\\{plot_name}", dpi=600)
    return


def get_flywheel() -> tuple[np.ndarray, np.ndarray]:
    # get first segment
    x1 = np.arange(0, 0.25, 0.0001)
    y1 = 1000 * np.sqrt(x1)

    # get second segment
    x2 = np.arange(0.25, 1, 0.0001)
    y2 = np.ones_like(x2) * 500

    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    return x, y


def get_force_curve(max_force: float, max_power: float, ek_max: float = 800) -> tuple[np.ndarray, np.ndarray]:
    # ek = 0.5 * v ** 2
    # solve the stage changing point
    v_change = max_power / max_force
    ek_change = 0.5 * v_change ** 2

    # get constant force stage
    x1 = np.arange(0, ek_change, step=0.1)
    y1 = np.ones_like(x1) * max_force

    # get constant power stage
    x2 = np.arange(ek_change, ek_max, step=0.1)
    y2 = max_power / np.sqrt(x2 * 2)

    x = np.hstack((x1, x2))
    y = np.hstack((y1, y2))
    return x, y


def plot_force_curve(tn_name: str = "Wu2021Train"):
    train = Train(tn_name)
    x, y = get_force_curve(
        max_force=train.data['max_force'] * 1000,
        max_power=train.data['max_power'] * 1000,
        ek_max=train.data['v_max'] ** 2 / 2
    )
    fix_x = [(train.data['max_power'] / train.data['max_force']) ** 2 / 2]
    fix_y = [train.data['max_force'] * 1000]
    print(fix_x, fix_y)
    plot_approximation_with_previous_studies(
        x=x, y=y, n_segments=4,
        fix_x=fix_x, fix_y=fix_y,
        plot_name=f"{tn_name}_force_Ek_PWA.pdf")


def plot_v_ek():
    v = np.arange(0, 40, 0.001)
    ek = v ** 2 / 2
    fix_x, fix_y = [0], [0]
    print(fix_x, fix_y)
    plot_approximation_with_previous_studies(
        x=ek, y=v, n_segments=4,
        fix_x=fix_x, fix_y=fix_y,
        ylabel="v (m/s)", plot_name="v_Ek_PWA.pdf"
    )
    return


def plot_1_over_v_ek():
    v = np.arange(0.001, 40.002, 0.001)
    ek = v ** 2 / 2
    fix_x, fix_y = [800], [0]
    # plot_approximation_with_previous_studies(
    #     x=ek, y=1 / v, n_segments=10,
    #     fix_x=fix_x, fix_y=fix_y,
    #     # bounds=[[5e-05, 0.003], [0.003, 1], [1, 250]],
    #     fitfast=True,
    #     ylabel=r"$\frac{1}{v}$ (s/m)",
    #     plot_name="1_over_v_Ek_PWA.pdf"
    # )
    plot_approximation_with_previous_studies(
        x=ek, y=1 / v, n_segments=0,
        prev_x=[
            5e-07,
            0.00015205,
            4.7,
            28,
            280.134,
            800
        ], prev_y=[
            1000,
            1.2353,
            0.26,
            0.1283,
            0.0395741061587183,
            0.02506459819129511
        ],
        ylabel=r"$\frac{1}{v}$ (s/m)",
        plot_name="1_over_v_Ek_PWA(detail).svg"
    )


def plot_supercapacitor():
    soe = np.arange(0, 1, 0.001)
    charge_power = soe * -750 + 750
    discharge_power = soe * 750
    sc = OESD(type_="supercapacitor")
    x1 = np.array(sc.data["charge"]["x"])
    y1 = np.array(sc.data["charge"]["y"]) / 1000
    x2 = np.array(sc.data["discharge"]["x"])
    y2 = np.array(sc.data["discharge"]["y"]) / 1000

    # 绘制结果
    plt.figure(figsize=(8, 4))
    plt.plot(soe, charge_power, label='Empirical Function for Charge')
    plt.plot(soe, discharge_power, label='Empirical Function for Discharge')

    plt.plot(x1, y1, "rx--", label="Piecewise Linear Approximation for Charge")
    plt.plot(x2, y2, "bo--", label="Piecewise Linear Approximation for Discharge")

    plt.legend()
    plt.xlabel("SOE (State of Energy)")
    plt.ylabel("Maximum power (kW)")
    plt.tight_layout()

    fig = plt.gcf()

    plt.show()
    if input("Save plot? (y/n):\n") == "y":
        fig.savefig(fr"..\\pic\\supercapacitor_PWA.pdf", dpi=600)
    return


def plot_flywheel():
    soe, power = get_flywheel()

    sc = OESD(type_="flywheel")
    x1 = np.array(sc.data["charge"]["x"])
    y1 = np.array(sc.data["charge"]["y"]) / 1000
    x2 = np.array(sc.data["discharge"]["x"])
    y2 = np.array(sc.data["discharge"]["y"]) / 1000

    # 绘制结果
    plt.figure(figsize=(8, 4))
    plt.plot(soe, power, label='Empirical Function for (Dis)Charge')
    # plt.plot(soe, power, label='Empirical Function for Discharge')

    plt.plot(x1, y1, "rx--", label="Piecewise Linear Approximation for (Dis)Charge")
    # plt.plot(x2, y2, "bo--", label="Piecewise Linear Approximation for Discharge")

    plt.legend()
    plt.xlabel("SOE (State of Energy)")
    plt.ylabel("Maximum power (kW)")
    plt.tight_layout()

    fig = plt.gcf()

    plt.show()
    if input("Save plot? (y/n):\n") == "y":
        fig.savefig(fr"..\\pic\\flywheel_PWA.pdf", dpi=600)
    return


def get_li() -> list[tuple[np.ndarray, np.ndarray]]:
    # todo 直接增加锂电池的充放电功率在这里。return [(soe, charge_power), (soe, discharge_power)]
    soe = np.arange(0, 100, 0.001)

    # y = -0.000000028417 x5 + 0.000005110909 x4 - 0.000340207363 x3 + 0.006393458682 x2 - 0.295010688554 x + 80
    # y = 0.000000004364 x5 - 0.000002596987 x4 + 0.000481189875 x3 - 0.041314899569 x2 + 2.276333465059 x
    charge_power = (-0.000000028417 * soe ** 5
                    + 0.000005110909 * soe ** 4
                    - 0.000340207363 * soe ** 3
                    + 0.006393458682 * soe ** 2
                    - 0.295010688554 * soe
                    + 80.00000000)
    discharge_power = (0.000000004364 * soe ** 5
                       - 0.000002596987 * soe ** 4
                       + 0.000481189875 * soe ** 3
                       - 0.041314899569 * soe ** 2
                       + 2.276333465059 * soe)
    return [(soe / 100, charge_power), (soe / 100, discharge_power)]
    pass


def plot_li():
    (soe, charge_power), (soe, discharge_power) = get_li()

    sc = OESD(type_="Li-ion")
    x1 = np.array(sc.data["charge"]["x"])
    y1 = np.array(sc.data["charge"]["y"]) / 1000
    x2 = np.array(sc.data["discharge"]["x"])
    y2 = np.array(sc.data["discharge"]["y"]) / 1000

    # 绘制结果
    plt.figure(figsize=(8, 4))
    plt.plot(soe, charge_power, label='Empirical Function for Charge')
    plt.plot(soe, discharge_power, label='Empirical Function for Discharge')

    plt.plot(x1, y1, "rx--", label="Piecewise Linear Approximation for Charge")
    plt.plot(x2, y2, "bo--", label="Piecewise Linear Approximation for Discharge")

    plt.legend()
    plt.xlabel("SOE (State of Energy)")
    plt.ylabel("Maximum power (kW)")
    plt.tight_layout()

    fig = plt.gcf()

    plt.show()
    if input("Save plot? (y/n):\n") == "y":
        fig.savefig(fr"..\\pic\\Li-ion_PWA.pdf", dpi=600)
    return


if __name__ == "__main__":
    # plot_force_curve("Scheepmaker2020")
    # plot_v_ek()
    # plot_1_over_v_ek()
    # plot_flywheel()
    plot_li()
    pass
