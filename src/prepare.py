from typing import Any

import numpy as np

from src import Train
from src import OESD

# case information titles # 10111020
SECTIONS: list[list[tuple[float, float]]] = [
    # section 1
    [(2286.83, 991.2117934),
     (2424.83, 990.9357934),
     (4289.83, 988.6807934),
     (4430.83, 988.9627934), ],
    # section 2
    [(4430.83, 988.9627934),
     (4535.83, 989.1727934),
     (6134.83, 988.4667934),
     (6302.83, 988.1307934), ],
    # section 3
    [(6302.83, 988.1307934),
     (6494.83, 987.7467934),
     (8577.614, 985.0987712),
     (8688.614, 984.8767712), ],
]
TRAINS: list[str] = ["Wu2021Train", "Wu2024Train", "Scheepmaker2020"]
TRAIN_LOADS: list[int] = [0, 3, 6, 9]  # number of passengers per square meter.
OESDS: list[str] = ["None", "supercapacitor", "Li-ion", "flywheel"]
VARPI_0s: list[float] = [0.6, 1.0]
NUM_INTERVALS: list[int] = [25, 50, 100, 103, 200, 500]  # >100就会遇到infeasible问题
DIRECTIONS: list[tuple[bool, bool]] = [(True, False), (False, True), (True, True)]  # (l2r, r2l)
SECTION_TIMES: list[tuple[float, float]] = [(98, 108), (98, 118), (98, 128), (98, 138)]

# technical parameters
I_MIN, I_MAX = 0.002, 0.03  # min max gradient
DELTA_G = 0.018  # max gradient difference
MIN_SLOPE_LEN = 200
WEIGHT_PER_PERSON_KG = 68  # kg


def get_case(case_id: int):
    if case_id < 1e07 or case_id >= 1e08:
        raise ValueError(f"Case {case_id} is out of range")

    # 将整数转换为字符串
    number_str = str(case_id)

    # 提取每一位并转换为整数
    section_id = int(number_str[0])  # section id must be in [1,2,3]
    train_id = int(number_str[1])
    train_load_id = int(number_str[2])
    oesd_id = int(number_str[3])
    varpi_0_id = int(number_str[4])
    S_id = int(number_str[5])
    direction_id = int(number_str[6])
    section_time_id = int(number_str[7])

    # processing raw data
    S1, P1, P2, S2 = SECTIONS[section_id - 1]
    train_type, load = TRAINS[train_id], TRAIN_LOADS[train_load_id]
    TRAIN = Train(train_type, load_spd=load, weight_per_person_kg=WEIGHT_PER_PERSON_KG)
    oesd_type = OESDS[oesd_id]
    OB = OESD(oesd_type)
    S_ = NUM_INTERVALS[S_id]
    varpi_0 = VARPI_0s[varpi_0_id]
    T_range = SECTION_TIMES[section_time_id]
    direction = DIRECTIONS[direction_id]  # l2r or r2l or both

    ds: float = (S2[0] - S1[0]) / S_  # interval length
    g_min, g_max = I_MIN * ds, I_MAX * ds  # min and max elevation difference between consecutive intervals
    dg_max = DELTA_G * ds  # max elevation difference between consecutive slopes
    M1: int = int(np.ceil((P1[0] - S1[0]) / ds))  # number of intervals in station 1 platform
    M2: int = int(np.ceil((S2[0] - P2[0]) / ds))  # number of intervals in station 2 platform
    S: int = S_ - M1 - M2  # number of intervals in section
    ALEPH1: dict[int, float] = {i: (P1[1] - S1[1]) / (P1[0] - S1[0]) * (i - 1) * ds + S1[1] for i in range(1, M1 + 1)}
    ALEPH2: dict[int, float] = {i: S2[1] - (P2[1] - S2[1]) / (P2[0] - S2[0]) * (S_ - i) * ds for i in
                                range(S + M1 + 1, S_ + 1)}
    SIGMA: int = int(np.ceil(MIN_SLOPE_LEN / ds))  # minimum number of intervals within a slope

    # quick info text
    quick_text = (f"========== sec{section_id}_{train_type}_load{load}_"
                  f"{oesd_type}_varpi{varpi_0}_int{S_}_{direction}_T{T_range} ==========")

    case_info_text = (f">>>>>> Initializing VA-RBTC-OESD optimization module... HEAD <<<<<<\n"
                      f"Displaying data:\t case_id: {case_id}\n"
                      f"section_id: {section_id}\t"
                      f"ds: {ds}\t g_min, g_max: {g_min, g_max}\t dg_max: {dg_max}\t "
                      f"M1, S, M2: {M1, S, M2}\t SIGMA: {SIGMA}\n"
                      f"S1, P1, P2, S2: {S1, P1, P2, S2}\n"
                      f"T_range: {T_range}\n"
                      f"ALEPH1: {ALEPH1}\n"
                      f"ALEPH2: {ALEPH2}\n"
                      f"====train data====\t{train_type}\tload: {load}\tdirection: {direction}\tT: {T_range}\n"
                      f"{TRAIN.data}\n"
                      f"====OESD data====\t{oesd_type}\tvarpi_0: {varpi_0}\n"
                      f"{OB.data}\n"
                      f">>>>>> Initializing VA-RBTC-OESD optimization module... TAIL <<<<<<\n")

    case: dict[str, Any] = {
        "case_id": case_id,
        "I_MIN": I_MIN,
        "I_MAX": I_MAX,
        "DELTA_G": DELTA_G,
        "MIN_SLOPE_LEN": MIN_SLOPE_LEN,
        "SECTION_ID": section_id,
        "S_": S_,
        "S1": S1,
        "P1": P1,
        "P2": P2,
        "S2": S2,
        "train": TRAIN,
        "oesd": OB,
        "ds": ds,
        "g_min": g_min,
        "g_max": g_max,
        "dg_max": dg_max,
        "M1": M1,
        "M2": M2,
        "S": S,
        "ALEPH1": ALEPH1,
        "ALEPH2": ALEPH2,
        "SIGMA": SIGMA,
        "T_range": T_range,
        "g": 9.8,  # gravitational acceleration
        "varpi_0": varpi_0,
        "direction": direction,
        "quick_text": quick_text,
        "INFO_TEXT": case_info_text,
    }
    return case


if __name__ == "__main__":
    cs = get_case(case_id=10011020)
    print(cs['quick_text'])
    print(cs["INFO_TEXT"])
