# -*- coding: utf-8 -*-
"""
三阶段高效数据集生成脚本尾部增强版（3 到 7 层，主体 Steel01 + 阻尼器 Steel02，周期约束）

设计目标
--------
1. 先按地震波特征划分 train / val / test 波池，而不是高保真算完再事后切分；
2. 使用分层 LHS（Latin hypercube sampling）做初始结构空间覆盖；
3. 采用三阶段生成：
   - Stage A: 空间填充初始设计
   - Stage B: 基于 committee disagreement 的主动补样
   - Stage C: 面向高层间位移角尾部区域的强化补样
4. 对 train / val / test 都增加高位移尾部补样，避免测试集缺少 0.01 / 0.02 附近样本；
5. 一次 OpenSees 仿真同时落盘：
   raw drift / 量化 drift / 阻尼器屈服标志 / 周期合规信息 / 失败原因 / 波特征；
6. 流式写盘，支持重复运行时按 sample_id 跳过已完成样本。

尾部增强版说明
------------
本版本继承轻量版的离散采样档位与 Steel02 阻尼器固定绝对值屈服力范围，并修正旧数据集暴露出的三个问题：
- 训练集中 `max_drift_ratio_raw >= 0.02` 样本过少；
- 验证集 / 测试集可能没有 0.02 水平样本，导致尾部泛化无法评估；
- 屈服样本统计需要同时区分主体 `steel01_yielded` 与阻尼器 `steel02_yielded`。

为此新增：
- `stage_d_tail_topup`：训练集尾部补样；
- `val_tail_topup` / `test_tail_topup`：验证集和测试集尾部补样；
- summary 中的阈值计数、分位数、主体/阻尼器屈服计数。

采样空间仍保持轻量版压缩后的离散档位：
- 楼层质量、层高、主体刚度候选值各压缩为 6 档；
- Steel02 阻尼器屈服力在 500 到 2500 kN 范围内取 5 档离散值。

参考文献（与脚本设计直接相关）
--------------------------
1. McKay, Beckman, Conover. "A Comparison of Three Methods for Selecting Values
   of Input Variables in the Analysis of Output from a Computer Code."
   Technometrics, 1979. DOI: 10.2307/1268522
2. Sinha, Malo, Deb, Sen. "Review of recent trends in design of computer experiments."
   Computer & Chemical Engineering, 2017.
   DOI: 10.1016/j.compchemeng.2017.05.010
3. Wu, Guo, Gao. "Efficient space-filling and near-orthogonality sequential Latin
   hypercube designs for computer experiments." Computer Methods in Applied
   Mechanics and Engineering, 2018. DOI: 10.1016/j.cma.2017.05.020
4. Khoshnevis, Hu, Bostanabad, et al. "Application of pool-based active learning
   in reducing the number of required response history analyses."
   Computer Methods in Applied Mechanics and Engineering, 2020.
   DOI: 10.1016/j.cma.2020.112925
5. Moustapha, Marelli, Sudret. "Active learning for structural reliability:
   Survey, general framework and benchmark." Structural Safety, 2022.
   DOI: 10.1016/j.strusafe.2021.102174
6. Zhang, Wang, Sun, et al. "Physics-guided convolutional neural network for
   data-driven seismic response modeling." Engineering Structures, 2020.
   DOI: 10.1016/j.engstruct.2020.110704
7. Aschheim, M. "Seismic Design Based on the Yield Displacement."
   Earthquake Spectra, 2002, 18(4): 581-600.
   DOI: 10.1193/1.1516754
8. FEMA P-58-1, Seismic Performance Assessment of Buildings, Volume 1:
   Methodology, Second Edition. FEMA / ATC / USRC.
   https://www.usrc.org/wp-content/uploads/FEMA_P-58-1-SE_Volume1_Methodology.pdf

说明
----
单位约定
--------
本脚本采用统一的 `kN-m-s-t` 单位体系：
- 力：`kN`
- 长度：`m`
- 时间：`s`
- 质量：`t`
- 加速度：地震波文件与 `wave_*` 强度特征按 `g` 存储，输入 OpenSees 时再乘 `9.81 m/s^2`
- 层间位移角、阻尼比、硬化比等比值量：无量纲

模型变更
--------
1. 主体结构由原先的 Elastic 层间弹簧改为 Steel01 层间弹簧；
2. 附加阻尼器保持为 Steel02，并把楼层安装位置作为离散参数参与采样；
3. 对原结构（不含附加阻尼器）增加合规性检查：一阶自振周期 T1 不得大于 0.9 s；
4. 结构采样阶段先做周期筛选，只有合规结构才进入高保真时程分析。

附加阻尼器强度采样
----------------
`Steel02` 的屈服强度 `Fy_add` 采用固定绝对值范围采样。
当前采用：

    500 kN <= Fy_add <= 2500 kN

并在该范围内按几何级数生成 5 个离散候选值。

主体 Steel01 屈服参数与刚度的关系
------------------------------
本脚本不再把主体结构的 Steel01 屈服力 `Fy_story` 当成独立随机量，而是按

    Fy_story = K_story * Delta_y_story
             = K_story * h_story * theta_y

来确定，其中：
- `K_story` 为层刚度；
- `h_story` 为层高；
- `theta_y` 为主体结构的层屈服漂移角。

采用这个关系的理由是：
1. Aschheim (2002) 指出，对规则钢框架而言，屈服位移 / 屈服漂移比周期更稳定，
   适合作为设计与建模的一阶参数；
2. FEMA P-58-1 明确指出，对 moment frame systems，yield drift ratio 应与
   构件达到预期塑性弯矩承载力时对应的 story drift 相联系；
3. 因此在当前简化的剪切型集中参数模型里，用恒定的主体屈服漂移角 `theta_y`
   来反推出 Steel01 的 `Fy_story`，比把 `Fy_story` 与刚度完全解耦更合理。

本脚本默认取：

    theta_y = 1.0% = 0.01

这对应钢矩形框架常见的屈服漂移量级；于是层刚度越大，主体 Steel01 屈服力也越大，
满足“刚度增大，屈服力随之增大”的约束。
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import openseespy.opensees as ops
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


HYPERPARAM_PATH = Path(__file__).with_name(f"{Path(__file__).stem}.hyperparams.json")


def load_hyperparameters(hyperparam_path: Path) -> dict:
    if not hyperparam_path.exists():
        raise FileNotFoundError(f"超参数文件不存在: {hyperparam_path}")
    with hyperparam_path.open("r", encoding="utf-8") as file:
        return json.load(file)


HYPERPARAMS = load_hyperparameters(HYPERPARAM_PATH)


def _hp_entry(section: str, name: str) -> dict:
    return HYPERPARAMS[section][name]


def _coerce_json_number(value: object) -> object:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "inf":
            return float("inf")
        if lowered == "-inf":
            return float("-inf")
    return value


def _dtype_from_name(dtype_name: str):
    return np.int64 if dtype_name == "int" else np.float64


def _hp_scalar(section: str, name: str):
    return _coerce_json_number(_hp_entry(section, name)["value"])


def _hp_array(section: str, name: str) -> np.ndarray:
    build = _hp_entry(section, name)["build"]
    build_type = str(build["type"])
    dtype = _dtype_from_name(str(build.get("dtype", "float")))
    if build_type == "arange":
        return np.arange(
            _coerce_json_number(build["start"]),
            _coerce_json_number(build["stop"]),
            _coerce_json_number(build["step"]),
            dtype=dtype,
        )
    if build_type == "array":
        values = [_coerce_json_number(value) for value in build["values"]]
        return np.asarray(values, dtype=dtype)
    raise ValueError(f"不支持的数组构造方式: section={section}, name={name}, type={build_type}")


def _scaled_int_from_base(base_value: int | float, scale: float) -> int:
    return max(1, int(math.ceil(float(base_value) * float(scale))))


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Config:
    HYPERPARAM_PATH = HYPERPARAM_PATH
    SCRIPT_DIR = Path(__file__).resolve().parent
    RAW_DATA_DIR = Path("C:/Users/12200/Origin-earthquake-file").parent if Path("C:/Users/12200/Origin-earthquake-file").exists() else SCRIPT_DIR
    PROJECT_ROOT = Path("C:/Users/12200/project/Dategeneral")
    CSV_DIR = _first_existing_path(
        PROJECT_ROOT / "CSV-dataset",
        PROJECT_ROOT / "数据集",
    )
    if not CSV_DIR.exists():
        CSV_DIR = PROJECT_ROOT / "CSV-dataset"
        CSV_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = _first_existing_path(
        Path("C:/Users/12200/project/Dategeneral/Newdatabase/rare_waves_scaled.h5"),
        RAW_DATA_DIR / "Origin-earthquake-file" / "6000-uniform-scale-0.1-1.0",
        RAW_DATA_DIR / "Origin-earthquake-file" / "6000",
    )
    IMAGE_DIR = _first_existing_path(
        RAW_DATA_DIR / "Scalogram" / "6000-uniform-scale-0.1-1.0",
        RAW_DATA_DIR / "Scalogram" / "6000",
    )

    RUN_TAG_PREFIX = "3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0"
    RUN_TAG = os.environ.get(
        "SURMOD_RUN_TAG",
        f"{RUN_TAG_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    DATASET_PREFIX = "opensees_surrogate_dataset_floors_3_to_7_"
    OUTPUT_BASENAME = f"{DATASET_PREFIX}{RUN_TAG}"

    TRAIN_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_train.csv"
    VAL_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_val.csv"
    TEST_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_test.csv"
    WAVE_FEATURE_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_wave_features.csv"
    WAVE_SPLIT_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_wave_split.csv"
    SUMMARY_OUTPUT = CSV_DIR / f"{OUTPUT_BASENAME}_summary.json"

    NUM_FLOOR_OPTIONS = _hp_array("grid", "num_floor_options")
    FLOOR_MASS_VALUES = _hp_array("grid", "floor_mass_values")
    FLOOR_HEIGHT_VALUES = _hp_array("grid", "floor_height_values")
    FRAME_STIFFNESS_SCALE = float(_hp_scalar("grid", "frame_stiffness_scale"))
    K_BASE_VALUES = _hp_array("grid", "k_base_grid_unscaled") * FRAME_STIFFNESS_SCALE

    TOP_FLOOR_FACTOR = float(_hp_scalar("grid", "top_floor_factor"))
    FRAME_B = float(_hp_scalar("frame_model", "frame_b"))
    FRAME_YIELD_DRIFT_RATIO = float(_hp_scalar("frame_model", "frame_yield_drift_ratio"))
    PERIOD_LIMIT_SEC = float(_hp_scalar("frame_model", "period_limit_sec"))
    K_ADD = float(_hp_scalar("damper_model", "k_add"))
    B = float(_hp_scalar("damper_model", "steel02_b"))
    R0 = float(_hp_scalar("damper_model", "r0"))
    CR1 = float(_hp_scalar("damper_model", "cr1"))
    CR2 = float(_hp_scalar("damper_model", "cr2"))
    DAMPING_RATIO = float(_hp_scalar("frame_model", "damping_ratio"))
    N_EIGEN = int(_hp_scalar("frame_model", "n_eigen"))
    DT = float(_hp_scalar("time_history", "dt"))
    NUM_STEPS = int(_hp_scalar("time_history", "num_steps"))
    _use_wave_time_params_default = "1" if bool(_hp_scalar("time_history", "use_wave_time_params_default")) else "0"
    USE_WAVE_TIME_PARAMS = os.environ.get("SURMOD_USE_WAVE_TIME_PARAMS", _use_wave_time_params_default).strip().lower() not in {
        "0",
        "false",
        "no",
    }
    ACCELERATION_FACTOR = float(_hp_scalar("time_history", "acceleration_factor"))

    FY_ADD_MIN_KN = float(_hp_scalar("damper_model", "fy_add_min_kn"))
    FY_ADD_MAX_KN = float(_hp_scalar("damper_model", "fy_add_max_kn"))
    FY_ADD_NUM_OPTIONS = int(_hp_scalar("damper_model", "fy_add_num_options"))

    TRAIN_WAVE_RATIO = float(_hp_scalar("wave_split", "train_wave_ratio"))
    VAL_WAVE_RATIO = float(_hp_scalar("wave_split", "val_wave_ratio"))
    TEST_WAVE_RATIO = float(_hp_scalar("wave_split", "test_wave_ratio"))
    WAVE_CLUSTER_COUNT = int(_hp_scalar("wave_split", "wave_cluster_count"))

    DATASET_SIZE_SCALE = float(
        os.environ.get(
            "SURMOD_DATASET_SIZE_SCALE",
            str(_hp_scalar("sampling_budget", "dataset_size_scale_default")),
        )
    )

    STAGE_A_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "stage_a_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    STAGE_A_WAVES_PER_STRUCTURE = int(_hp_scalar("sampling_budget", "stage_a_waves_per_structure"))

    STAGE_B_ROUNDS = int(_hp_scalar("sampling_budget", "stage_b_rounds"))
    STAGE_B_CANDIDATE_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "stage_b_candidate_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    STAGE_B_WAVES_PER_STRUCTURE = int(_hp_scalar("sampling_budget", "stage_b_waves_per_structure"))
    STAGE_B_SELECT_PER_ROUND = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "stage_b_select_per_round_base"),
        DATASET_SIZE_SCALE,
    )

    STAGE_C_CANDIDATE_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "stage_c_candidate_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    STAGE_C_WAVES_PER_STRUCTURE = int(_hp_scalar("sampling_budget", "stage_c_waves_per_structure"))
    STAGE_C_SELECT_TOTAL = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "stage_c_select_total_base"),
        DATASET_SIZE_SCALE,
    )
    STAGE_C_STRONG_WAVE_TOP_RATIO = float(_hp_scalar("sampling_budget", "stage_c_strong_wave_top_ratio"))

    VAL_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "val_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    TEST_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "test_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    HOLDOUT_WAVES_PER_STRUCTURE = int(_hp_scalar("sampling_budget", "holdout_waves_per_structure"))
    TAIL_TOPUP_CANDIDATE_STRUCTURES_PER_FLOOR = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "tail_topup_candidate_structures_per_floor_base"),
        DATASET_SIZE_SCALE,
    )
    TAIL_TOPUP_WAVES_PER_STRUCTURE = int(_hp_scalar("sampling_budget", "tail_topup_waves_per_structure"))
    TAIL_TOPUP_TRAIN_SELECT_PER_ROUND = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "tail_topup_train_select_per_round_base"),
        DATASET_SIZE_SCALE,
    )
    TAIL_TOPUP_HOLDOUT_SELECT_PER_ROUND = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "tail_topup_holdout_select_per_round_base"),
        DATASET_SIZE_SCALE,
    )
    TAIL_TOPUP_TEST_SELECT_PER_ROUND = _scaled_int_from_base(
        _hp_scalar("sampling_budget", "test_tail_topup_select_per_round_base"),
        DATASET_SIZE_SCALE,
    )
    TAIL_TOPUP_TRAIN_MAX_ROUNDS = int(_hp_scalar("sampling_budget", "tail_topup_train_max_rounds"))
    TAIL_TOPUP_HOLDOUT_MAX_ROUNDS = int(_hp_scalar("sampling_budget", "tail_topup_holdout_max_rounds"))
    TAIL_STRONG_WAVE_TOP_RATIO = float(_hp_scalar("sampling_budget", "tail_strong_wave_top_ratio"))
    TAIL_MIN_PER_FLOOR_RATIO = float(_hp_scalar("sampling_budget", "tail_min_per_floor_ratio"))
    TRAIN_MIN_DRIFT_GE_010 = int(_hp_scalar("sampling_budget", "train_min_drift_ge_010"))
    TRAIN_MIN_DRIFT_GE_020 = int(_hp_scalar("sampling_budget", "train_min_drift_ge_020"))
    TRAIN_MIN_STEEL01_YIELDED = int(_hp_scalar("sampling_budget", "train_min_steel01_yielded"))
    HOLDOUT_MIN_DRIFT_GE_010 = int(_hp_scalar("sampling_budget", "holdout_min_drift_ge_010"))
    HOLDOUT_MIN_DRIFT_GE_020 = int(_hp_scalar("sampling_budget", "holdout_min_drift_ge_020"))
    HOLDOUT_MIN_STEEL01_YIELDED = int(_hp_scalar("sampling_budget", "holdout_min_steel01_yielded"))

    ACTIVE_TARGET_SCALE = float(_hp_scalar("postprocess", "active_target_scale"))
    DRIFT_QUANTIZATION_STEP = float(_hp_scalar("postprocess", "drift_quantization_step"))
    DRIFT_BINS = _hp_array("postprocess", "drift_bins")

    RANDOM_SEED = int(_hp_scalar("runtime", "random_seed"))
    _num_workers_mode = str(_hp_scalar("runtime", "num_workers_mode"))
    if _num_workers_mode == "auto_cpu_count":
        NUM_WORKERS = max(1, multiprocessing.cpu_count())
    else:
        NUM_WORKERS = max(1, int(_hp_scalar("runtime", "num_workers_value")))
    WRITE_FLUSH_EVERY = int(_hp_scalar("runtime", "write_flush_every"))

    # Output column units:
    # floor_mass[t], floor_height[m], k_base_1_4[kN/m], k_add[kN/m], Fy_add[kN],
    # damper_layout is a per-story installation flag string such as "(0,0,1)".
    # period_1_sec[s], period_limit_sec[s], wave_dt[s], wave_pga[g], wave_rms[g],
    # wave_mean_abs[g], wave_cav[g*s], wave_arias_proxy[g^2*s],
    # wave_duration_5_95[s], wave_dominant_freq[Hz], wave_spectral_centroid[Hz],
    # wave_predominant_period[s], max_drift_ratio_*[-], steel01_yielded[-],
    # steel02_yielded[-], peak_damper_force[kN].
    OUTPUT_COLUMNS = [
        "sample_id",
        "split",
        "stage",
        "round_idx",
        "txt_path",
        "image_path",
        "num_floors",
        "floor_mass",
        "floor_height",
        "k_base_1_4",
        "k_add",
        "Fy_add",
        "damper_layout",
        "frame_b",
        "frame_yield_drift_ratio",
        "period_1_sec",
        "period_limit_sec",
        "period_compliant",
        "period_check_status",
        "wave_cluster",
        "wave_dt",
        "wave_length",
        "wave_pga",
        "wave_rms",
        "wave_mean_abs",
        "wave_cav",
        "wave_arias_proxy",
        "wave_duration_5_95",
        "wave_zero_crossing_rate",
        "wave_dominant_freq",
        "wave_spectral_centroid",
        "wave_predominant_period",
        "wave_intensity_score",
        "max_drift_ratio_raw",
        "max_drift_ratio_q1e4",
        "steel01_yielded",
        "steel02_yielded",
        "peak_damper_force",
        "yield_margin",
        "peak_story_id",
        "analysis_status",
        "failure_reason",
        "failure_category",
        "analysis_return_code",
        "analysis_failed_step",
        "analysis_failed_time",
        "nonconvergence_flag",
    ]


STRUCTURE_PERIOD_CACHE: dict[tuple, dict] = {}


def build_fy_add_values_for_structure(_k_base_1_4: float, _floor_height: float) -> np.ndarray:
    fy_add_min = float(Config.FY_ADD_MIN_KN)
    fy_add_max = float(Config.FY_ADD_MAX_KN)

    fy_add_min = max(fy_add_min, 1.0e-6)
    fy_add_max = max(fy_add_max, fy_add_min)
    if Config.FY_ADD_NUM_OPTIONS <= 1 or math.isclose(fy_add_min, fy_add_max, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return np.asarray([fy_add_max], dtype=float)
    return np.geomspace(fy_add_min, fy_add_max, Config.FY_ADD_NUM_OPTIONS, dtype=float)


def build_floor_masses(floor_mass: float, num_floors: int) -> list[float]:
    masses = [floor_mass] * num_floors
    masses[-1] = floor_mass * Config.TOP_FLOOR_FACTOR
    return masses


def build_story_stiffnesses(k_base_1_4: float, num_floors: int) -> list[float]:
    stiffnesses = [float(k_base_1_4)] * num_floors
    stiffnesses[-1] = float(k_base_1_4) * Config.TOP_FLOOR_FACTOR
    return stiffnesses


def compute_frame_story_yield_force(story_stiffness: float, floor_height: float) -> float:
    # Literature-aligned simplification for steel moment-frame springs:
    # Fy_story = K_story[kN/m] * h_story[m] * theta_y[-] = kN
    return float(story_stiffness * floor_height * Config.FRAME_YIELD_DRIFT_RATIO)


def format_damper_layout(flags: list[int] | tuple[int, ...]) -> str:
    # Example for a 3-story frame: "(0,0,0)" = no damper, "(0,0,1)" = damper only at story 3.
    return "(" + ",".join(str(int(flag)) for flag in flags) + ")"


def normalize_damper_layout_value(damper_layout: object, num_floors: int) -> str:
    if num_floors <= 0:
        return "()"

    if damper_layout is None:
        return format_damper_layout([1] * num_floors)

    try:
        if pd.isna(damper_layout):
            return format_damper_layout([1] * num_floors)
    except TypeError:
        pass

    if isinstance(damper_layout, str):
        text = damper_layout.strip()
        if not text:
            return format_damper_layout([1] * num_floors)
        bits = re.findall(r"[01]", text)
        if len(bits) == num_floors:
            return format_damper_layout([int(bit) for bit in bits])
        if re.fullmatch(r"[01]+", text) and len(text) <= num_floors:
            padded = text.zfill(num_floors)
            return format_damper_layout([int(bit) for bit in padded])

    if isinstance(damper_layout, (list, tuple, np.ndarray)):
        bits = [int(value) for value in damper_layout]
        if len(bits) == num_floors and all(bit in (0, 1) for bit in bits):
            return format_damper_layout(bits)

    raise ValueError(f"非法阻尼器安装位置参数: floors={num_floors}, damper_layout={damper_layout!r}")


def parse_damper_layout_flags(damper_layout: object, num_floors: int) -> tuple[int, ...]:
    normalized = normalize_damper_layout_value(damper_layout, num_floors)
    return tuple(int(bit) for bit in re.findall(r"[01]", normalized))


def damper_layout_code(damper_layout: object, num_floors: int) -> int:
    flags = parse_damper_layout_flags(damper_layout, num_floors)
    return int("".join(str(bit) for bit in flags), 2)


def canonicalize_structure_row(row: dict) -> dict:
    num_floors = int(row["num_floors"])
    normalized_layout = normalize_damper_layout_value(row.get("damper_layout"), num_floors)
    flags = parse_damper_layout_flags(normalized_layout, num_floors)
    row["damper_layout"] = normalized_layout
    if any(flags):
        row["k_add"] = float(row.get("k_add", Config.K_ADD))
        row["Fy_add"] = float(row["Fy_add"])
    else:
        row["k_add"] = 0.0
        row["Fy_add"] = 0.0
    return row


def build_damper_layout_options(num_floors: int) -> list[str]:
    return [
        format_damper_layout([(mask >> shift) & 1 for shift in range(num_floors - 1, -1, -1)])
        for mask in range(2 ** num_floors)
    ]


def sample_damper_layouts(
    num_floors: int,
    n_samples: int,
    rng: np.random.RandomState,
    mode: str = "space_filling",
) -> np.ndarray:
    # The installation scheme is sampled as one categorical parameter:
    # for n floors, enumerate all binary layouts such as (0,0,0), (0,0,1), ..., (1,1,1),
    # then shuffle / sample from that layout set directly.
    layout_values = build_damper_layout_options(num_floors)
    if n_samples <= 0:
        return np.asarray([], dtype=object)

    if mode in {"tail", "tail_extreme"}:
        install_counts = np.asarray(
            [sum(parse_damper_layout_flags(layout, num_floors)) for layout in layout_values],
            dtype=float,
        )
        decay = 1.1 if mode == "tail_extreme" else 0.75
        weights = np.exp(-decay * install_counts)
        weights = weights / weights.sum()
        return rng.choice(np.asarray(layout_values, dtype=object), size=n_samples, replace=True, p=weights)

    repeats = int(math.ceil(n_samples / len(layout_values)))
    sampled = np.asarray(layout_values * repeats, dtype=object)
    rng.shuffle(sampled)
    return sampled[:n_samples]


def structure_period_cache_key(row: dict) -> tuple:
    return (
        int(row["num_floors"]),
        float(row["floor_mass"]),
        float(row["floor_height"]),
        float(row["k_base_1_4"]),
        float(Config.FRAME_B),
        float(Config.FRAME_YIELD_DRIFT_RATIO),
        float(Config.PERIOD_LIMIT_SEC),
    )


def evaluate_structure_period_compliance(row: dict) -> dict:
    cache_key = structure_period_cache_key(row)
    cached = STRUCTURE_PERIOD_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    floor_mass = float(row["floor_mass"])
    num_floors = int(row["num_floors"])
    floor_height = float(row["floor_height"])
    k_base = float(row["k_base_1_4"])
    masses = build_floor_masses(floor_mass, num_floors)
    story_stiffnesses = build_story_stiffnesses(k_base, num_floors)

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    ops.node(0, 0.0)
    ops.fix(0, 1)
    for floor_idx in range(num_floors):
        ops.node(floor_idx + 1, 0.0)
        ops.mass(floor_idx + 1, masses[floor_idx], 0.0, 0.0)

    for floor_idx, story_stiffness in enumerate(story_stiffnesses, start=1):
        fy_story = compute_frame_story_yield_force(story_stiffness, floor_height)
        ops.uniaxialMaterial("Steel01", floor_idx, fy_story, story_stiffness, Config.FRAME_B)
        ops.element("zeroLength", floor_idx, floor_idx - 1, floor_idx, "-mat", floor_idx, "-dir", 1)

    try:
        eigenvalues = ops.eigen(1)
        if not eigenvalues or len(eigenvalues) < 1 or float(eigenvalues[0]) <= 0.0:
            result = {
                "period_1_sec": np.nan,
                "period_limit_sec": float(Config.PERIOD_LIMIT_SEC),
                "period_compliant": 0,
                "period_check_status": "eigen_failed",
            }
        else:
            omega_1 = math.sqrt(float(eigenvalues[0]))
            period_1_sec = 2.0 * math.pi / omega_1
            result = {
                "period_1_sec": float(period_1_sec),
                "period_limit_sec": float(Config.PERIOD_LIMIT_SEC),
                "period_compliant": int(period_1_sec <= Config.PERIOD_LIMIT_SEC),
                "period_check_status": "ok" if period_1_sec <= Config.PERIOD_LIMIT_SEC else "period_exceeds_limit",
            }
    finally:
        ops.wipe()

    STRUCTURE_PERIOD_CACHE[cache_key] = dict(result)
    return dict(result)


def quantize_drift(value: float) -> float:
    step = Config.DRIFT_QUANTIZATION_STEP
    return float(np.round(np.round(value / step) * step, 4))


def classify_failure_reason(failure_reason: str) -> str:
    reason = str(failure_reason or "")
    if not reason:
        return "ok"
    if reason.startswith("analysis_nonconvergence_step_"):
        return "opensees_nonconvergence"
    if reason == "period_exceeds_limit":
        return "period_exceeds_limit"
    if reason == "eigen_failed":
        return "eigen_failed"
    if reason.startswith("python_exception:"):
        return "python_exception"
    if reason.startswith("analysis_failed_step_"):
        return "opensees_analysis_failed"
    return "other_failure"


def default_output_value(column: str, analysis_status: str = ""):
    if column == "analysis_status":
        return analysis_status
    if column == "failure_reason":
        return ""
    if column == "failure_category":
        return "ok" if analysis_status == "ok" else ""
    if column == "analysis_return_code":
        return 0 if analysis_status == "ok" else -1
    if column == "analysis_failed_step":
        return -1
    if column == "analysis_failed_time":
        return np.nan
    if column == "nonconvergence_flag":
        return 0
    if column in {"steel01_yielded", "steel02_yielded"}:
        return -1
    return ""


def build_normalized_output_row(row_map: dict[str, object]) -> dict[str, object]:
    analysis_status = str(row_map.get("analysis_status", "") or "")
    normalized: dict[str, object] = {}
    for column in Config.OUTPUT_COLUMNS:
        value = row_map.get(column, default_output_value(column, analysis_status))
        if value == "" and column in {"analysis_return_code", "analysis_failed_step"}:
            value = default_output_value(column, analysis_status)
        if value == "" and column == "nonconvergence_flag":
            value = 0
        normalized[column] = value

    failure_reason = str(normalized.get("failure_reason", "") or "")
    failure_category = str(normalized.get("failure_category", "") or "")
    if not failure_category:
        failure_category = classify_failure_reason(failure_reason)
        normalized["failure_category"] = failure_category

    if str(normalized.get("analysis_status", "") or "") == "ok":
        normalized["analysis_return_code"] = 0 if str(normalized.get("analysis_return_code", "")) in {"", "nan"} else normalized["analysis_return_code"]
        normalized["nonconvergence_flag"] = 0
    else:
        if str(normalized.get("analysis_return_code", "")) == "":
            normalized["analysis_return_code"] = -1
        if failure_category == "opensees_nonconvergence":
            normalized["nonconvergence_flag"] = 1
    return normalized


def normalize_output_file_schema(output_path: Path) -> None:
    if not output_path.exists():
        return

    expected_header = list(Config.OUTPUT_COLUMNS)
    with output_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            header = []
            rows: list[list[str]] = []
        else:
            rows = [row for row in reader if row]

    if not header:
        with output_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=expected_header)
            writer.writeheader()
        return

    needs_rewrite = header != expected_header
    normalized_rows: list[dict[str, object]] = []

    for row in rows:
        if len(row) != len(expected_header):
            needs_rewrite = True
        if len(row) == len(expected_header):
            row_map = dict(zip(expected_header, row))
        else:
            row_map = dict(zip(header[: len(row)], row))
        normalized_rows.append(build_normalized_output_row(row_map))

    if not needs_rewrite:
        return

    backup_path = output_path.with_name(f"{output_path.stem}_schema-backup{output_path.suffix}")
    if not backup_path.exists():
        shutil.copy2(output_path, backup_path)

    temp_path = output_path.with_name(f"{output_path.stem}.schema_fix.tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=expected_header)
        writer.writeheader()
        writer.writerows(normalized_rows)
    temp_path.replace(output_path)
    print(f">>> Normalized output schema: {output_path.name}")
    print(f">>> Backup saved to: {backup_path}")


def structure_key(row: dict) -> tuple:
    normalized = canonicalize_structure_row(dict(row))
    return (
        int(normalized["num_floors"]),
        float(normalized["floor_mass"]),
        float(normalized["floor_height"]),
        float(normalized["k_base_1_4"]),
        float(normalized["k_add"]),
        float(normalized["Fy_add"]),
        str(normalized["damper_layout"]),
    )


def build_sample_id(split: str, row: dict, txt_path: str) -> str:
    normalized = canonicalize_structure_row(dict(row))
    wave_name = Path(txt_path).name
    layout_code = damper_layout_code(normalized["damper_layout"], int(normalized["num_floors"]))
    return (
        f"{split}|F{int(normalized['num_floors'])}"
        f"|M{float(normalized['floor_mass']):.1f}"
        f"|H{float(normalized['floor_height']):.1f}"
        f"|K{float(normalized['k_base_1_4']):.1f}"
        f"|Fy{float(normalized['Fy_add']):.1f}"
        f"|DL{layout_code:0{int(normalized['num_floors'])}b}"
        f"|W{wave_name}"
    )


def parse_dt_from_name(txt_path: str) -> float:
    if str(txt_path).startswith("h5://"):
        h5_path = str(txt_path)[5:].split("|")[0]
        try:
            with h5py.File(h5_path, 'r') as f:
                return float(f.attrs.get("dt", Config.DT))
        except Exception:
            return Config.DT
    match = re.search(r"DT_(\d+\.\d+)", Path(txt_path).name)
    if match:
        return float(match.group(1))
    return Config.DT


def resolve_analysis_time_params(task: dict) -> tuple[float, int]:
    if not Config.USE_WAVE_TIME_PARAMS:
        return float(Config.DT), int(Config.NUM_STEPS)

    wave_dt = task.get("wave_dt", Config.DT)
    try:
        analysis_dt = float(wave_dt)
    except (TypeError, ValueError):
        analysis_dt = Config.DT
    if not np.isfinite(analysis_dt) or analysis_dt <= 0.0:
        analysis_dt = parse_dt_from_name(str(task["txt_path"]))
    if not np.isfinite(analysis_dt) or analysis_dt <= 0.0:
        analysis_dt = Config.DT

    wave_length = task.get("wave_length", Config.NUM_STEPS)
    try:
        analysis_steps = int(wave_length)
    except (TypeError, ValueError):
        analysis_steps = Config.NUM_STEPS
    if analysis_steps <= 0:
        analysis_steps = Config.NUM_STEPS

    return float(analysis_dt), int(analysis_steps)


def resolve_image_path(txt_path: str) -> str:
    if str(txt_path).startswith("h5://"):
        return ""
    base_name = Path(txt_path).stem + ".png"
    return str((Config.IMAGE_DIR / base_name).resolve()).replace("\\", "/")


def load_wave_array(txt_path: str) -> np.ndarray:
    if txt_path.startswith("h5://"):
        h5_path, idx_str = txt_path[5:].split("|")
        with h5py.File(h5_path, 'r') as f:
            data = f["scaled_acceleration"][int(idx_str)]
    else:
        data = np.loadtxt(txt_path, dtype=np.float32)
    if data.ndim > 1:
        data = data.reshape(-1)
    return data


def extract_wave_features(txt_path: str) -> dict:
    data = load_wave_array(txt_path)
    dt = parse_dt_from_name(txt_path)
    abs_data = np.abs(data)
    energy = np.square(data, dtype=np.float64)

    pga = float(abs_data.max())
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    mean_abs = float(abs_data.mean())
    cav = float(abs_data.sum() * dt)
    arias_proxy = float(energy.sum() * dt)

    if len(data) > 1:
        zero_crossing_rate = float(np.mean((data[:-1] * data[1:]) < 0.0))
    else:
        zero_crossing_rate = 0.0

    if arias_proxy > 0.0:
        cumulative = np.cumsum(energy)
        total = cumulative[-1]
        idx_5 = int(np.searchsorted(cumulative, 0.05 * total, side="left"))
        idx_95 = int(np.searchsorted(cumulative, 0.95 * total, side="left"))
        duration_5_95 = float(max(0, idx_95 - idx_5) * dt)
    else:
        duration_5_95 = 0.0

    spectrum = np.abs(np.fft.rfft(data.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(data), d=dt)
    if len(freqs) > 1 and np.any(spectrum[1:] > 0.0):
        idx_peak = int(np.argmax(spectrum[1:]) + 1)
        dominant_freq = float(freqs[idx_peak])
        spectral_centroid = float(
            np.sum(freqs[1:] * spectrum[1:]) / (np.sum(spectrum[1:]) + 1e-12)
        )
    else:
        dominant_freq = 0.0
        spectral_centroid = 0.0
    predominant_period = float(1.0 / dominant_freq) if dominant_freq > 1e-9 else 0.0

    # All wave amplitude features below are kept in g-based units from the source file.
    # They are converted to m/s^2 only when fed into OpenSees via ACCELERATION_FACTOR.
    resolved_path = str(txt_path) if str(txt_path).startswith("h5://") else str(Path(txt_path).resolve()).replace("\\", "/")
    return {
        "txt_path": resolved_path,
        "image_path": resolve_image_path(txt_path),
        "wave_dt": dt,
        "wave_length": int(len(data)),
        "wave_pga": pga,
        "wave_rms": rms,
        "wave_mean_abs": mean_abs,
        "wave_cav": cav,
        "wave_arias_proxy": arias_proxy,
        "wave_duration_5_95": duration_5_95,
        "wave_zero_crossing_rate": zero_crossing_rate,
        "wave_dominant_freq": dominant_freq,
        "wave_spectral_centroid": spectral_centroid,
        "wave_predominant_period": predominant_period,
    }


def build_wave_feature_table(earthquake_files: list[str]) -> pd.DataFrame:
    paths = [str(path) if str(path).startswith("h5://") else str(Path(path).resolve()).replace("\\", "/") for path in earthquake_files]

    if Config.WAVE_FEATURE_OUTPUT.exists():
        try:
            cached = pd.read_csv(Config.WAVE_FEATURE_OUTPUT)
            if set(cached["txt_path"]) == set(paths):
                print(f">>> Reusing cached wave feature table: {Config.WAVE_FEATURE_OUTPUT}")
                return cached
        except Exception:
            pass

    rows = []
    for txt_path in tqdm(paths, desc="Extracting wave features"):
        rows.append(extract_wave_features(txt_path))

    df = pd.DataFrame(rows).sort_values("txt_path").reset_index(drop=True)
    intensity_raw = (
        np.log1p(df["wave_pga"])
        + np.log1p(df["wave_cav"])
        + np.log1p(df["wave_arias_proxy"])
    )
    df["wave_intensity_score"] = intensity_raw.rank(method="average", pct=True)
    df.to_csv(Config.WAVE_FEATURE_OUTPUT, index=False, encoding="utf-8-sig")
    print(f">>> Wave feature table saved to: {Config.WAVE_FEATURE_OUTPUT}")
    return df


def split_cluster_counts(cluster_size: int) -> tuple[int, int, int]:
    if cluster_size <= 1:
        return cluster_size, 0, 0
    if cluster_size == 2:
        return 1, 0, 1

    n_train = max(1, int(round(cluster_size * Config.TRAIN_WAVE_RATIO)))
    n_val = int(round(cluster_size * Config.VAL_WAVE_RATIO))
    n_test = cluster_size - n_train - n_val

    if cluster_size >= 4 and n_val == 0:
        n_val = 1
        n_train = max(1, n_train - 1)
    if cluster_size >= 3 and n_test == 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    while n_train + n_val + n_test > cluster_size:
        if n_train >= n_val and n_train >= n_test and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 0:
            n_val -= 1
        else:
            n_test -= 1
    while n_train + n_val + n_test < cluster_size:
        n_train += 1

    return n_train, n_val, n_test


def split_wave_pools(wave_df: pd.DataFrame) -> pd.DataFrame:
    if Config.WAVE_SPLIT_OUTPUT.exists():
        try:
            cached = pd.read_csv(Config.WAVE_SPLIT_OUTPUT)
            if set(cached["txt_path"]) == set(wave_df["txt_path"]):
                print(f">>> Reusing cached wave split table: {Config.WAVE_SPLIT_OUTPUT}")
                return cached
        except Exception:
            pass

    feature_cols = [
        "wave_pga",
        "wave_rms",
        "wave_cav",
        "wave_arias_proxy",
        "wave_duration_5_95",
        "wave_zero_crossing_rate",
        "wave_dominant_freq",
        "wave_spectral_centroid",
        "wave_predominant_period",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(wave_df[feature_cols].values.astype(np.float64))

    n_clusters = min(Config.WAVE_CLUSTER_COUNT, len(wave_df))
    model = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_SEED, n_init=10)

    split_df = wave_df.copy()
    split_df["wave_cluster"] = model.fit_predict(scaled)
    split_df["split"] = "train"

    rng = np.random.RandomState(Config.RANDOM_SEED)
    for cluster_id in sorted(split_df["wave_cluster"].unique()):
        cluster_df = split_df[split_df["wave_cluster"] == cluster_id].copy()
        order = np.array(cluster_df.index, dtype=int, copy=True)
        rng.shuffle(order)
        n_train, n_val, n_test = split_cluster_counts(len(order))
        split_df.loc[order[:n_train], "split"] = "train"
        split_df.loc[order[n_train : n_train + n_val], "split"] = "val"
        split_df.loc[order[n_train + n_val :], "split"] = "test"

    split_df = split_df.sort_values(["split", "wave_cluster", "txt_path"]).reset_index(drop=True)
    split_df.to_csv(Config.WAVE_SPLIT_OUTPUT, index=False, encoding="utf-8-sig")
    print(f">>> Wave split table saved to: {Config.WAVE_SPLIT_OUTPUT}")
    print(split_df["split"].value_counts().to_string())
    return split_df


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.RandomState) -> np.ndarray:
    lhs = np.empty((n_samples, n_dims), dtype=float)
    for dim in range(n_dims):
        cut = (np.arange(n_samples, dtype=float) + rng.random_sample(n_samples)) / n_samples
        rng.shuffle(cut)
        lhs[:, dim] = cut
    return lhs


def map_unit_to_values(unit_values: np.ndarray, discrete_values: np.ndarray) -> np.ndarray:
    indices = np.minimum((unit_values * len(discrete_values)).astype(int), len(discrete_values) - 1)
    return discrete_values[indices]


def generate_structure_design(
    structures_per_floor: int,
    used_structure_keys: set[tuple],
    seed: int,
    mode: str = "space_filling",
) -> list[dict]:
    rng = np.random.RandomState(seed)
    structures: list[dict] = []
    rejected_noncompliant = 0

    for num_floors in Config.NUM_FLOOR_OPTIONS:
        local_seen: set[tuple] = set()
        local_rows: list[dict] = []
        while len(local_rows) < structures_per_floor:
            need = structures_per_floor - len(local_rows)
            batch = max(64, need * 3)
            unit = latin_hypercube(batch, 4, rng)

            if mode in {"tail", "tail_extreme"}:
                # Tail drift is favored by high mass, low story stiffness and shorter story height.
                # Keep a low/high Fy mixture: low Fy encourages Steel02 yielding; high Fy with sparse layouts
                # keeps dampers effectively elastic and can expose larger frame drift.
                unit[:, 0] = np.sqrt(unit[:, 0])
                unit[:, 1] = np.square(unit[:, 1])
                unit[:, 2] = np.power(unit[:, 2], 3.0 if mode == "tail_extreme" else 2.0)
                if mode == "tail_extreme":
                    fy_units = unit[:, 3].copy()
                    low_fy_mask = rng.random_sample(batch) < 0.55
                    fy_units[low_fy_mask] = np.power(fy_units[low_fy_mask], 2.5)
                    fy_units[~low_fy_mask] = np.sqrt(fy_units[~low_fy_mask])
                    unit[:, 3] = fy_units
                else:
                    unit[:, 3] = np.power(unit[:, 3], 1.5)

            masses = map_unit_to_values(unit[:, 0], Config.FLOOR_MASS_VALUES)
            heights = map_unit_to_values(unit[:, 1], Config.FLOOR_HEIGHT_VALUES)
            stiffnesses = map_unit_to_values(unit[:, 2], Config.K_BASE_VALUES)
            layouts = sample_damper_layouts(int(num_floors), batch, rng, mode=mode)

            for mass, height, stiffness, fy_unit, damper_layout in zip(
                masses,
                heights,
                stiffnesses,
                unit[:, 3],
                layouts,
            ):
                fy_add_values = build_fy_add_values_for_structure(float(stiffness), float(height))
                fy_add = float(map_unit_to_values(np.asarray([fy_unit], dtype=float), fy_add_values)[0])
                row = {
                    "num_floors": int(num_floors),
                    "floor_mass": float(mass),
                    "floor_height": float(np.round(height, 1)),
                    "k_base_1_4": float(stiffness),
                    "k_add": float(Config.K_ADD),
                    "Fy_add": float(fy_add),
                    "damper_layout": str(damper_layout),
                    "frame_b": float(Config.FRAME_B),
                    "frame_yield_drift_ratio": float(Config.FRAME_YIELD_DRIFT_RATIO),
                }
                row = canonicalize_structure_row(row)
                key = structure_key(row)
                if key in used_structure_keys or key in local_seen:
                    continue
                period_info = evaluate_structure_period_compliance(row)
                if int(period_info["period_compliant"]) != 1:
                    rejected_noncompliant += 1
                    continue
                row.update(period_info)
                local_seen.add(key)
                local_rows.append(row)
                if len(local_rows) >= structures_per_floor:
                    break

        used_structure_keys.update(local_seen)
        structures.extend(local_rows)

    print(
        f">>> Structure design mode={mode}: kept {len(structures)} compliant structures, "
        f"rejected {rejected_noncompliant} for period limit {Config.PERIOD_LIMIT_SEC:.2f}s"
    )
    return structures


class WavePoolManager:
    def __init__(self, wave_df: pd.DataFrame, existing_txt_paths: list[str] | None = None):
        self.wave_df = wave_df.reset_index(drop=True).copy()
        self.by_cluster: dict[int, list[dict]] = {}
        self.wave_usage: Counter[str] = Counter()
        self.cluster_usage: Counter[int] = Counter()

        for cluster_id, cluster_df in self.wave_df.groupby("wave_cluster"):
            rows = cluster_df.to_dict("records")
            rows.sort(key=lambda row: (row["wave_intensity_score"], row["txt_path"]))
            self.by_cluster[int(cluster_id)] = rows

        if existing_txt_paths:
            self.mark_used(existing_txt_paths)

    def mark_used(self, txt_paths: list[str]) -> None:
        path_to_cluster = dict(zip(self.wave_df["txt_path"], self.wave_df["wave_cluster"]))
        for txt_path in txt_paths:
            self.wave_usage[txt_path] += 1
            cluster_id = int(path_to_cluster[txt_path])
            self.cluster_usage[cluster_id] += 1

    def _pick_from_cluster(
        self,
        cluster_id: int,
        local_used: set[str],
        local_counts: Counter[str],
        rng: np.random.RandomState,
        strong_bias: bool,
    ) -> dict:
        candidates = [row for row in self.by_cluster[cluster_id] if row["txt_path"] not in local_used]
        if not candidates:
            candidates = self.by_cluster[cluster_id]

        if strong_bias:
            candidates = sorted(
                candidates,
                key=lambda row: (
                    local_counts[row["txt_path"]] + self.wave_usage[row["txt_path"]],
                    -(row["wave_intensity_score"]),
                    row["txt_path"],
                ),
            )
        else:
            candidates = sorted(
                candidates,
                key=lambda row: (
                    local_counts[row["txt_path"]] + self.wave_usage[row["txt_path"]],
                    self.cluster_usage[int(row["wave_cluster"])],
                    row["txt_path"],
                ),
            )

        best_count = local_counts[candidates[0]["txt_path"]] + self.wave_usage[candidates[0]["txt_path"]]
        tied = [
            row
            for row in candidates
            if local_counts[row["txt_path"]] + self.wave_usage[row["txt_path"]] == best_count
        ]
        return tied[int(rng.randint(0, len(tied)))]

    def select_waves(
        self,
        n_waves: int,
        rng: np.random.RandomState,
        strong_bias: bool = False,
        update_usage: bool = False,
    ) -> list[dict]:
        selected: list[dict] = []
        local_used: set[str] = set()
        local_counts: Counter[str] = Counter()
        clusters = sorted(self.by_cluster.keys())

        while len(selected) < n_waves:
            cluster_order = sorted(
                clusters,
                key=lambda cid: (
                    self.cluster_usage[cid] + sum(
                        local_counts[row["txt_path"]]
                        for row in self.by_cluster[cid]
                    ),
                    cid,
                ),
            )
            cluster_id = cluster_order[len(selected) % len(cluster_order)]
            wave_row = self._pick_from_cluster(cluster_id, local_used, local_counts, rng, strong_bias)
            selected.append(wave_row)
            local_used.add(wave_row["txt_path"])
            local_counts[wave_row["txt_path"]] += 1

        if update_usage:
            self.mark_used([row["txt_path"] for row in selected])
        return selected


def build_task_records(
    structures: list[dict],
    wave_manager: WavePoolManager,
    split: str,
    stage: str,
    round_idx: int,
    waves_per_structure: int,
    rng_seed: int,
    existing_sample_ids: set[str],
    strong_wave_bias: bool = False,
) -> list[dict]:
    rng = np.random.RandomState(rng_seed)
    tasks: list[dict] = []
    for structure in structures:
        waves = wave_manager.select_waves(
            waves_per_structure,
            rng=rng,
            strong_bias=strong_wave_bias,
            update_usage=False,
        )
        for wave_row in waves:
            task = dict(structure)
            task["split"] = split
            task["stage"] = stage
            task["round_idx"] = round_idx
            task["txt_path"] = wave_row["txt_path"]
            task["image_path"] = wave_row["image_path"]
            task["sample_id"] = build_sample_id(split, structure, wave_row["txt_path"])
            task["wave_cluster"] = int(wave_row["wave_cluster"])
            task["wave_dt"] = float(wave_row["wave_dt"])
            task["wave_length"] = int(wave_row["wave_length"])
            task["wave_pga"] = float(wave_row["wave_pga"])
            task["wave_rms"] = float(wave_row["wave_rms"])
            task["wave_mean_abs"] = float(wave_row["wave_mean_abs"])
            task["wave_cav"] = float(wave_row["wave_cav"])
            task["wave_arias_proxy"] = float(wave_row["wave_arias_proxy"])
            task["wave_duration_5_95"] = float(wave_row["wave_duration_5_95"])
            task["wave_zero_crossing_rate"] = float(wave_row["wave_zero_crossing_rate"])
            task["wave_dominant_freq"] = float(wave_row["wave_dominant_freq"])
            task["wave_spectral_centroid"] = float(wave_row["wave_spectral_centroid"])
            task["wave_predominant_period"] = float(wave_row["wave_predominant_period"])
            task["wave_intensity_score"] = float(wave_row["wave_intensity_score"])
            if task["sample_id"] in existing_sample_ids:
                continue
            tasks.append(task)
    return tasks


def ensure_output_file(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        normalize_output_file_schema(output_path)
        return
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=Config.OUTPUT_COLUMNS)
        writer.writeheader()


def load_existing_output(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(columns=Config.OUTPUT_COLUMNS)
    normalize_output_file_schema(output_path)
    df = pd.read_csv(output_path, low_memory=False)
    for column in Config.OUTPUT_COLUMNS:
        if column not in df.columns:
            if column == "analysis_status":
                df[column] = ""
            elif column in {"analysis_return_code", "analysis_failed_step", "nonconvergence_flag"}:
                df[column] = -1
            elif column == "analysis_failed_time":
                df[column] = np.nan
            else:
                df[column] = ""
    return df[Config.OUTPUT_COLUMNS]


def collect_existing_state() -> dict:
    state = {
        "sample_ids": set(),
        "structure_keys_all": set(),
        "structure_keys_train": set(),
        "txt_paths_by_split": {"train": [], "val": [], "test": []},
    }
    for split, path in (
        ("train", Config.TRAIN_OUTPUT),
        ("val", Config.VAL_OUTPUT),
        ("test", Config.TEST_OUTPUT),
    ):
        df = load_existing_output(path)
        if df.empty:
            continue
        state["sample_ids"].update(df["sample_id"].astype(str).tolist())
        state["txt_paths_by_split"][split].extend(df["txt_path"].astype(str).tolist())
        for _, row in df.iterrows():
            key = structure_key(row.to_dict())
            state["structure_keys_all"].add(key)
            if split == "train":
                state["structure_keys_train"].add(key)
    return state


def initialize_output_files() -> None:
    ensure_output_file(Config.TRAIN_OUTPUT)
    ensure_output_file(Config.VAL_OUTPUT)
    ensure_output_file(Config.TEST_OUTPUT)


def run_peak_drift_analysis(task: dict) -> dict:
    task = canonicalize_structure_row(dict(task))
    floor_mass = float(task["floor_mass"])
    num_floors = int(task["num_floors"])
    floor_height = float(task["floor_height"])
    fy_add = float(task["Fy_add"])
    k_add_param = float(task["k_add"])
    k_base = float(task["k_base_1_4"])
    damper_flags = parse_damper_layout_flags(task.get("damper_layout", ""), num_floors)
    frame_b = float(task.get("frame_b", Config.FRAME_B))
    frame_yield_drift_ratio = float(task.get("frame_yield_drift_ratio", Config.FRAME_YIELD_DRIFT_RATIO))
    masses = build_floor_masses(floor_mass, num_floors)
    story_stiffnesses = build_story_stiffnesses(k_base, num_floors)

    period_1_sec = task.get("period_1_sec", np.nan)
    period_limit_sec = float(task.get("period_limit_sec", Config.PERIOD_LIMIT_SEC))
    period_compliant = int(task.get("period_compliant", -1))
    period_check_status = str(task.get("period_check_status", "unknown"))
    if not np.isfinite(period_1_sec) or period_compliant < 0:
        period_info = evaluate_structure_period_compliance(task)
        period_1_sec = period_info["period_1_sec"]
        period_limit_sec = period_info["period_limit_sec"]
        period_compliant = int(period_info["period_compliant"])
        period_check_status = str(period_info["period_check_status"])
    if period_compliant != 1:
        failure_reason = (
            "eigen_failed"
            if period_check_status == "eigen_failed"
            else "period_exceeds_limit"
        )
        return {
            "analysis_status": "failed",
            "failure_reason": failure_reason,
            "failure_category": classify_failure_reason(failure_reason),
            "analysis_return_code": -1,
            "analysis_failed_step": -1,
            "analysis_failed_time": np.nan,
            "nonconvergence_flag": 0,
            "period_1_sec": float(period_1_sec) if np.isfinite(period_1_sec) else np.nan,
            "period_limit_sec": float(period_limit_sec),
            "period_compliant": int(period_compliant),
            "period_check_status": period_check_status,
        }

    analysis_dt, analysis_steps = resolve_analysis_time_params(task)

    ops.wipe()
    ops.model("basic", "-ndm", 1, "-ndf", 1)

    ops.node(0, 0.0)
    ops.fix(0, 1)
    for idx in range(num_floors):
        ops.node(idx + 1, 0.0)
    for idx in range(num_floors):
        ops.mass(idx + 1, masses[idx], 0.0, 0.0)

    for idx, story_stiffness in enumerate(story_stiffnesses, start=1):
        fy_story = float(story_stiffness * floor_height * frame_yield_drift_ratio)
        ops.uniaxialMaterial("Steel01", idx, fy_story, story_stiffness, frame_b)
        ops.element("zeroLength", idx, idx - 1, idx, "-mat", idx, "-dir", 1)

    damper_ele_specs: list[tuple[int, int]] = []
    for floor_idx, install_flag in enumerate(damper_flags, start=1):
        if int(install_flag) != 1:
            continue
        mat_add_tag = num_floors + floor_idx
        ele_add_tag = num_floors + floor_idx
        ops.uniaxialMaterial("Steel02", mat_add_tag, fy_add, k_add_param, Config.B, Config.R0, Config.CR1, Config.CR2)
        ops.element("zeroLength", ele_add_tag, floor_idx - 1, floor_idx, "-mat", mat_add_tag, "-dir", 1)
        damper_ele_specs.append((ele_add_tag, floor_idx))

    eigenvalues = ops.eigen(Config.N_EIGEN)
    if not eigenvalues or len(eigenvalues) < Config.N_EIGEN:
        return {
            "analysis_status": "failed",
            "failure_reason": "eigen_failed",
            "failure_category": "eigen_failed",
            "analysis_return_code": -1,
            "analysis_failed_step": -1,
            "analysis_failed_time": np.nan,
            "nonconvergence_flag": 0,
            "period_1_sec": float(period_1_sec) if np.isfinite(period_1_sec) else np.nan,
            "period_limit_sec": float(period_limit_sec),
            "period_compliant": int(period_compliant),
            "period_check_status": period_check_status,
        }

    omega_i = math.sqrt(eigenvalues[0])
    omega_j = math.sqrt(eigenvalues[1])
    alpha_m = Config.DAMPING_RATIO * (2.0 * omega_i * omega_j) / (omega_i + omega_j)
    beta_k = Config.DAMPING_RATIO * 2.0 / (omega_i + omega_j)
    ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)

    txt_path_str = str(task["txt_path"])
    if txt_path_str.startswith("h5://"):
        wave_arr = load_wave_array(txt_path_str)
        ops.timeSeries("Path", 1, "-dt", analysis_dt, "-values", *wave_arr.tolist(), "-factor", Config.ACCELERATION_FACTOR)
    else:
        accel_file_path_tcl = txt_path_str.replace("\\", "/")
        ops.timeSeries("Path", 1, "-dt", analysis_dt, "-filePath", accel_file_path_tcl, "-factor", Config.ACCELERATION_FACTOR)
    ops.pattern("UniformExcitation", 1, 1, "-accel", 1)

    ops.wipeAnalysis()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-6, 10, 0)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    max_drift = 0.0
    peak_story_id = 1
    peak_damper_force = 0.0
    steel01_yielded = 0
    steel02_yielded = 0
    steel01_yield_disp = float(floor_height * frame_yield_drift_ratio)
    yield_disp = fy_add / k_add_param if k_add_param > 0.0 else math.inf

    for step_idx in range(analysis_steps):
        analysis_return_code = int(ops.analyze(1, analysis_dt))
        if analysis_return_code < 0:
            return {
                "analysis_status": "failed",
                "failure_reason": f"analysis_nonconvergence_step_{step_idx + 1}_code_{analysis_return_code}",
                "failure_category": "opensees_nonconvergence",
                "analysis_return_code": analysis_return_code,
                "analysis_failed_step": int(step_idx + 1),
                "analysis_failed_time": float((step_idx + 1) * analysis_dt),
                "nonconvergence_flag": 1,
                "period_1_sec": float(period_1_sec) if np.isfinite(period_1_sec) else np.nan,
                "period_limit_sec": float(period_limit_sec),
                "period_compliant": int(period_compliant),
                "period_check_status": period_check_status,
            }

        first_disp = ops.nodeDisp(1, 1)
        first_drift = abs(first_disp / floor_height)
        if first_drift > max_drift:
            max_drift = first_drift
            peak_story_id = 1

        story_rel_disps = [float(first_disp)]
        prev_disp = first_disp
        for floor_idx in range(2, num_floors + 1):
            curr_disp = ops.nodeDisp(floor_idx, 1)
            story_rel_disp = curr_disp - prev_disp
            story_rel_disps.append(float(story_rel_disp))
            drift = abs(story_rel_disp / floor_height)
            if drift > max_drift:
                max_drift = drift
                peak_story_id = floor_idx
            prev_disp = curr_disp

        if any(abs(float(story_rel_disp)) >= steel01_yield_disp for story_rel_disp in story_rel_disps):
            steel01_yielded = 1

        for damper_ele_tag, floor_idx in damper_ele_specs:
            story_rel_disp = story_rel_disps[floor_idx - 1]
            try:
                force_response = ops.eleResponse(damper_ele_tag, "force")
                if force_response and len(force_response) > 0:
                    current_damper_force = abs(float(force_response[0]))
                else:
                    current_damper_force = abs(float(story_rel_disp)) * k_add_param
            except Exception:
                current_damper_force = abs(float(story_rel_disp)) * k_add_param

            peak_damper_force = max(peak_damper_force, current_damper_force)
            if current_damper_force >= fy_add or abs(float(story_rel_disp)) >= yield_disp:
                steel02_yielded = 1

    return {
        "analysis_status": "ok",
        "failure_reason": "",
        "failure_category": "ok",
        "analysis_return_code": 0,
        "analysis_failed_step": -1,
        "analysis_failed_time": np.nan,
        "nonconvergence_flag": 0,
        "period_1_sec": float(period_1_sec) if np.isfinite(period_1_sec) else np.nan,
        "period_limit_sec": float(period_limit_sec),
        "period_compliant": int(period_compliant),
        "period_check_status": period_check_status,
        "max_drift_ratio_raw": float(max_drift),
        "max_drift_ratio_q1e4": quantize_drift(float(max_drift)),
        "steel01_yielded": int(steel01_yielded),
        "steel02_yielded": int(steel02_yielded),
        "peak_damper_force": float(peak_damper_force),
        "yield_margin": float(peak_damper_force / fy_add) if fy_add > 0.0 else np.nan,
        "peak_story_id": int(peak_story_id),
    }


def process_task(task: dict) -> dict:
    result = canonicalize_structure_row(dict(task))
    try:
        analysis = run_peak_drift_analysis(result)
        result.update(analysis)
    except Exception as exc:
        result["analysis_status"] = "failed"
        result["failure_reason"] = f"python_exception:{type(exc).__name__}:{exc}"
        result["failure_category"] = "python_exception"
        result["analysis_return_code"] = -1
        result["analysis_failed_step"] = -1
        result["analysis_failed_time"] = np.nan
        result["nonconvergence_flag"] = 0
    finally:
        ops.wipe()

    if result["analysis_status"] != "ok":
        result["max_drift_ratio_raw"] = np.nan
        result["max_drift_ratio_q1e4"] = np.nan
        result["steel01_yielded"] = -1
        result["steel02_yielded"] = -1
        result["peak_damper_force"] = np.nan
        result["yield_margin"] = np.nan
        result["peak_story_id"] = -1
        result.setdefault("period_1_sec", np.nan)
        result.setdefault("period_limit_sec", float(Config.PERIOD_LIMIT_SEC))
        result.setdefault("period_compliant", -1)
        result.setdefault("period_check_status", "unknown")
        result.setdefault("failure_category", classify_failure_reason(result.get("failure_reason", "")))
        result.setdefault("analysis_return_code", -1)
        result.setdefault("analysis_failed_step", -1)
        result.setdefault("analysis_failed_time", np.nan)
        result.setdefault("nonconvergence_flag", 0)
    else:
        result.setdefault("failure_category", "ok")
        result.setdefault("analysis_return_code", 0)
        result.setdefault("analysis_failed_step", -1)
        result.setdefault("analysis_failed_time", np.nan)
        result.setdefault("nonconvergence_flag", 0)

    for column in Config.OUTPUT_COLUMNS:
        result.setdefault(column, "")
    return {column: result[column] for column in Config.OUTPUT_COLUMNS}


def append_rows(output_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_output_file(output_path)
    with output_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=Config.OUTPUT_COLUMNS)
        writer.writerows(rows)


def execute_tasks(tasks: list[dict], output_path: Path) -> int:
    if not tasks:
        return 0

    print(f">>> Running {len(tasks)} tasks -> {output_path.name}")
    written = 0
    buffer: list[dict] = []
    success_count = 0
    failed_count = 0
    nonconvergence_count = 0
    failure_category_counts: Counter[str] = Counter()
    with multiprocessing.Pool(Config.NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(process_task, tasks)
        for row in tqdm(iterator, total=len(tasks), desc=f"Generating {output_path.stem}"):
            buffer.append(row)
            if str(row.get("analysis_status", "")) == "ok":
                success_count += 1
            else:
                failed_count += 1
                failure_category = str(row.get("failure_category", "") or "other_failure")
                failure_category_counts[failure_category] += 1
                try:
                    if int(row.get("nonconvergence_flag", 0)) == 1:
                        nonconvergence_count += 1
                except (TypeError, ValueError):
                    pass
            if len(buffer) >= Config.WRITE_FLUSH_EVERY:
                append_rows(output_path, buffer)
                written += len(buffer)
                buffer.clear()

    if buffer:
        append_rows(output_path, buffer)
        written += len(buffer)
    print(
        ">>> Task stats:"
        f" success={success_count}, failed={failed_count}, nonconverged={nonconvergence_count}"
    )
    if failure_category_counts:
        print(f">>> Failure categories: {dict(failure_category_counts)}")
    return written


def load_training_successes() -> pd.DataFrame:
    df = load_existing_output(Config.TRAIN_OUTPUT)
    if df.empty:
        return df
    return df[df["analysis_status"] == "ok"].reset_index(drop=True)


def build_active_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=df.index)
    feature_df["num_floors"] = df["num_floors"].astype(float)
    feature_df["floor_mass"] = df["floor_mass"].astype(float)
    feature_df["floor_height"] = df["floor_height"].astype(float)
    feature_df["k_base_1_4"] = df["k_base_1_4"].astype(float)
    feature_df["k_add"] = df["k_add"].astype(float)
    feature_df["Fy_add"] = df["Fy_add"].astype(float)
    feature_df["period_1_sec"] = df["period_1_sec"].astype(float)
    feature_df["wave_cluster"] = df["wave_cluster"].astype(float)

    damper_layout_codes: list[int] = []
    damper_install_counts: list[int] = []
    for _, row in df.iterrows():
        num_floors = int(row["num_floors"])
        flags = parse_damper_layout_flags(row.get("damper_layout", ""), num_floors)
        damper_layout_codes.append(damper_layout_code(flags, num_floors))
        damper_install_counts.append(sum(flags))
    feature_df["damper_layout_code"] = np.asarray(damper_layout_codes, dtype=float)
    feature_df["damper_install_count"] = np.asarray(damper_install_counts, dtype=float)
    feature_df["damper_install_ratio"] = feature_df["damper_install_count"] / feature_df["num_floors"]

    feature_df["log_floor_mass"] = np.log(feature_df["floor_mass"])
    feature_df["log_k_base"] = np.log(feature_df["k_base_1_4"])
    feature_df["log_fy"] = np.log1p(feature_df["Fy_add"])
    feature_df["stiffness_mass_ratio"] = feature_df["k_base_1_4"] / feature_df["floor_mass"]
    feature_df["yield_disp"] = np.where(
        feature_df["k_add"] > 0.0,
        feature_df["Fy_add"] / feature_df["k_add"],
        0.0,
    )
    feature_df["period_ratio_to_limit"] = feature_df["period_1_sec"] / Config.PERIOD_LIMIT_SEC

    wave_cols = [
        "wave_pga",
        "wave_rms",
        "wave_mean_abs",
        "wave_cav",
        "wave_arias_proxy",
        "wave_duration_5_95",
        "wave_zero_crossing_rate",
        "wave_dominant_freq",
        "wave_spectral_centroid",
        "wave_predominant_period",
        "wave_intensity_score",
    ]
    for column in wave_cols:
        feature_df[column] = df[column].astype(float)

    for column in ["wave_pga", "wave_rms", "wave_mean_abs", "wave_cav", "wave_arias_proxy"]:
        feature_df[f"log_{column}"] = np.log1p(feature_df[column])
    return feature_df


def fit_active_committee(train_df: pd.DataFrame) -> dict | None:
    if len(train_df) < 500:
        return None

    x_train = build_active_features(train_df).values.astype(np.float32)
    y_train = train_df["max_drift_ratio_raw"].astype(float).values
    y_train_trans = np.log1p(y_train * Config.ACTIVE_TARGET_SCALE)

    regressors = [
        RandomForestRegressor(
            n_estimators=200,
            random_state=Config.RANDOM_SEED,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        ExtraTreesRegressor(
            n_estimators=240,
            random_state=Config.RANDOM_SEED + 1,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        RandomForestRegressor(
            n_estimators=160,
            random_state=Config.RANDOM_SEED + 2,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        ),
    ]
    for model in regressors:
        model.fit(x_train, y_train_trans)

    classifier_models = []
    y_class = train_df["steel02_yielded"].astype(int).values
    if len(np.unique(y_class)) >= 2:
        classifier_models = [
            RandomForestClassifier(
                n_estimators=160,
                random_state=Config.RANDOM_SEED,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
            ExtraTreesClassifier(
                n_estimators=200,
                random_state=Config.RANDOM_SEED + 3,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ]
        for model in classifier_models:
            model.fit(x_train, y_class)

    return {
        "regressors": regressors,
        "classifiers": classifier_models,
        "label_counts": np.bincount(
            np.digitize(y_train, Config.DRIFT_BINS, right=False),
            minlength=len(Config.DRIFT_BINS) + 1,
        ),
    }


def predict_committee(candidate_df: pd.DataFrame, committee: dict | None) -> pd.DataFrame:
    out = candidate_df.copy()
    if committee is None:
        out["pred_mean"] = 0.0
        out["pred_std"] = 0.0
        out["yield_uncertainty"] = 0.0
        return out

    x_cand = build_active_features(candidate_df).values.astype(np.float32)
    pred_stack = []
    for model in committee["regressors"]:
        pred_trans = model.predict(x_cand)
        pred_stack.append(np.expm1(pred_trans) / Config.ACTIVE_TARGET_SCALE)
    pred_stack_arr = np.vstack(pred_stack)
    out["pred_mean"] = pred_stack_arr.mean(axis=0)
    out["pred_std"] = pred_stack_arr.std(axis=0)

    if committee["classifiers"]:
        prob_stack = []
        for model in committee["classifiers"]:
            prob_stack.append(model.predict_proba(x_cand)[:, 1])
        prob_mean = np.vstack(prob_stack).mean(axis=0)
        out["yield_uncertainty"] = 1.0 - np.abs(prob_mean - 0.5) * 2.0
    else:
        out["yield_uncertainty"] = 0.0
    return out


def normalize_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def add_stage_b_scores(candidate_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    drift_counts = np.bincount(
        np.digitize(train_df["max_drift_ratio_raw"].astype(float).values, Config.DRIFT_BINS, right=False),
        minlength=len(Config.DRIFT_BINS) + 1,
    )
    pred_bins = np.digitize(candidate_df["pred_mean"].astype(float).values, Config.DRIFT_BINS, right=False)
    scarcity = np.array([1.0 / math.sqrt(drift_counts[idx] + 1.0) for idx in pred_bins], dtype=float)

    uncertainty = normalize_score(candidate_df["pred_std"].astype(float).values)
    scarcity_norm = normalize_score(scarcity)
    yield_unc = normalize_score(candidate_df["yield_uncertainty"].astype(float).values)

    candidate_df = candidate_df.copy()
    candidate_df["selection_score"] = 0.55 * uncertainty + 0.25 * scarcity_norm + 0.20 * yield_unc
    return candidate_df


def add_stage_c_scores(candidate_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    drift_counts = np.bincount(
        np.digitize(train_df["max_drift_ratio_raw"].astype(float).values, Config.DRIFT_BINS, right=False),
        minlength=len(Config.DRIFT_BINS) + 1,
    )
    pred_bins = np.digitize(candidate_df["pred_mean"].astype(float).values, Config.DRIFT_BINS, right=False)
    scarcity = np.array([1.0 / math.sqrt(drift_counts[idx] + 1.0) for idx in pred_bins], dtype=float)
    tail_strength = np.clip((candidate_df["pred_mean"].astype(float).values - 0.01) / 0.04, 0.0, 1.0)
    strong_wave = normalize_score(candidate_df["wave_intensity_score"].astype(float).values)
    uncertainty = normalize_score(candidate_df["pred_std"].astype(float).values)
    scarcity_norm = normalize_score(scarcity)
    yield_unc = normalize_score(candidate_df["yield_uncertainty"].astype(float).values)

    candidate_df = candidate_df.copy()
    candidate_df["selection_score"] = (
        0.35 * uncertainty
        + 0.30 * tail_strength
        + 0.20 * scarcity_norm
        + 0.10 * strong_wave
        + 0.05 * yield_unc
    )
    return candidate_df


def select_with_floor_quota(candidate_df: pd.DataFrame, total_select: int) -> pd.DataFrame:
    floors = sorted(candidate_df["num_floors"].astype(int).unique())
    base_quota = total_select // len(floors)
    remainder = total_select % len(floors)

    selected_parts = []
    chosen_ids = set()
    for idx, floor in enumerate(floors):
        quota = base_quota + (1 if idx < remainder else 0)
        floor_df = candidate_df[candidate_df["num_floors"].astype(int) == floor]
        floor_top = floor_df.nlargest(quota, "selection_score")
        selected_parts.append(floor_top)
        chosen_ids.update(floor_top["sample_id"].tolist())

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidate_df.iloc[0:0].copy()
    if len(selected_df) < total_select:
        remaining = candidate_df[~candidate_df["sample_id"].isin(chosen_ids)]
        selected_df = pd.concat(
            [selected_df, remaining.nlargest(total_select - len(selected_df), "selection_score")],
            ignore_index=True,
        )
    return selected_df.head(total_select).reset_index(drop=True)


def _sigmoid_score(values: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(float(width), 1.0e-9)
    z = np.clip((np.asarray(values, dtype=float) - float(center)) / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def build_structural_tail_risk(candidate_df: pd.DataFrame) -> np.ndarray:
    install_counts: list[int] = []
    for _, row in candidate_df.iterrows():
        num_floors = int(row["num_floors"])
        install_counts.append(sum(parse_damper_layout_flags(row.get("damper_layout", ""), num_floors)))

    num_floors_arr = candidate_df["num_floors"].astype(float).values
    mass_arr = candidate_df["floor_mass"].astype(float).values
    height_arr = candidate_df["floor_height"].astype(float).values
    stiffness_arr = candidate_df["k_base_1_4"].astype(float).values
    period_arr = candidate_df["period_1_sec"].astype(float).values
    wave_arr = candidate_df["wave_intensity_score"].astype(float).values
    install_ratio = np.asarray(install_counts, dtype=float) / np.maximum(num_floors_arr, 1.0)

    return (
        0.18 * normalize_score(num_floors_arr)
        + 0.18 * normalize_score(mass_arr)
        + 0.18 * normalize_score(-stiffness_arr)
        + 0.14 * normalize_score(-height_arr)
        + 0.14 * normalize_score(1.0 - install_ratio)
        + 0.10 * normalize_score(period_arr / Config.PERIOD_LIMIT_SEC)
        + 0.08 * normalize_score(wave_arr)
    )


def add_tail_focus_scores(candidate_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if train_df.empty or "max_drift_ratio_raw" not in train_df.columns:
        drift_counts = np.ones(len(Config.DRIFT_BINS) + 1, dtype=float)
    else:
        drift_counts = np.bincount(
            np.digitize(train_df["max_drift_ratio_raw"].astype(float).values, Config.DRIFT_BINS, right=False),
            minlength=len(Config.DRIFT_BINS) + 1,
        ).astype(float)

    pred_mean = candidate_df["pred_mean"].astype(float).values
    pred_bins = np.digitize(pred_mean, Config.DRIFT_BINS, right=False)
    scarcity = np.array([1.0 / math.sqrt(drift_counts[idx] + 1.0) for idx in pred_bins], dtype=float)

    uncertainty = normalize_score(candidate_df["pred_std"].astype(float).values)
    scarcity_norm = normalize_score(scarcity)
    strong_wave = normalize_score(candidate_df["wave_intensity_score"].astype(float).values)
    yield_unc = normalize_score(candidate_df["yield_uncertainty"].astype(float).values)
    pred_tail = normalize_score(pred_mean)
    tail_ge_010 = _sigmoid_score(pred_mean, center=0.006, width=0.002)
    tail_ge_020 = _sigmoid_score(pred_mean, center=0.012, width=0.003)
    structural_tail = build_structural_tail_risk(candidate_df)

    candidate_df = candidate_df.copy()
    candidate_df["structural_tail_risk"] = structural_tail
    candidate_df["selection_score"] = (
        0.22 * pred_tail
        + 0.18 * tail_ge_010
        + 0.10 * tail_ge_020
        + 0.18 * structural_tail
        + 0.14 * scarcity_norm
        + 0.10 * uncertainty
        + 0.05 * strong_wave
        + 0.03 * yield_unc
    )
    return candidate_df


def select_with_soft_floor_quota(candidate_df: pd.DataFrame, total_select: int) -> pd.DataFrame:
    if candidate_df.empty or total_select <= 0:
        return candidate_df.iloc[0:0].copy()

    total_select = min(int(total_select), len(candidate_df))
    floors = sorted(candidate_df["num_floors"].astype(int).unique())
    min_per_floor = max(1, int(round(total_select * Config.TAIL_MIN_PER_FLOOR_RATIO)))
    max_equal_quota = max(1, total_select // max(len(floors), 1))
    min_per_floor = min(min_per_floor, max_equal_quota)

    selected_parts = []
    chosen_ids = set()
    for floor in floors:
        floor_df = candidate_df[candidate_df["num_floors"].astype(int) == floor]
        floor_top = floor_df.nlargest(min_per_floor, "selection_score")
        selected_parts.append(floor_top)
        chosen_ids.update(floor_top["sample_id"].tolist())

    selected_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidate_df.iloc[0:0].copy()
    if len(selected_df) < total_select:
        remaining = candidate_df[~candidate_df["sample_id"].isin(chosen_ids)]
        selected_df = pd.concat(
            [selected_df, remaining.nlargest(total_select - len(selected_df), "selection_score")],
            ignore_index=True,
        )
    return selected_df.head(total_select).reset_index(drop=True)


TAIL_DRIFT_THRESHOLDS = (0.005, 0.01, 0.015, 0.02)


def summarize_tail_coverage(output_path: Path) -> dict:
    df = load_existing_output(output_path)
    success_df = df[df["analysis_status"] == "ok"].copy()
    coverage = {
        "successful_rows": int(len(success_df)),
        "drift_threshold_counts": {f"ge_{threshold:.3f}": 0 for threshold in TAIL_DRIFT_THRESHOLDS},
        "drift_threshold_ratios": {f"ge_{threshold:.3f}": 0.0 for threshold in TAIL_DRIFT_THRESHOLDS},
        "steel01_yielded_count": 0,
        "steel02_yielded_count": 0,
        "steel01_yield_ratio": 0.0,
        "steel02_yield_ratio": 0.0,
    }
    if success_df.empty:
        return coverage

    drift = pd.to_numeric(success_df["max_drift_ratio_raw"], errors="coerce")
    valid_drift = drift.dropna()
    for threshold in TAIL_DRIFT_THRESHOLDS:
        key = f"ge_{threshold:.3f}"
        count = int((valid_drift >= threshold).sum())
        coverage["drift_threshold_counts"][key] = count
        coverage["drift_threshold_ratios"][key] = float(count / len(success_df))

    steel01 = pd.to_numeric(success_df["steel01_yielded"], errors="coerce").fillna(0).astype(int)
    steel02 = pd.to_numeric(success_df["steel02_yielded"], errors="coerce").fillna(0).astype(int)
    coverage["steel01_yielded_count"] = int((steel01 == 1).sum())
    coverage["steel02_yielded_count"] = int((steel02 == 1).sum())
    coverage["steel01_yield_ratio"] = float((steel01 == 1).mean())
    coverage["steel02_yield_ratio"] = float((steel02 == 1).mean())
    return coverage


def tail_target_requirements(split: str) -> dict:
    if split == "train":
        return {
            "drift_ge_010": Config.TRAIN_MIN_DRIFT_GE_010,
            "drift_ge_020": Config.TRAIN_MIN_DRIFT_GE_020,
            "steel01_yielded": Config.TRAIN_MIN_STEEL01_YIELDED,
        }
    return {
        "drift_ge_010": Config.HOLDOUT_MIN_DRIFT_GE_010,
        "drift_ge_020": Config.HOLDOUT_MIN_DRIFT_GE_020,
        "steel01_yielded": Config.HOLDOUT_MIN_STEEL01_YIELDED,
    }


def tail_targets_met(output_path: Path, split: str) -> bool:
    coverage = summarize_tail_coverage(output_path)
    targets = tail_target_requirements(split)
    return (
        coverage["drift_threshold_counts"]["ge_0.010"] >= targets["drift_ge_010"]
        and coverage["drift_threshold_counts"]["ge_0.020"] >= targets["drift_ge_020"]
        and coverage["steel01_yielded_count"] >= targets["steel01_yielded"]
    )


def stage_a_train(
    train_wave_manager: WavePoolManager,
    existing_state: dict,
) -> int:
    structures = generate_structure_design(
        structures_per_floor=Config.STAGE_A_STRUCTURES_PER_FLOOR,
        used_structure_keys=existing_state["structure_keys_train"],
        seed=Config.RANDOM_SEED + 10,
        mode="space_filling",
    )
    tasks = build_task_records(
        structures=structures,
        wave_manager=train_wave_manager,
        split="train",
        stage="stage_a",
        round_idx=0,
        waves_per_structure=Config.STAGE_A_WAVES_PER_STRUCTURE,
        rng_seed=Config.RANDOM_SEED + 101,
        existing_sample_ids=existing_state["sample_ids"],
        strong_wave_bias=False,
    )
    written = execute_tasks(tasks, Config.TRAIN_OUTPUT)
    if tasks:
        existing_state["sample_ids"].update([task["sample_id"] for task in tasks])
        task_keys = [structure_key(task) for task in tasks]
        existing_state["structure_keys_train"].update(task_keys)
        existing_state["structure_keys_all"].update(task_keys)
        train_wave_manager.mark_used([task["txt_path"] for task in tasks])
    return written


def stage_b_active_rounds(
    train_wave_manager: WavePoolManager,
    existing_state: dict,
) -> int:
    total_written = 0
    for round_idx in range(1, Config.STAGE_B_ROUNDS + 1):
        train_df = load_training_successes()
        committee = fit_active_committee(train_df)
        candidate_structure_keys = set(existing_state["structure_keys_train"])
        structures = generate_structure_design(
            structures_per_floor=Config.STAGE_B_CANDIDATE_STRUCTURES_PER_FLOOR,
            used_structure_keys=candidate_structure_keys,
            seed=Config.RANDOM_SEED + 200 + round_idx,
            mode="space_filling",
        )
        tasks = build_task_records(
            structures=structures,
            wave_manager=train_wave_manager,
            split="train",
            stage="stage_b",
            round_idx=round_idx,
            waves_per_structure=Config.STAGE_B_WAVES_PER_STRUCTURE,
            rng_seed=Config.RANDOM_SEED + 300 + round_idx,
            existing_sample_ids=existing_state["sample_ids"],
            strong_wave_bias=False,
        )
        if not tasks:
            continue
        candidate_df = pd.DataFrame(tasks)
        candidate_df = predict_committee(candidate_df, committee)
        candidate_df = add_stage_b_scores(candidate_df, train_df if not train_df.empty else candidate_df.assign(max_drift_ratio_raw=0.0))
        selected_df = select_with_floor_quota(candidate_df, Config.STAGE_B_SELECT_PER_ROUND)
        selected_tasks = selected_df.to_dict("records")
        written = execute_tasks(selected_tasks, Config.TRAIN_OUTPUT)
        total_written += written
        if selected_tasks:
            existing_state["sample_ids"].update([task["sample_id"] for task in selected_tasks])
            task_keys = [structure_key(task) for task in selected_tasks]
            existing_state["structure_keys_train"].update(task_keys)
            existing_state["structure_keys_all"].update(task_keys)
            train_wave_manager.mark_used([task["txt_path"] for task in selected_tasks])
    return total_written


def stage_c_tail(
    train_wave_manager: WavePoolManager,
    existing_state: dict,
) -> int:
    train_df = load_training_successes()
    committee = fit_active_committee(train_df)
    candidate_structure_keys = set(existing_state["structure_keys_train"])
    structures = generate_structure_design(
        structures_per_floor=Config.STAGE_C_CANDIDATE_STRUCTURES_PER_FLOOR,
        used_structure_keys=candidate_structure_keys,
        seed=Config.RANDOM_SEED + 500,
        mode="tail_extreme",
    )
    tasks = build_task_records(
        structures=structures,
        wave_manager=train_wave_manager,
        split="train",
        stage="stage_c",
        round_idx=0,
        waves_per_structure=Config.STAGE_C_WAVES_PER_STRUCTURE,
        rng_seed=Config.RANDOM_SEED + 600,
        existing_sample_ids=existing_state["sample_ids"],
        strong_wave_bias=True,
    )
    if not tasks:
        return 0

    candidate_df = pd.DataFrame(tasks)
    if not candidate_df.empty:
        threshold = candidate_df["wave_intensity_score"].quantile(1.0 - Config.TAIL_STRONG_WAVE_TOP_RATIO)
        candidate_df = candidate_df[candidate_df["wave_intensity_score"] >= threshold].reset_index(drop=True)
    candidate_df = predict_committee(candidate_df, committee)
    candidate_df = add_tail_focus_scores(
        candidate_df,
        train_df if not train_df.empty else candidate_df.assign(max_drift_ratio_raw=0.0),
    )
    selected_df = select_with_soft_floor_quota(candidate_df, Config.STAGE_C_SELECT_TOTAL)
    selected_tasks = selected_df.to_dict("records")
    written = execute_tasks(selected_tasks, Config.TRAIN_OUTPUT)
    if selected_tasks:
        existing_state["sample_ids"].update([task["sample_id"] for task in selected_tasks])
        task_keys = [structure_key(task) for task in selected_tasks]
        existing_state["structure_keys_train"].update(task_keys)
        existing_state["structure_keys_all"].update(task_keys)
        train_wave_manager.mark_used([task["txt_path"] for task in selected_tasks])
    return written


def generate_holdout_split(
    split: str,
    output_path: Path,
    wave_manager: WavePoolManager,
    existing_state: dict,
    structures_per_floor: int,
    waves_per_structure: int,
    seed_offset: int,
) -> int:
    structures = generate_structure_design(
        structures_per_floor=structures_per_floor,
        used_structure_keys=existing_state["structure_keys_all"],
        seed=Config.RANDOM_SEED + seed_offset,
        mode="space_filling",
    )
    tasks = build_task_records(
        structures=structures,
        wave_manager=wave_manager,
        split=split,
        stage=f"{split}_holdout",
        round_idx=0,
        waves_per_structure=waves_per_structure,
        rng_seed=Config.RANDOM_SEED + seed_offset + 1,
        existing_sample_ids=existing_state["sample_ids"],
        strong_wave_bias=False,
    )
    written = execute_tasks(tasks, output_path)
    if tasks:
        existing_state["sample_ids"].update([task["sample_id"] for task in tasks])
        existing_state["structure_keys_all"].update([structure_key(task) for task in tasks])
        wave_manager.mark_used([task["txt_path"] for task in tasks])
    return written


def generate_tail_topup_split(
    split: str,
    output_path: Path,
    wave_manager: WavePoolManager,
    existing_state: dict,
    select_per_round: int,
    max_rounds: int,
    seed_offset: int,
) -> int:
    total_written = 0
    stage_name = "stage_d_tail_topup" if split == "train" else f"{split}_tail_topup"

    for topup_round in range(1, max_rounds + 1):
        coverage = summarize_tail_coverage(output_path)
        targets = tail_target_requirements(split)
        print(
            f">>> Tail coverage before {stage_name} round {topup_round}: "
            f">=0.01 {coverage['drift_threshold_counts']['ge_0.010']}/{targets['drift_ge_010']}, "
            f">=0.02 {coverage['drift_threshold_counts']['ge_0.020']}/{targets['drift_ge_020']}, "
            f"steel01 {coverage['steel01_yielded_count']}/{targets['steel01_yielded']}"
        )
        if tail_targets_met(output_path, split):
            print(f">>> Tail targets already satisfied for split={split}; skip remaining top-up.")
            break

        train_df = load_training_successes()
        committee = fit_active_committee(train_df)
        candidate_structure_keys = set(existing_state["structure_keys_all"])
        if split == "train":
            candidate_structure_keys.update(existing_state["structure_keys_train"])

        structures = generate_structure_design(
            structures_per_floor=Config.TAIL_TOPUP_CANDIDATE_STRUCTURES_PER_FLOOR,
            used_structure_keys=candidate_structure_keys,
            seed=Config.RANDOM_SEED + seed_offset + topup_round * 10,
            mode="tail_extreme",
        )
        tasks = build_task_records(
            structures=structures,
            wave_manager=wave_manager,
            split=split,
            stage=stage_name,
            round_idx=topup_round,
            waves_per_structure=Config.TAIL_TOPUP_WAVES_PER_STRUCTURE,
            rng_seed=Config.RANDOM_SEED + seed_offset + topup_round * 10 + 1,
            existing_sample_ids=existing_state["sample_ids"],
            strong_wave_bias=True,
        )
        if not tasks:
            print(f">>> No candidate tasks generated for {stage_name} round {topup_round}.")
            break

        candidate_df = pd.DataFrame(tasks)
        threshold = candidate_df["wave_intensity_score"].quantile(1.0 - Config.TAIL_STRONG_WAVE_TOP_RATIO)
        candidate_df = candidate_df[candidate_df["wave_intensity_score"] >= threshold].reset_index(drop=True)
        candidate_df = predict_committee(candidate_df, committee)
        candidate_df = add_tail_focus_scores(
            candidate_df,
            train_df if not train_df.empty else candidate_df.assign(max_drift_ratio_raw=0.0),
        )
        selected_df = select_with_soft_floor_quota(candidate_df, select_per_round)
        selected_tasks = selected_df.to_dict("records")
        if not selected_tasks:
            print(f">>> No selected tasks for {stage_name} round {topup_round}.")
            break

        written = execute_tasks(selected_tasks, output_path)
        total_written += written
        existing_state["sample_ids"].update([task["sample_id"] for task in selected_tasks])
        task_keys = [structure_key(task) for task in selected_tasks]
        existing_state["structure_keys_all"].update(task_keys)
        if split == "train":
            existing_state["structure_keys_train"].update(task_keys)
        wave_manager.mark_used([task["txt_path"] for task in selected_tasks])

    coverage = summarize_tail_coverage(output_path)
    print(
        f">>> Tail coverage after {stage_name}: "
        f">=0.01 {coverage['drift_threshold_counts']['ge_0.010']}, "
        f">=0.02 {coverage['drift_threshold_counts']['ge_0.020']}, "
        f"steel01 {coverage['steel01_yielded_count']}, "
        f"steel02 {coverage['steel02_yielded_count']}"
    )
    return total_written


def summarize_outputs() -> dict:
    summary = {
        "run_tag": Config.RUN_TAG,
        "paths": {
            "train": str(Config.TRAIN_OUTPUT),
            "val": str(Config.VAL_OUTPUT),
            "test": str(Config.TEST_OUTPUT),
            "wave_features": str(Config.WAVE_FEATURE_OUTPUT),
            "wave_split": str(Config.WAVE_SPLIT_OUTPUT),
        },
        "splits": {},
    }

    for split_name, output_path in (
        ("train", Config.TRAIN_OUTPUT),
        ("val", Config.VAL_OUTPUT),
        ("test", Config.TEST_OUTPUT),
    ):
        df = load_existing_output(output_path)
        if df.empty:
            summary["splits"][split_name] = {"rows": 0}
            continue
        success_df = df[df["analysis_status"] == "ok"]
        failed_df = df[df["analysis_status"] != "ok"].copy()
        failed_df["failure_category"] = failed_df["failure_reason"].map(classify_failure_reason).where(
            failed_df["failure_reason"].notna(),
            failed_df.get("failure_category", ""),
        )
        nonconvergence_series = pd.to_numeric(df["nonconvergence_flag"], errors="coerce").fillna(0).astype(int)
        nonconvergence_count = int((nonconvergence_series == 1).sum())
        split_summary = {
            "rows": int(len(df)),
            "successful_rows": int(len(success_df)),
            "failed_rows": int(len(failed_df)),
            "nonconverged_rows": nonconvergence_count,
            "nonconvergence_ratio_all": float(nonconvergence_count / len(df)) if len(df) > 0 else 0.0,
            "nonconvergence_ratio_failed": float(nonconvergence_count / len(failed_df)) if len(failed_df) > 0 else 0.0,
            "unique_waves": int(df["txt_path"].nunique()),
            "unique_structures": int(
                df[["num_floors", "floor_mass", "floor_height", "k_base_1_4", "k_add", "Fy_add", "damper_layout"]]
                .drop_duplicates()
                .shape[0]
            ),
            "stage_counts": df["stage"].value_counts().to_dict(),
            "failure_category_counts": failed_df["failure_category"].astype(str).value_counts().to_dict() if not failed_df.empty else {},
        }
        if not failed_df.empty:
            return_code_counts = (
                pd.to_numeric(failed_df["analysis_return_code"], errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            split_summary["analysis_return_code_counts"] = {
                str(code): int(count) for code, count in return_code_counts.items()
            }
        if not success_df.empty:
            drift = pd.to_numeric(success_df["max_drift_ratio_raw"], errors="coerce").dropna()
            split_summary["drift_mean"] = float(drift.mean())
            split_summary["drift_q50"] = float(drift.quantile(0.50))
            split_summary["drift_q90"] = float(drift.quantile(0.90))
            split_summary["drift_q95"] = float(drift.quantile(0.95))
            split_summary["drift_q99"] = float(drift.quantile(0.99))
            split_summary["drift_max"] = float(drift.max())
            tail_coverage = summarize_tail_coverage(output_path)
            split_summary["drift_threshold_counts"] = tail_coverage["drift_threshold_counts"]
            split_summary["drift_threshold_ratios"] = tail_coverage["drift_threshold_ratios"]
            steel01_yielded = success_df["steel01_yielded"].astype(int)
            steel02_yielded = success_df["steel02_yielded"].astype(int)
            split_summary["steel01_yielded_count"] = int((steel01_yielded == 1).sum())
            split_summary["steel02_yielded_count"] = int((steel02_yielded == 1).sum())
            split_summary["steel01_yield_ratio"] = float((steel01_yielded == 1).mean())
            split_summary["steel02_yield_ratio"] = float((steel02_yielded == 1).mean())
            split_summary["yield_ratio"] = float((steel02_yielded == 1).mean())
            split_summary["tail_target_requirements"] = tail_target_requirements(split_name)
            split_summary["tail_targets_met"] = bool(tail_targets_met(output_path, split_name))
            split_summary["period_1_mean"] = float(success_df["period_1_sec"].astype(float).mean())
            split_summary["period_1_max"] = float(success_df["period_1_sec"].astype(float).max())
            split_summary["period_limit_sec"] = float(Config.PERIOD_LIMIT_SEC)
        summary["splits"][split_name] = split_summary

    with Config.SUMMARY_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f">>> Summary saved to: {Config.SUMMARY_OUTPUT}")
    return summary


def main() -> None:
    if Config.OUTPUT_DIR.is_file() and Config.OUTPUT_DIR.suffix == ".h5":
        import h5py
        with h5py.File(Config.OUTPUT_DIR, "r") as f:
            total_waves = f["scaled_acceleration"].shape[0]
            earthquake_files = [f"h5://{Config.OUTPUT_DIR.resolve().as_posix()}|{i}" for i in range(total_waves)]
    else:
        if not Config.OUTPUT_DIR.exists():
            raise FileNotFoundError(f"地震波目录不存在: {Config.OUTPUT_DIR}")

        earthquake_files = sorted(str(path) for path in Config.OUTPUT_DIR.glob("*.txt"))
        if not earthquake_files:
            raise FileNotFoundError(f"目录 '{Config.OUTPUT_DIR}' 中未找到任何 .txt 文件")

    print(f">>> Found {len(earthquake_files)} earthquake files in: {Config.OUTPUT_DIR}")
    print(f">>> Output basename: {Config.OUTPUT_BASENAME}")
    print(f">>> Hyperparameter file: {Config.HYPERPARAM_PATH}")
    print(f">>> CPU workers: {Config.NUM_WORKERS}")
    print(f">>> Dataset size scale: {Config.DATASET_SIZE_SCALE:.2f}x")
    if Config.USE_WAVE_TIME_PARAMS:
        print(">>> Analysis time params: per-wave dt / wave_length")
    else:
        print(f">>> Analysis time params: fixed dt={Config.DT}, steps={Config.NUM_STEPS}")

    initialize_output_files()
    wave_feature_df = build_wave_feature_table(earthquake_files)
    wave_split_df = split_wave_pools(wave_feature_df)

    existing_state = collect_existing_state()

    train_wave_manager = WavePoolManager(
        wave_split_df[wave_split_df["split"] == "train"],
        existing_txt_paths=existing_state["txt_paths_by_split"]["train"],
    )
    val_wave_manager = WavePoolManager(
        wave_split_df[wave_split_df["split"] == "val"],
        existing_txt_paths=existing_state["txt_paths_by_split"]["val"],
    )
    test_wave_manager = WavePoolManager(
        wave_split_df[wave_split_df["split"] == "test"],
        existing_txt_paths=existing_state["txt_paths_by_split"]["test"],
    )

    budget_stage_a = len(Config.NUM_FLOOR_OPTIONS) * Config.STAGE_A_STRUCTURES_PER_FLOOR * Config.STAGE_A_WAVES_PER_STRUCTURE
    budget_stage_b = Config.STAGE_B_ROUNDS * Config.STAGE_B_SELECT_PER_ROUND
    budget_stage_c = Config.STAGE_C_SELECT_TOTAL
    budget_val = len(Config.NUM_FLOOR_OPTIONS) * Config.VAL_STRUCTURES_PER_FLOOR * Config.HOLDOUT_WAVES_PER_STRUCTURE
    budget_test = len(Config.NUM_FLOOR_OPTIONS) * Config.TEST_STRUCTURES_PER_FLOOR * Config.HOLDOUT_WAVES_PER_STRUCTURE
    budget_train_tail_topup = Config.TAIL_TOPUP_TRAIN_MAX_ROUNDS * Config.TAIL_TOPUP_TRAIN_SELECT_PER_ROUND
    budget_val_tail_topup = Config.TAIL_TOPUP_HOLDOUT_MAX_ROUNDS * Config.TAIL_TOPUP_HOLDOUT_SELECT_PER_ROUND
    budget_test_tail_topup = Config.TAIL_TOPUP_HOLDOUT_MAX_ROUNDS * Config.TAIL_TOPUP_TEST_SELECT_PER_ROUND
    print(
        ">>> Planned high-fidelity budgets:"
        f" stage_a={budget_stage_a}, stage_b={budget_stage_b}, stage_c={budget_stage_c},"
        f" train_tail_topup<={budget_train_tail_topup},"
        f" val={budget_val}+tail<={budget_val_tail_topup},"
        f" test={budget_test}+tail<={budget_test_tail_topup}"
    )

    written_a = stage_a_train(train_wave_manager, existing_state)
    print(f">>> Stage A newly written rows: {written_a}")

    written_b = stage_b_active_rounds(train_wave_manager, existing_state)
    print(f">>> Stage B newly written rows: {written_b}")

    written_c = stage_c_tail(train_wave_manager, existing_state)
    print(f">>> Stage C newly written rows: {written_c}")

    written_train_tail = generate_tail_topup_split(
        split="train",
        output_path=Config.TRAIN_OUTPUT,
        wave_manager=train_wave_manager,
        existing_state=existing_state,
        select_per_round=Config.TAIL_TOPUP_TRAIN_SELECT_PER_ROUND,
        max_rounds=Config.TAIL_TOPUP_TRAIN_MAX_ROUNDS,
        seed_offset=1100,
    )
    print(f">>> Train tail top-up newly written rows: {written_train_tail}")

    written_val = generate_holdout_split(
        split="val",
        output_path=Config.VAL_OUTPUT,
        wave_manager=val_wave_manager,
        existing_state=existing_state,
        structures_per_floor=Config.VAL_STRUCTURES_PER_FLOOR,
        waves_per_structure=Config.HOLDOUT_WAVES_PER_STRUCTURE,
        seed_offset=700,
    )
    print(f">>> Val holdout newly written rows: {written_val}")

    written_val_tail = generate_tail_topup_split(
        split="val",
        output_path=Config.VAL_OUTPUT,
        wave_manager=val_wave_manager,
        existing_state=existing_state,
        select_per_round=Config.TAIL_TOPUP_HOLDOUT_SELECT_PER_ROUND,
        max_rounds=Config.TAIL_TOPUP_HOLDOUT_MAX_ROUNDS,
        seed_offset=1300,
    )
    print(f">>> Val tail top-up newly written rows: {written_val_tail}")

    written_test = generate_holdout_split(
        split="test",
        output_path=Config.TEST_OUTPUT,
        wave_manager=test_wave_manager,
        existing_state=existing_state,
        structures_per_floor=Config.TEST_STRUCTURES_PER_FLOOR,
        waves_per_structure=Config.HOLDOUT_WAVES_PER_STRUCTURE,
        seed_offset=900,
    )
    print(f">>> Test holdout newly written rows: {written_test}")

    written_test_tail = generate_tail_topup_split(
        split="test",
        output_path=Config.TEST_OUTPUT,
        wave_manager=test_wave_manager,
        existing_state=existing_state,
        select_per_round=Config.TAIL_TOPUP_TEST_SELECT_PER_ROUND,
        max_rounds=Config.TAIL_TOPUP_HOLDOUT_MAX_ROUNDS,
        seed_offset=1500,
    )
    print(f">>> Test tail top-up newly written rows: {written_test_tail}")

    summary = summarize_outputs()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
