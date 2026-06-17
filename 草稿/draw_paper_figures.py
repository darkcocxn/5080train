from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "草稿" / "论文插图"
PNG = OUT / "png"
SVG = OUT / "svg"
PNG.mkdir(parents=True, exist_ok=True)
SVG.mkdir(parents=True, exist_ok=True)


def pick_font() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"


FONT = pick_font()
mpl.rcParams["font.family"] = [FONT]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42


BLUE = "#2E74B5"
BLUE_DARK = "#1F4D78"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
RED = "#C84C4C"
INK = "#263238"
GRAY = "#E8EEF5"
GRAY2 = "#F6F7F9"
LINE = "#8A94A6"


def save(fig: plt.Figure, name: str):
    fig.savefig(PNG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc=GRAY2, ec=LINE, lw=1.2, fs=10, color=INK, r=0.04, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.014,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=color,
        weight="bold" if bold else "normal",
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color=LINE, lw=1.4, ms=13, rad=0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def panel_title(ax, title):
    ax.text(0.01, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=15, weight="bold", color=BLUE_DARK)


def blank_ax(figsize=(11, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def fig01_workflow():
    fig, ax = blank_ax((13, 5.6))
    panel_title(ax, "图 1 研究技术路线")
    xs = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    labels = [
        "NGA-West2\n天然强震记录",
        "筛选、截取\n与调幅",
        "OpenSees\n非线性时程分析",
        "样本与特征\n数据集",
        "多模态/时序/表格\n代理模型",
        "泛化、尾部\n与阈值评价",
    ]
    for i, x in enumerate(xs):
        box(ax, x, 0.44, 0.13, 0.18, labels[i], fc="#FFFFFF", ec=BLUE, fs=9.5, bold=True)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.13, 0.53), (xs[i + 1], 0.53), color=BLUE)
    save(fig, "fig01_research_workflow")


def fig02_ground_motion_pipeline():
    fig = plt.figure(figsize=(12.5, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_title(ax, "图 2 地震动筛选与调幅流程")
    steps = [
        ("原始记录", 1966, "NGA-West2"),
        ("时间步筛选", 1672, "0.005/0.01 s"),
        ("Arias 窗口截取", 1473, "20 s 标准片段"),
        ("罕遇级筛选", 275, "PGA ≥ 0.10g"),
        ("剔除极高强度", 233, "最高 15% 去除"),
    ]
    x0, y0 = 0.06, 0.28
    maxn = steps[0][1]
    for i, (name, count, note) in enumerate(steps):
        x = x0 + i * 0.19
        h = 0.1 + 0.34 * count / maxn
        box(ax, x, y0, 0.13, h, f"{count}\n{name}", fc="#FFFFFF", ec=BLUE if i == 0 else TEAL, fs=10, bold=True)
        if i < len(steps) - 1:
            arrow(ax, (x + 0.13, y0 + h / 2), (x0 + (i + 1) * 0.19, y0 + h / 2), color=LINE)
    ax.text(0.06, 0.11, "最终地震动：233 条；调幅系数约 0.998-2.0；调幅后 PGA 约 0.2005-0.3559g。", fontsize=10, color=INK)

    ax2 = fig.add_subplot(gs[1])
    rng = np.random.default_rng(16)
    pga = np.clip(rng.normal(0.28, 0.04, 1800), 0.2005, 0.3559)
    ax2.hist(pga, bins=28, color=BLUE, alpha=0.82, edgecolor="white")
    ax2.axvline(0.28, color=RED, lw=2, label="目标均值 0.28g")
    ax2.axvline(0.2005, color=INK, lw=1, ls="--")
    ax2.axvline(0.3559, color=INK, lw=1, ls="--")
    ax2.set_title("调幅后峰值加速度的目标分布示意", loc="left", fontsize=12, color=BLUE_DARK, weight="bold")
    ax2.set_xlabel("PGA / g")
    ax2.set_ylabel("相对频数")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False)
    save(fig, "fig02_ground_motion_screening")


def fig03_arias_window():
    rng = np.random.default_rng(8)
    t = np.linspace(0, 45, 4501)
    envelope = np.exp(-0.5 * ((t - 22) / 5.2) ** 2) + 0.18 * np.exp(-0.5 * ((t - 10) / 2.5) ** 2)
    acc = envelope * (0.55 * np.sin(2 * np.pi * 1.7 * t) + 0.25 * np.sin(2 * np.pi * 4.4 * t)) + 0.04 * rng.normal(size=t.size)
    energy = np.cumsum(acc**2)
    energy = energy / energy[-1]
    win_start, win_end = 12.4, 32.4
    mask = (t >= win_start) & (t <= win_end)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    axs[0].plot(t, acc, color=INK, lw=0.8)
    axs[0].fill_between(t[mask], acc[mask], 0, color=BLUE, alpha=0.18)
    axs[0].axvline(win_start, color=BLUE, lw=1.5)
    axs[0].axvline(win_end, color=BLUE, lw=1.5)
    axs[0].set_ylabel("加速度")
    axs[0].set_title("图 3 Arias 强度滑动窗口截取示意", loc="left", fontsize=15, weight="bold", color=BLUE_DARK)
    axs[0].text((win_start + win_end) / 2, max(acc) * 0.86, "能量最集中的 20 s 核心片段", ha="center", color=BLUE_DARK, fontsize=10)
    axs[1].plot(t, energy, color=TEAL, lw=2)
    axs[1].fill_between(t[mask], energy[mask], energy[mask].min(), color=TEAL, alpha=0.15)
    axs[1].axvline(win_start, color=BLUE, lw=1.5)
    axs[1].axvline(win_end, color=BLUE, lw=1.5)
    axs[1].set_ylabel("归一化累计 Arias 强度")
    axs[1].set_xlabel("时间 / s")
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.22)
    save(fig, "fig03_arias_window")


def fig04_frame_encoding():
    fig, ax = blank_ax((12, 6.2))
    panel_title(ax, "图 4 阻尼钢框架参数化与布置编码")
    base_x, base_y = 0.12, 0.16
    width, story_h = 0.28, 0.11
    n = 5
    for i in range(n + 1):
        y = base_y + i * story_h
        ax.plot([base_x, base_x + width], [y, y], color=INK, lw=2)
    for x in [base_x, base_x + width]:
        ax.plot([x, x], [base_y, base_y + n * story_h], color=INK, lw=2.2)
    damped = [1, 0, 1, 1, 0]
    for i, d in enumerate(damped):
        y1 = base_y + i * story_h
        y2 = base_y + (i + 1) * story_h
        if d:
            ax.plot([base_x + 0.04, base_x + width - 0.04], [y1 + 0.02, y2 - 0.02], color=RED, lw=2.4)
            ax.add_patch(Rectangle((base_x + width / 2 - 0.018, (y1 + y2) / 2 - 0.014), 0.036, 0.028, ec=RED, fc="white", lw=1.5))
        ax.text(base_x - 0.045, (y1 + y2) / 2, f"{i + 1}F", ha="right", va="center", fontsize=9, color=INK)
    ax.text(base_x + width / 2, base_y - 0.055, "3-7 层参数化阻尼钢框架", ha="center", fontsize=10, color=INK, weight="bold")

    box(ax, 0.52, 0.58, 0.36, 0.18, "结构参数\n楼层数、楼层质量、层高、主体刚度、附加刚度、屈服强度、一阶周期", fc="#FFFFFF", ec=BLUE, fs=10)
    box(ax, 0.52, 0.34, 0.36, 0.14, "阻尼器逐层编码\n[d1, d2, d3, d4, d5, d6, d7]", fc="#FFFFFF", ec=TEAL, fs=10, bold=True)
    vector = [1, 0, 1, 1, 0, 0, 0]
    for i, v in enumerate(vector):
        x = 0.535 + i * 0.045
        y = 0.22
        box(ax, x, y, 0.034, 0.05, str(v), fc=RED if v else GRAY2, ec=RED if v else LINE, fs=10, color="white" if v else INK, r=0.01, bold=True)
    arrow(ax, (0.42, 0.44), (0.52, 0.43), color=LINE)
    save(fig, "fig04_damped_frame_encoding")


def scalogram_array(n=160):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    z = (
        1.2 * np.exp(-((X - 0.35) ** 2 / 0.018 + (Y - 0.28) ** 2 / 0.03))
        + 0.9 * np.exp(-((X - 0.63) ** 2 / 0.02 + (Y - 0.65) ** 2 / 0.015))
        + 0.35 * np.sin(18 * X) * np.exp(-5 * Y)
    )
    return z


def fig05_input_representations():
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], height_ratios=[1, 0.9], hspace=0.38, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 20, 2001)
    acc = np.sin(2 * np.pi * 1.3 * t) * np.exp(-0.5 * ((t - 9) / 3.3) ** 2) + 0.35 * np.sin(2 * np.pi * 4.1 * t) * np.exp(-0.5 * ((t - 12) / 2.1) ** 2)
    ax0.plot(t, acc, color=INK, lw=0.8)
    ax0.set_title("原始加速度时程", fontsize=11, color=BLUE_DARK, weight="bold")
    ax0.set_xlabel("时间 / s")
    ax0.set_ylabel("a(t)")
    ax0.grid(alpha=0.22)
    ax0.spines[["top", "right"]].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(scalogram_array(), cmap="Blues", aspect="auto", origin="lower")
    ax1.set_title("小波时频图 160×160", fontsize=11, color=BLUE_DARK, weight="bold")
    ax1.set_xlabel("时间")
    ax1.set_ylabel("频率")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_axis_off()
    panel_title(ax2, "标量与派生特征")
    chips = ["结构参数", "地震动统计", "周期耦合", "阻尼器布置", "尾部风险代理"]
    for i, c in enumerate(chips):
        box(ax2, 0.08, 0.78 - i * 0.14, 0.74, 0.08, c, fc="#FFFFFF", ec=TEAL if i >= 3 else BLUE, fs=9.5)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_axis_off()
    panel_title(ax3, "不同模型使用的输入表达")
    paths = [
        ("2D-CNN", "小波时频图 + 标量特征", BLUE),
        ("LSTM / WaveNet", "原始序列 + 标量特征", TEAL),
        ("树模型 / MLP", "标量特征 + 256 点下采样波形", AMBER),
    ]
    for i, (name, inp, col) in enumerate(paths):
        x = 0.08 + i * 0.31
        box(ax3, x, 0.35, 0.24, 0.22, f"{name}\n{inp}", fc="#FFFFFF", ec=col, fs=10, bold=True)
    save(fig, "fig05_input_representations")


def fig06_multimodal_cnn():
    fig, ax = blank_ax((13, 6.5))
    panel_title(ax, "图 6 多模态 2D-CNN 主模型结构")
    # Image branch
    box(ax, 0.04, 0.67, 0.12, 0.12, "小波时频图\n160×160", fc="#FFFFFF", ec=BLUE, fs=9.5, bold=True)
    convs = [("残差卷积\n32通道", 0.21), ("残差卷积\n64通道", 0.35), ("残差卷积\n128通道", 0.49), ("全局池化\nAvg+Max", 0.63)]
    for text, x in convs:
        box(ax, x, 0.66, 0.10, 0.14, text, fc=GRAY2, ec=BLUE, fs=9)
    for x1, x2 in [(0.16, 0.21), (0.31, 0.35), (0.45, 0.49), (0.59, 0.63)]:
        arrow(ax, (x1, 0.73), (x2, 0.73), color=BLUE)
    # Scalar branch
    box(ax, 0.04, 0.30, 0.13, 0.13, "63 维标量\n结构+地震动+布置", fc="#FFFFFF", ec=TEAL, fs=9.3, bold=True)
    box(ax, 0.24, 0.30, 0.12, 0.13, "残差 MLP\n标量嵌入", fc=GRAY2, ec=TEAL, fs=9.5)
    box(ax, 0.43, 0.27, 0.16, 0.18, "条件调制\nγ(s), β(s), g(s)\nF'=g⊙[(1+γ)⊙F+β]", fc="#FFFFFF", ec=TEAL, fs=8.5)
    arrow(ax, (0.17, 0.365), (0.24, 0.365), color=TEAL)
    arrow(ax, (0.36, 0.365), (0.43, 0.365), color=TEAL)
    arrow(ax, (0.51, 0.45), (0.54, 0.66), color=TEAL, rad=0.18)
    # Fusion
    box(ax, 0.78, 0.52, 0.13, 0.20, "门控双线性融合\n[hI, hs, hI⊙hs]\n384 维", fc="#FFFFFF", ec=AMBER, fs=9.2, bold=True)
    box(ax, 0.78, 0.24, 0.13, 0.16, "回归头\nLinear + ReLU\nDropout", fc=GRAY2, ec=AMBER, fs=9.2)
    box(ax, 0.95, 0.34, 0.04, 0.20, "ŷ\n最大层间\n位移角", fc=BLUE_DARK, ec=BLUE_DARK, fs=9, color="white", bold=True, r=0.02)
    arrow(ax, (0.73, 0.73), (0.78, 0.64), color=BLUE)
    arrow(ax, (0.59, 0.36), (0.78, 0.56), color=TEAL)
    arrow(ax, (0.845, 0.52), (0.845, 0.40), color=AMBER)
    arrow(ax, (0.91, 0.32), (0.95, 0.44), color=AMBER)
    save(fig, "fig06_multimodal_2dcnn_architecture")


def fig07_sequence_baselines():
    fig, ax = blank_ax((12.5, 6.4))
    panel_title(ax, "图 7 LSTM 与 WaveNet 时序基线模型")
    # LSTM lane
    y = 0.64
    box(ax, 0.05, y, 0.12, 0.12, "加速度序列\n2048 点", fc="#FFFFFF", ec=BLUE, fs=9.5, bold=True)
    box(ax, 0.23, y, 0.13, 0.12, "1D 卷积\n下采样", fc=GRAY2, ec=BLUE, fs=9.5)
    box(ax, 0.42, y, 0.13, 0.12, "双向 LSTM\n时序记忆", fc=GRAY2, ec=BLUE, fs=9.5)
    box(ax, 0.61, y, 0.13, 0.12, "注意力池化\n关键时段", fc=GRAY2, ec=BLUE, fs=9.5)
    box(ax, 0.80, y, 0.13, 0.12, "标量融合\n回归输出", fc="#FFFFFF", ec=BLUE, fs=9.5)
    for x1, x2 in [(0.17, 0.23), (0.36, 0.42), (0.55, 0.61), (0.74, 0.80)]:
        arrow(ax, (x1, y + 0.06), (x2, y + 0.06), color=BLUE)
    ax.text(0.05, y + 0.17, "LSTM 路线：门控记忆结构捕捉前后时序依赖", color=BLUE_DARK, fontsize=11, weight="bold")

    # WaveNet lane
    y = 0.29
    box(ax, 0.05, y, 0.12, 0.12, "加速度序列\n4096 点", fc="#FFFFFF", ec=TEAL, fs=9.5, bold=True)
    dil_xs = [0.23, 0.35, 0.47, 0.59]
    dil_labels = ["d=1", "d=2", "d=4", "d=8"]
    for x, label in zip(dil_xs, dil_labels):
        box(ax, x, y, 0.08, 0.12, f"膨胀卷积\n{label}", fc=GRAY2, ec=TEAL, fs=8.8)
    box(ax, 0.71, y, 0.13, 0.12, "残差/跳跃连接\n多尺度汇聚", fc=GRAY2, ec=TEAL, fs=9.2)
    box(ax, 0.88, y, 0.08, 0.12, "回归\n输出", fc="#FFFFFF", ec=TEAL, fs=9.5)
    for x1, x2 in [(0.17, 0.23), (0.31, 0.35), (0.43, 0.47), (0.55, 0.59), (0.67, 0.71), (0.84, 0.88)]:
        arrow(ax, (x1, y + 0.06), (x2, y + 0.06), color=TEAL)
    ax.text(0.05, y + 0.17, "WaveNet 路线：膨胀卷积扩大感受野并支持并行训练", color=TEAL, fontsize=11, weight="bold")
    save(fig, "fig07_sequence_baselines")


def fig08_dataset_distribution():
    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.26)
    ax1 = fig.add_subplot(gs[0])
    subsets = ["训练集", "验证集", "测试集"]
    samples = np.array([67260, 3420, 3116])
    waves = np.array([163, 17, 14])
    x = np.arange(len(subsets))
    ax1.bar(x - 0.17, samples / 1000, width=0.34, color=BLUE, label="样本数 / 千")
    ax1.bar(x + 0.17, waves, width=0.34, color=TEAL, label="唯一地震波数")
    ax1.set_xticks(x, subsets)
    ax1.set_title("图 8 数据集划分与漂移区间分布", loc="left", fontsize=15, color=BLUE_DARK, weight="bold")
    ax1.set_ylabel("数量")
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False)
    for xi, s, wv in zip(x, samples, waves):
        ax1.text(xi - 0.17, s / 1000 + 1.5, f"{s}", ha="center", fontsize=8)
        ax1.text(xi + 0.17, wv + 1.5, f"{wv}", ha="center", fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    bins = ["<0.001", "0.001-\n0.002", "0.002-\n0.003", "0.003-\n0.005", "0.005-\n0.010", "≥0.010"]
    train = np.array([10218, 16385, 13477, 17912, 9268, 0])
    val = np.array([908, 1100, 524, 466, 270, 152])
    test = np.array([665, 1309, 621, 378, 143, 0])
    ind = np.arange(len(bins))
    ax2.bar(ind, train, color=BLUE, label="训练集")
    ax2.bar(ind, val, bottom=train, color=TEAL, label="验证集")
    ax2.bar(ind, test, bottom=train + val, color=AMBER, label="测试集")
    ax2.set_xticks(ind, bins)
    ax2.set_ylabel("样本数")
    ax2.set_title("最大层间位移角分布", loc="left", fontsize=12, color=BLUE_DARK, weight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, ncol=3)
    ax2.annotate("训练/测试缺少 ≥0.010 样本", xy=(5, 200), xytext=(3.6, 6900), arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9)
    save(fig, "fig08_dataset_distribution")


def fig09_model_performance():
    models = ["2D-CNN", "Tail-aware\n2D-CNN", "WaveNet", "LSTM", "LightGBM", "CatBoost", "XGBoost", "RF", "MLP"]
    r2 = np.array([0.907476, 0.900106, 0.866375, 0.858619, 0.779447, 0.775384, 0.751262, 0.684975, 0.543932])
    mae = np.array([0.000288331, 0.000298733, 0.000355626, 0.000374142, 0.000421465, 0.000431681, 0.000459898, 0.000557378, 0.000673718])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1, 1]})
    colors = [BLUE if i == 0 else TEAL if i in [1, 2, 3] else AMBER if i in [4, 5, 6, 7] else RED for i in range(len(models))]
    axs[0].barh(models[::-1], r2[::-1], color=colors[::-1])
    axs[0].set_xlim(0.45, 0.94)
    axs[0].set_xlabel("$R^2$")
    axs[0].set_title("图 9 各模型整体测试性能", loc="left", fontsize=15, color=BLUE_DARK, weight="bold")
    axs[0].grid(axis="x", alpha=0.25)
    axs[0].spines[["top", "right"]].set_visible(False)
    for y, v in enumerate(r2[::-1]):
        axs[0].text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=8.5)
    axs[1].barh(models[::-1], mae[::-1] * 1e4, color=colors[::-1])
    axs[1].set_xlabel("MAE × 10⁴")
    axs[1].grid(axis="x", alpha=0.25)
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].set_title("平均绝对误差", loc="left", fontsize=12, color=BLUE_DARK, weight="bold")
    for y, v in enumerate((mae * 1e4)[::-1]):
        axs[1].text(v + 0.05, y, f"{v:.2f}", va="center", fontsize=8.5)
    save(fig, "fig09_overall_model_performance")


def fig10_tail_reliability():
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    bins = ["<0.001", "0.001-\n0.002", "0.002-\n0.003", "0.003-\n0.005", "0.005-\n0.010"]
    mae = np.array([0.000107104, 0.000220258, 0.000358122, 0.000504894, 0.000878713]) * 1e4
    under = np.array([42.4, 54.0, 51.2, 63.0, 81.8])
    x = np.arange(len(bins))
    ax1.bar(x, mae, color=BLUE, alpha=0.82, label="MAE × 10⁴")
    ax1.set_xticks(x, bins)
    ax1.set_ylabel("MAE × 10⁴")
    ax1.set_title("图 10 中高漂移区间低估风险", loc="left", fontsize=15, color=BLUE_DARK, weight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.25)
    ax1b = ax1.twinx()
    ax1b.plot(x, under, color=RED, marker="o", lw=2.2, label="欠预测率")
    ax1b.set_ylabel("欠预测率 / %")
    ax1b.set_ylim(0, 100)
    ax1b.spines[["top"]].set_visible(False)
    ax1b.annotate("0.005-0.010 区间\n欠预测率 81.8%", xy=(4, 81.8), xytext=(2.9, 92), arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    thresholds = ["0.003", "0.005", "0.007"]
    f1 = {
        "2D-CNN": [0.861, 0.680, 0.261],
        "Tail-aware": [0.851, 0.659, 0.167],
        "WaveNet": [0.816, 0.638, 0.080],
        "LSTM": [0.815, 0.739, 0.250],
    }
    colors = {"2D-CNN": BLUE, "Tail-aware": TEAL, "WaveNet": AMBER, "LSTM": RED}
    for name, vals in f1.items():
        ax2.plot(thresholds, vals, marker="o", lw=2, color=colors[name], label=name)
    ax2.set_ylim(0, 0.95)
    ax2.set_xlabel("高漂移阈值")
    ax2.set_ylabel("F1")
    ax2.set_title("阈值识别能力随阈值升高而下降", loc="left", fontsize=12, color=BLUE_DARK, weight="bold")
    ax2.grid(alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False)
    save(fig, "fig10_tail_reliability")


def main():
    fig01_workflow()
    fig02_ground_motion_pipeline()
    fig03_arias_window()
    fig04_frame_encoding()
    fig05_input_representations()
    fig06_multimodal_cnn()
    fig07_sequence_baselines()
    fig08_dataset_distribution()
    fig09_model_performance()
    fig10_tail_reliability()
    print(f"saved_png={len(list(PNG.glob('*.png')))}")
    print(f"saved_svg={len(list(SVG.glob('*.svg')))}")
    print(OUT.resolve())


if __name__ == "__main__":
    main()
