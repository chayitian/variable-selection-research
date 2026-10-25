"""逐步选择法可运行实验。

三组实验：
  实验一  合成稀疏数据：三种变体的选择行为与预测表现（统计性质）
  实验二  小规模穷举对照：贪心次优性的量化与算法不变量（工程性质）
  实验三  选择不稳定性：抽样扰动与共线场景（Miller 1984 记录、Meinshausen & Bühlmann 2010 量化的核心缺陷）

运行方式（在仓库根目录执行）：
    python methods/03-stepwise-selection/experiment.py
"""

import csv
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from lib import make_synthetic, mse, set_style, support_recovery, save_fig  # noqa: E402
from stepwise import StepwiseSelector  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

# 三种变体及其判据参数（与 README §2.2-2.4 对应）
VARIANTS = [("forward", "前向选择", {"p_in": 0.05}),
            ("stepwise", "逐步回归", {"p_in": 0.05, "p_out": 0.10}),
            ("backward", "后向消除", {"p_out": 0.10})]


def _write_json(name, payload):
    # 约定：JSON 存汇总指标（人读友好），CSV 存逐行明细（便于再分析）；
    # ensure_ascii=False 保证中文内容可读
    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _write_csv(name, header, rows):
    os.makedirs(RES_DIR, exist_ok=True)
    path = os.path.join(RES_DIR, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def experiment1():
    """实验一：合成稀疏数据上的三种变体（统计性质）。

    数据与 01-least-squares 模块实验一完全相同（n=100, p=20, 非零 5，
    rho=0.3, seed=0），保证跨模块可比；测试集用 seed=1 生成 2000 个样本。
    对照基线：全模型 OLS（20 变量）与真支撑集 OLS（oracle，5 变量）——
    回答"相比不选择和相比理想选择，逐步法处在什么位置"。
    """
    X, y, beta_true = make_synthetic(n_samples=100, n_features=20,
                                     n_informative=5, rho=0.3,
                                     noise_std=1.0, seed=0)
    Xte, yte, _ = make_synthetic(n_samples=2000, n_features=20,
                                 n_informative=5, rho=0.3,
                                 noise_std=1.0, seed=1)

    def ols_baseline(cols):
        """给定列子集的 OLS 测试 MSE（直接最小二乘，不走选择器）。"""
        A = np.hstack([np.ones((len(y), 1)), X[:, cols]])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        Ate = np.hstack([np.ones((len(yte), 1)), Xte[:, cols]])
        return mse(yte, Ate @ b)

    true_cols = [j for j in range(20) if beta_true[j] != 0]
    baselines = {"full_ols_test_mse": ols_baseline(list(range(20))),
                 "oracle_test_mse": ols_baseline(true_cols)}

    summary = {"setting": {"n_samples": 100, "n_features": 20,
                           "n_informative": 5, "rho": 0.3,
                           "noise_std": 1.0, "seeds": "train=0, test=1"},
               "baselines": baselines, "variants": {}}
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["C0", "C3", "C2"]
    for (key, label, kw), color in zip(VARIANTS, colors):
        model = StepwiseSelector(method=key, **kw).fit(X, y)
        support = support_recovery(beta_true, model.coef_)
        test_mse = mse(yte, model.predict(Xte))
        summary["variants"][key] = {
            "n_selected": len(model.selected_),
            "selected": model.selected_,
            "support": support,
            "train_mse": model.rss_ / len(y),
            "test_mse": test_mse,
            "r2": model.r2_,
            "n_steps": model.n_steps_,
        }
        rows.append([label, len(model.selected_), support["tp"], support["fp"],
                     support["fn"], support["jaccard"],
                     model.rss_ / len(y), test_mse])

        # 左图：RSS 随步数的下降路径
        axes[0].plot(range(len(model.rss_path_)), model.rss_path_,
                     marker="o", ms=4, color=color, label=label)

    # 右图：支撑集恢复 tp / fp / fn 分组柱状图
    x = np.arange(3)
    width = 0.25
    for i, (key, label, _) in enumerate(VARIANTS):
        s = summary["variants"][key]["support"]
        axes[1].bar(x + (i - 1) * width, [s["tp"], s["fp"], s["fn"]],
                    width, color=colors[i], alpha=0.85, label=label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["tp（选中真非零）", "fp（误选）", "fn（漏选）"])
    axes[1].set_ylabel("变量个数")
    axes[1].set_title("支撑集恢复（真支撑 = 5 个）")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[0].set_xlabel("步数")
    axes[0].set_ylabel("RSS")
    axes[0].set_title("三种变体的 RSS 下降路径")
    axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle("实验一：合成数据上的逐步选择（n=100, p=20, 非零=5）", y=1.03)
    path = save_fig(fig, "fig1_selection_behavior", FIG_DIR)
    _write_json("exp1_variants.json", summary)
    _write_csv("exp1_selection.csv",
               ["variant", "n_selected", "tp", "fp", "fn", "jaccard",
                "train_mse", "test_mse"],
               rows)
    return summary, [path]


def experiment2():
    """实验二：小规模穷举对照（工程性质）。

    p=12，枚举全部 2^12 = 4096 个子集的 RSS，得到每个规模 k 的最优子集
    RSS；前向选择自然产出规模 1..k* 的嵌套路径，将路径上每点 RSS 与
    穷举最优比较，量化贪心次优性。另验证两条算法不变量：
      (a) RSS 路径单调不增（每步都不应变差）；
      (b) 选择器最终系数 = 选中子集上直接 OLS 的系数。
    """
    p = 12
    X, y, beta_true = make_synthetic(n_samples=80, n_features=p,
                                     n_informative=4, rho=0.9,
                                     noise_std=1.0, seed=7)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()

    # 穷举：每个规模 k 的最优 RSS（2^12 个子集，教学规模可接受）
    best_rss = {}
    for k in range(p + 1):
        best = np.inf
        for cols in itertools.combinations(range(p), k):
            if k == 0:
                rss = float(yc @ yc)
            else:
                A = Xc[:, list(cols)]
                b, *_ = np.linalg.lstsq(A, yc, rcond=None)
                rss = float(np.sum((yc - A @ b) ** 2))
            best = min(best, rss)
        best_rss[k] = best

    # 前向路径 vs 穷举最优
    fwd = StepwiseSelector(method="forward", p_in=0.05).fit(X, y)
    k_star = len(fwd.selected_)
    gap_rows = []
    for k in range(1, k_star + 1):
        cols = sorted(h["variable"] for h in fwd.history_[:k])
        A = Xc[:, cols]
        b, *_ = np.linalg.lstsq(A, yc, rcond=None)
        rss = float(np.sum((yc - A @ b) ** 2))
        gap_rows.append([k, rss, best_rss[k],
                         (rss - best_rss[k]) / best_rss[k]])

    # 三种变体在各自规模 k* 处的相对差距
    variant_gaps = {}
    for key, label, kw in VARIANTS:
        m = StepwiseSelector(method=key, **kw).fit(X, y)
        k = len(m.selected_)
        variant_gaps[key] = {"k": k, "rss": m.rss_, "rss_best_k": best_rss[k],
                             "rel_gap": (m.rss_ - best_rss[k]) / best_rss[k]}

    # 不变量 (a)：RSS 路径单调不增
    monotone = bool(all(fwd.rss_path_[i + 1] <= fwd.rss_path_[i] + 1e-9
                        for i in range(len(fwd.rss_path_) - 1)))
    # 不变量 (b)：最终系数 = 选中子集上的直接 OLS
    A = Xc[:, fwd.selected_]
    b_direct, *_ = np.linalg.lstsq(A, yc, rcond=None)
    coef_err = float(np.max(np.abs(fwd.coef_[fwd.selected_] - b_direct)))

    summary = {"setting": {"n_samples": 80, "n_features": p,
                           "n_informative": 4, "rho": 0.3, "seed": 2,
                           "n_subsets_enumerated": 2 ** p},
               "variant_gaps": variant_gaps,
               "forward_k_star": k_star,
               "forward_final_gap": (fwd.rss_ - best_rss[k_star]) / best_rss[k_star],
               "invariants": {"rss_monotone": monotone,
                              "coef_max_abs_err": coef_err}}
    _write_json("exp2_exhaustive.json", summary)
    _write_csv("exp2_gap_path.csv",
               ["k", "rss_forward", "rss_best_k", "rel_gap"], gap_rows)

    # 图 (a)：前向路径 RSS vs 穷举最优；图 (b)：相对差距（线性轴；
    # 差距在多数 k 处为 0、在个别 k 处很大，线性轴才能同时呈现两者）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = [r[0] for r in gap_rows]
    axes[0].plot(ks, [r[1] for r in gap_rows], marker="o", ms=4, color="C0",
                 label="前向选择路径")
    axes[0].plot(ks, [r[2] for r in gap_rows], marker="s", ms=4, color="C3",
                 label="穷举最优（同规模）")
    axes[0].set_xlabel("子集规模 k")
    axes[0].set_ylabel("RSS")
    axes[0].set_title("(a) 前向路径 vs 穷举最优（p=12）")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(ks, [r[3] for r in gap_rows], marker="o", ms=4, color="C2")
    axes[1].set_xlabel("子集规模 k")
    axes[1].set_ylabel("相对差距 (RSS_fwd - RSS_opt) / RSS_opt")
    axes[1].set_title("(b) 贪心次优性：相对差距随 k 的变化")
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle=":")

    fig.suptitle("实验二：贪心次优性的穷举对照", y=1.03)
    path = save_fig(fig, "fig2_exhaustive_gap", FIG_DIR)
    return summary, [path]


def _make_masking(n, p, n_true, rho_pair, noise, seed):
    """构造"共线掩蔽"合成回归数据（局部教学生成器）。

    变量 0 为真实预测变量（系数 2.0）；变量 1 是其相关 rho_pair 的
    "影子"（系数为 0，仅通过与变量 0 的相关性获得边际关联）；变量
    2..n_true 为其他真实预测变量（系数 2.0）；其余为噪声变量。
    低信噪比下，贪心难以在边际上区分真实变量与其影子。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    w = rng.standard_normal(n)
    cols = [z, rho_pair * z + np.sqrt(1.0 - rho_pair ** 2) * w]
    cols += [rng.standard_normal(n) for _ in range(p - 2)]
    X = np.column_stack(cols)
    beta = np.zeros(p)
    beta[0] = 2.0
    beta[2:n_true + 1] = 2.0
    y = X @ beta + rng.standard_normal(n) * noise
    return X, y, beta, [0] + list(range(2, n_true + 1))


def experiment3():
    """实验三：选择不稳定性——Miller (1984) 记录、Meinshausen & Bühlmann (2010) 量化的核心缺陷。

    两个场景（均为低信噪比：噪声 3.0，B=50 次重拟合）：
      (a) 抽样不稳定性：同一生成过程、不同种子，考察选择集合的波动——
          真支撑是否稳定命中、假阳性变量如何"换手"；
      (b) 共线掩蔽：一个真实变量 + 一个相关 0.95 的影子变量，考察选择
          在两者之间如何翻转。
    高信噪比对照（噪声 1.0）的结论写入 README §5.3：同一算法在强信号下
    两个场景都完全稳定——不稳定性是信噪比与共线结构的函数，而非算法的
    固有标签。
    """
    B = 50
    n, p, n_inf = 100, 20, 5
    noise = 3.0

    # (a) 抽样不稳定性：同一生成过程、不同种子
    freq = np.zeros(p)
    selections = []
    for b in range(B):
        Xb, yb, _ = make_synthetic(n_samples=n, n_features=p,
                                   n_informative=n_inf, rho=0.3,
                                   noise_std=noise, seed=b)
        m = StepwiseSelector(method="stepwise", p_in=0.05, p_out=0.10).fit(Xb, yb)
        sel = set(m.selected_)
        selections.append(sel)
        for j in sel:
            freq[j] += 1
    freq /= B

    # 选择结果两两 Jaccard 与对真支撑集的平均 Jaccard
    jaccards = []
    for i in range(B):
        for j in range(i + 1, B):
            a, c = selections[i], selections[j]
            union = len(a | c)
            jaccards.append(len(a & c) / union if union else 1.0)
    true_set = set(range(n_inf))
    vs_true = [len(s & true_set) / len(s | true_set) for s in selections]

    # (b) 共线掩蔽：真实变量 0 与影子变量 1
    pair_freq = np.zeros(2)
    for b in range(B):
        Xb, yb, _, _ = _make_masking(n, p, n_inf - 1, 0.95, noise, seed=100 + b)
        m = StepwiseSelector(method="stepwise", p_in=0.05, p_out=0.10).fit(Xb, yb)
        for j in (0, 1):
            if j in m.selected_:
                pair_freq[j] += 1
    pair_freq /= B

    # 对照组：同一算法、高信噪比（噪声 1.0）——验证"不稳定性是信噪比
    # 的函数"：强信号下两个场景都应恢复稳定
    ctrl = {}
    freq_c = np.zeros(p)
    for b in range(B):
        Xb, yb, _ = make_synthetic(n_samples=n, n_features=p,
                                   n_informative=n_inf, rho=0.3,
                                   noise_std=1.0, seed=b)
        m = StepwiseSelector(method="stepwise", p_in=0.05, p_out=0.10).fit(Xb, yb)
        for j in m.selected_:
            freq_c[j] += 1
    ctrl["resample_freq_on_true"] = [float(freq_c[j] / B) for j in range(n_inf)]
    pair_c = np.zeros(2)
    for b in range(B):
        Xb, yb, _, _ = _make_masking(n, p, n_inf - 1, 0.95, 1.0, seed=100 + b)
        m = StepwiseSelector(method="stepwise", p_in=0.05, p_out=0.10).fit(Xb, yb)
        for j in (0, 1):
            if j in m.selected_:
                pair_c[j] += 1
    ctrl["masking_pair_freq"] = {"true_variable_0": float(pair_c[0] / B),
                                 "shadow_variable_1": float(pair_c[1] / B)}

    summary = {
        "setting": {"B": B, "n_samples": n, "n_features": p,
                    "n_informative": n_inf, "rho": 0.3, "noise_std": noise,
                    "method": "stepwise(0.05/0.10)",
                    "seeds_resample": "0..49", "seeds_masking": "100..149",
                    "rho_pair": 0.95},
        "resample_freq": freq.tolist(),
        "true_support": sorted(true_set),
        "freq_on_true_support": [float(freq[j]) for j in sorted(true_set)],
        "mean_n_selected": float(np.mean([len(s) for s in selections])),
        "pairwise_jaccard": {"mean": float(np.mean(jaccards)),
                             "std": float(np.std(jaccards)),
                             "min": float(np.min(jaccards)),
                             "max": float(np.max(jaccards))},
        "jaccard_vs_true": {"mean": float(np.mean(vs_true)),
                            "std": float(np.std(vs_true))},
        "masking_pair_freq": {"true_variable_0": float(pair_freq[0]),
                              "shadow_variable_1": float(pair_freq[1])},
        "control_noise_1.0": ctrl,
    }
    _write_json("exp3_stability.json", summary)
    _write_csv("exp3_frequencies.csv",
               ["variable", "is_true_support", "freq_resample"],
               [[j, int(j in true_set), freq[j]] for j in range(p)])

    # 图 (a)：重抽样下的变量选择频率；图 (b)：共线掩蔽对的选择频率
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["C3" if j in true_set else "C0" for j in range(p)]
    axes[0].bar(range(p), freq, color=colors, alpha=0.85)
    axes[0].set_xlabel("变量编号（0-4 为真实非零，红色）")
    axes[0].set_ylabel("被选中频率（B=50）")
    axes[0].set_title("(a) 抽样不稳定性：真支撑稳定，假阳性换手")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(["真实变量 0", "影子变量 1（系数为 0）"], pair_freq,
                color=["C3", "C0"], alpha=0.85)
    axes[1].set_ylabel("被选中频率（B=50）")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) 共线掩蔽：相关 0.95 的一对变量")
    for i, v in enumerate(pair_freq):
        axes[1].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=10)

    fig.suptitle("实验三：逐步选择的不稳定性（Miller 1984 记录、M&B 2010 量化）", y=1.03)
    path = save_fig(fig, "fig3_selection_instability", FIG_DIR)
    return summary, [path]


def main():
    # 以下三行是控制台快速摘要；正式结果以 results/ 下的 JSON（汇总）与
    # CSV（逐行明细）为准，供文档引用与再分析
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    s1, _ = experiment1()
    v = s1["variants"]
    print("[实验一] 前向 k=%d(J=%.2f) 逐步 k=%d(J=%.2f) 后向 k=%d(J=%.2f) | 测试MSE 前向=%.3f 逐步=%.3f 后向=%.3f | 对照 全模型=%.3f oracle=%.3f"
          % (v["forward"]["n_selected"], v["forward"]["support"]["jaccard"],
             v["stepwise"]["n_selected"], v["stepwise"]["support"]["jaccard"],
             v["backward"]["n_selected"], v["backward"]["support"]["jaccard"],
             v["forward"]["test_mse"], v["stepwise"]["test_mse"],
             v["backward"]["test_mse"],
             s1["baselines"]["full_ols_test_mse"],
             s1["baselines"]["oracle_test_mse"]))

    s2, _ = experiment2()
    print("[实验二] 前向 k*=%d，与穷举最优相对差距=%.4f；不变量：RSS 单调=%s，系数一致误差=%.2e"
          % (s2["forward_k_star"], s2["forward_final_gap"],
             s2["invariants"]["rss_monotone"],
             s2["invariants"]["coef_max_abs_err"]))

    s3, _ = experiment3()
    pj = s3["pairwise_jaccard"]
    mp = s3["masking_pair_freq"]
    print("[实验三] 重抽样(噪声3.0)：真支撑平均频率=%.3f，两两 Jaccard 均值=%.3f（min=%.3f）；共线掩蔽：真实变量=%.3f 影子变量=%.3f"
          % (np.mean(s3["freq_on_true_support"]), pj["mean"], pj["min"],
             mp["true_variable_0"], mp["shadow_variable_1"]))

    print("实验完成：3 张图已存至 figures/，数值结果已存至 results/")


if __name__ == "__main__":
    main()
