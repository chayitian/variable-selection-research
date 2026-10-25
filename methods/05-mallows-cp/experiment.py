"""Mallows Cp 与模型定模准则可运行实验。

三组实验：
  实验一  R=200 重复验证 Cp 的无偏性：E[Cp(k)] 与 q=k+1 的关系（统计性质）
  实验二  R=200 重复考察 Cp 定模行为：k* 分布与过选/欠选频率（定模行为）
  实验三  五准则对照 + σ̂² 估计敏感性（适用边界）

运行方式（在仓库根目录执行）：
    python methods/05-mallows-cp/experiment.py
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from lib import make_synthetic, set_style, support_recovery, save_fig  # noqa: E402
from mallows_cp import (ModelCriteriaSelector, aic_value, bic_value,  # noqa: E402
                        cp_curve, exhaustive_best_subsets, fit_subset_rss,
                        gcv_rss, mallows_cp, press_rss, sigma2_hat_full)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

# 实验设置：与 01/03/04 模块同族（n=100, rho=0.3, 噪声 1.0）；
# p=12/非零 4 是为了让 R=200 重复的穷举在可接受时间内完成
N, P, N_INF, RHO, NOISE = 100, 12, 4, 0.3, 1.0
KMAX = 10
R_UNBIAS = 200       # 实验一/二的重复数
R_CRIT = 50          # 实验三的重复数
TRUE_K = N_INF


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


def _cp_tables(R, kmax=KMAX):
    """R 个重复的（最优子集表, σ̂²_full, 真支撑包含情况, 固定模型 Cp）列表。

    固定模型臂：对两个**预先指定**的模型（真支撑 S_true；真支撑去掉最强
    变量后的 S_miss）计算 Cp——没有"从候选中挑选"的动作，直接对应
    README 公式 (4) 的经典无偏性陈述。
    """
    out = []
    for r in range(R):
        X, y, beta = make_synthetic(n_samples=N, n_features=P,
                                    n_informative=N_INF, rho=RHO,
                                    noise_std=NOISE, seed=r)
        best, _ = exhaustive_best_subsets(X, y, kmax=kmax)
        Xc, yc = X - X.mean(axis=0), y - y.mean()
        s2 = sigma2_hat_full(Xc, yc)
        true_cols = [j for j in range(P) if abs(beta[j]) > 1e-8]
        strongest = max(true_cols, key=lambda j: abs(beta[j]))
        miss_cols = [j for j in true_cols if j != strongest]
        fixed = {}
        for name, cols in [("s_true", true_cols), ("s_miss", miss_cols)]:
            _, rss, _ = fit_subset_rss(Xc, yc, cols)
            fixed[name] = mallows_cp(rss, len(cols) + 1, N, s2)
        contains_true = {k: set(true_cols).issubset(set(best[k][0]))
                         for k in best if k > 0}
        out.append((best, s2, contains_true, fixed))
    return out


def experiment1():
    """实验一：理论参考线与选择后的偏离（统计性质）。

    两个臂：
      (a) 最优子集臂：对每个规模 k 的 RSS 最优子集计算 Cp（数据驱动选择），
          报告 R 个重复的均值与标准差——与 q=k+1 参考线比较；
      (b) 固定模型臂：对两个预先指定的模型（真支撑；遗漏最强变量的真支撑
          子集）计算 Cp，报告 R 个重复的均值——公式 (4) 的无选择污染验证。
    """
    tables = _cp_tables(R_UNBIAS)
    cp_by_k = {k: [] for k in range(0, KMAX + 1)}
    contain_by_k = {k: [] for k in range(1, KMAX + 1)}
    fixed_by_name = {"s_true": [], "s_miss": []}
    for best, s2, contains, fixed in tables:
        curve = cp_curve(best, N, s2)
        for k, v in curve.items():
            cp_by_k[k].append(v)
        for k, c in contains.items():
            contain_by_k[k].append(c)
        for name in fixed_by_name:
            fixed_by_name[name].append(fixed[name])

    fixed_summary = {
        name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
               "q": (N_INF + 1) if name == "s_true" else N_INF,
               "model": ("真支撑（预先指定）" if name == "s_true"
                         else "真支撑去最强变量（预先指定）")}
        for name, v in fixed_by_name.items()}

    summary = {
        "setting": {"R": R_UNBIAS, "n_samples": N, "n_features": P,
                    "n_informative": N_INF, "rho": RHO, "noise_std": NOISE,
                    "kmax": KMAX, "seeds": "0..%d" % (R_UNBIAS - 1)},
        "true_k": TRUE_K,
        "cp_mean": {str(k): float(np.mean(v)) for k, v in cp_by_k.items()},
        "cp_std": {str(k): float(np.std(v)) for k, v in cp_by_k.items()},
        "q_reference": {str(k): k + 1 for k in range(0, KMAX + 1)},
        "truth_contained_rate": {str(k): float(np.mean(v))
                                 for k, v in contain_by_k.items()},
        "fixed_model": fixed_summary,
    }
    _write_json("exp1_unbiasedness.json", summary)
    _write_csv("exp1_cp_curves.csv",
               ["replicate", "k", "cp", "cp_s_true", "cp_s_miss"],
               [[r, k, v, t[3]["s_true"], t[3]["s_miss"]]
                for r, t in enumerate(tables)
                for k, v in cp_curve(t[0], N, t[1]).items()])

    # 图 (a)：重复均值 Cp 曲线 + q=k+1 参考线 + ±1 标准差带；
    # 图 (b)：固定模型基准（无选择污染）的 E[Cp] 与各自 q；
    # 图 (c)：5 个单个重复的 Cp 曲线（展示最小点附近的噪声）
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ks = list(range(0, KMAX + 1))
    means = [np.mean(cp_by_k[k]) for k in ks]
    stds = [np.std(cp_by_k[k]) for k in ks]
    axes[0].plot(ks, means, marker="o", ms=4, color="C0",
                 label="$E[C_p(k)]$（%d 个重复）" % R_UNBIAS)
    axes[0].fill_between(ks, [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         color="C0", alpha=0.15, label="±1 标准差")
    axes[0].plot(ks, [k + 1 for k in ks], color="C3", linestyle="--",
                 linewidth=1.2, label="$q = k+1$（理论参考）")
    axes[0].axvline(TRUE_K, color="gray", linestyle=":", linewidth=1.0)
    axes[0].text(TRUE_K + 0.15, max(means) * 0.55, "真支撑规模 k=4",
                 fontsize=9, color="gray")
    axes[0].set_xlabel("子集规模 $k$")
    axes[0].set_ylabel("Mallows $C_p$")
    axes[0].set_title("(a) 最优子集的 $E[C_p(k)]$（含选择）")
    axes[0].legend(loc="upper right", fontsize=9)

    names = ["s_true", "s_miss"]
    labels = ["真支撑\n(q=5)", "去最强变量\n(q=4)"]
    vals = [fixed_summary[n]["mean"] for n in names]
    qs = [fixed_summary[n]["q"] for n in names]
    bars = axes[1].bar(labels, vals, color=["C2", "C3"], alpha=0.85)
    axes[1].set_yscale("log")
    for i, (v, q) in enumerate(zip(vals, qs)):
        axes[1].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=10)
        axes[1].plot([i - 0.4, i + 0.4], [q, q], color="k",
                     linestyle="--", linewidth=1.2)
        axes[1].text(i + 0.42, q, "q=%d" % q, va="center", fontsize=9)
    axes[1].set_ylabel("$E[C_p(S)]$（对数轴）")
    axes[1].set_ylim(top=max(vals) * 8)
    axes[1].set_title("(b) 固定模型基准（无选择污染）")

    for r in range(5):
        curve = cp_curve(tables[r][0], N, tables[r][1])
        axes[2].plot(list(curve), list(curve.values()), marker="o", ms=3,
                     alpha=0.75, label="重复 %d（最小 k=%d）"
                     % (r, min(curve, key=lambda k: curve[k])))
    axes[2].axvline(TRUE_K, color="gray", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("子集规模 $k$")
    axes[2].set_ylabel("Mallows $C_p$")
    axes[2].set_title("(c) 单个重复的 $C_p$ 曲线")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.suptitle("实验一：$C_p$ 的理论参考线与选择后的偏离（n=100, p=12, 非零 4, R=%d）"
                 % R_UNBIAS, y=1.03)
    path = save_fig(fig, "fig1_cp_unbiasedness", FIG_DIR)
    return summary, [path]


def experiment2():
    """实验二：Cp 的定模行为（过选倾向）。

    R 个重复下记录 Cp 选定规模 k*，报告分布与欠选/恰好/过选频率，
    并与 04 模块 §5.3 的不稳定性结论互参。
    """
    tables = _cp_tables(R_UNBIAS)
    k_stars = []
    for best, s2, _, _ in tables:
        curve = cp_curve(best, N, s2)
        k_stars.append(min(curve, key=lambda k: (curve[k], k)))

    counts = {k: int(k_stars.count(k)) for k in sorted(set(k_stars))}
    p_under = float(np.mean([k < TRUE_K for k in k_stars]))
    p_exact = float(np.mean([k == TRUE_K for k in k_stars]))
    p_over = float(np.mean([k > TRUE_K for k in k_stars]))

    summary = {
        "setting": {"R": R_UNBIAS, "n_samples": N, "n_features": P,
                    "n_informative": N_INF, "rho": RHO, "noise_std": NOISE,
                    "kmax": KMAX, "seeds": "0..%d" % (R_UNBIAS - 1)},
        "true_k": TRUE_K,
        "k_star_mean": float(np.mean(k_stars)),
        "k_star_std": float(np.std(k_stars)),
        "k_star_counts": {str(k): v for k, v in counts.items()},
        "p_under": p_under, "p_exact": p_exact, "p_over": p_over,
    }
    _write_json("exp2_selection.json", summary)
    _write_csv("exp2_kstar.csv",
               ["replicate", "k_star", "relation"],
               [[r, k, "under" if k < TRUE_K else ("exact" if k == TRUE_K
                                                   else "over")]
                for r, k in enumerate(k_stars)])

    # 图 (a)：k* 分布直方图（标注真支撑规模与均值）；
    # 图 (b)：欠选/恰好/过选频率 + 均值 k*
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = sorted(counts)
    axes[0].bar(ks, [counts[k] for k in ks], color="C0", alpha=0.85)
    axes[0].axvline(TRUE_K, color="C3", linestyle="--", linewidth=1.2,
                    label="真支撑规模 k=4")
    axes[0].axvline(np.mean(k_stars), color="C2", linestyle="-",
                    linewidth=1.2, label="均值 k*=%.2f" % np.mean(k_stars))
    axes[0].set_xlabel("$C_p$ 选定规模 $k^\\ast$")
    axes[0].set_ylabel("重复次数（共 %d）" % R_UNBIAS)
    axes[0].set_title("(a) 选定规模的分布")
    axes[0].legend(loc="upper right", fontsize=9)

    labels = ["欠选\n(k*<4)", "恰好\n(k*=4)", "过选\n(k*>4)"]
    vals = [p_under, p_exact, p_over]
    colors = ["C2", "C3", "C0"]
    axes[1].bar(labels, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        axes[1].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("频率（R=%d）" % R_UNBIAS)
    axes[1].set_ylim(0, max(vals) * 1.25)
    axes[1].set_title("(b) 欠选/恰好/过选频率")

    fig.suptitle("实验二：$C_p$ 定模的过选倾向", y=1.03)
    path = save_fig(fig, "fig2_kstar_distribution", FIG_DIR)
    return summary, [path]


def experiment3():
    """实验三：五准则对照 + σ̂² 估计敏感性（适用边界）。

    (a) 同一子集表上应用五种准则（Cp/AIC/BIC/PRESS/GCV），比较定模行为；
    (b) Cp 在三种 σ̂² 估计（全模型/选定模型/真实噪声方差）下的定模行为。
    """
    tables = _cp_tables(R_CRIT)

    def kstar_for(best, s2, criterion):
        """给定子集表与 σ̂²，返回该准则选定的规模 k*（闭包使用当前 X, y）。"""
        if criterion == "press":
            curve = {k: press_rss(X, y, best[k][0]) for k in best}
        elif criterion == "gcv":
            curve = {k: gcv_rss(X, y, best[k][0]) for k in best}
        elif criterion == "aic":
            curve = {k: aic_value(best[k][1], k + 1, N) for k in best}
        elif criterion == "bic":
            curve = {k: bic_value(best[k][1], k + 1, N) for k in best}
        else:
            curve = cp_curve(best, N, s2)
        return min(curve, key=lambda k: (curve[k], k))

    # (a) 五准则
    crit_names = ["cp", "aic", "bic", "press", "gcv"]
    crit_k = {c: [] for c in crit_names}
    # (b) σ̂² 变体（full / selected / true / 2x_true）
    s2_k = {"full": [], "selected": [], "true": [], "2x_true": []}
    rows = []
    for r, (best, s2, _, _) in enumerate(tables):
        # 重新生成同一份数据（与 _cp_tables 的 seed 一致），供 press/gcv 使用
        X, y, _ = make_synthetic(n_samples=N, n_features=P, n_informative=N_INF,
                                 rho=RHO, noise_std=NOISE, seed=r)
        Xc, yc = X - X.mean(axis=0), y - y.mean()
        for c in crit_names:
            crit_k[c].append(kstar_for(best, s2, c))
        # σ̂² 变体
        k_full = kstar_for(best, s2, "cp")
        s2_k["full"].append(k_full)
        # selected：用 full 的 k* 对应子集重新估计 σ̂²（循环）
        _, rss0, _ = fit_subset_rss(Xc, yc, best[k_full][0])
        s2_sel = rss0 / (N - k_full - 1)
        s2_k["selected"].append(kstar_for(best, s2_sel, "cp"))
        s2_k["true"].append(kstar_for(best, NOISE ** 2, "cp"))
        s2_k["2x_true"].append(kstar_for(best, 2 * NOISE ** 2, "cp"))
        rows.append([r] + [crit_k[c][-1] for c in crit_names]
                    + [s2_k[v][-1] for v in ["full", "selected", "true", "2x_true"]])

    def stats(ks):
        return {"mean": float(np.mean(ks)),
                "p_under": float(np.mean([k < TRUE_K for k in ks])),
                "p_exact": float(np.mean([k == TRUE_K for k in ks])),
                "p_over": float(np.mean([k > TRUE_K for k in ks]))}

    summary = {
        "setting": {"R": R_CRIT, "n_samples": N, "n_features": P,
                    "n_informative": N_INF, "rho": RHO, "noise_std": NOISE,
                    "kmax": KMAX, "seeds": "0..%d" % (R_CRIT - 1)},
        "true_k": TRUE_K,
        "criteria": {c: stats(crit_k[c]) for c in crit_names},
        "sigma2_variants": {v: stats(s2_k[v])
                            for v in ["full", "selected", "true", "2x_true"]},
    }
    _write_json("exp3_criteria.json", summary)
    _write_csv("exp3_kstar_by_config.csv",
               ["replicate", "cp", "aic", "bic", "press", "gcv",
                "s2_full", "s2_selected", "s2_true", "s2_2x_true"],
               rows)

    # 图 (a)：五准则的均值 k*（含欠/恰/过堆叠或误差）；
    # 图 (b)：σ̂² 三变体的均值 k* 与过选频率
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    means = [np.mean(crit_k[c]) for c in crit_names]
    axes[0].bar(crit_names, means, color=["C0", "C0", "C3", "C2", "C2"],
                alpha=0.85)
    axes[0].axhline(TRUE_K, color="gray", linestyle="--", linewidth=1.2,
                    label="真支撑规模 k=4")
    for i, v in enumerate(means):
        axes[0].text(i, v, "%.2f" % v, ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("均值 $k^\\ast$（R=%d）" % R_CRIT)
    axes[0].set_ylim(0, max(means) * 1.3)
    axes[0].set_title("(a) 五准则的选定规模")
    axes[0].legend(loc="upper right", fontsize=9)

    variants = ["full", "selected", "true", "2x_true"]
    vmeans = [np.mean(s2_k[v]) for v in variants]
    pover = [np.mean([k > TRUE_K for k in s2_k[v]]) for v in variants]
    x = np.arange(len(variants))
    width = 0.35
    axes[1].bar(x - width / 2, vmeans, width, color="C0", alpha=0.85,
                label="均值 $k^\\ast$")
    axes[1].bar(x + width / 2, pover, width, color="C3", alpha=0.85,
                label="过选频率")
    axes[1].axhline(TRUE_K, color="gray", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["全模型\n（标准）", "选定模型\n（循环）", "真实 $\\sigma^2$\n（oracle）", "2倍 $\\sigma^2$\n（高估）"])
    for i, v in enumerate(vmeans):
        axes[1].text(i - width / 2, v, "%.2f" % v, ha="center", va="bottom",
                     fontsize=9)
    for i, v in enumerate(pover):
        axes[1].text(i + width / 2, v, "%.2f" % v, ha="center", va="bottom",
                     fontsize=9)
    axes[1].set_ylabel("值")
    axes[1].set_ylim(0, max(vmeans + pover) * 1.3)
    axes[1].set_title("(b) $\\hat\\sigma^2$ 估计方式的影响")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle("实验三：准则对照与 $\\hat\\sigma^2$ 敏感性", y=1.03)
    path = save_fig(fig, "fig3_criteria_and_sigma2", FIG_DIR)
    return summary, [path]


def main():
    # 以下三行是控制台快速摘要；正式结果以 results/ 下的 JSON（汇总）与
    # CSV（逐行明细）为准，供文档引用与再分析
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    s1, _ = experiment1()
    fm = s1["fixed_model"]
    print("[实验一] 最优子集臂：k=4 处 E[Cp]=%.2f（q=5），k=8 处 %.2f（q=9），k=2 处 %.2f（q=3）；固定模型臂：E[Cp(真支撑)]=%.2f（q=5），E[Cp(去最强变量)]=%.1f（q=4）；真支撑被包含率(k>=4)=%.3f"
          % (s1["cp_mean"]["4"], s1["cp_mean"]["8"], s1["cp_mean"]["2"],
             fm["s_true"]["mean"], fm["s_miss"]["mean"],
             np.mean([s1["truth_contained_rate"][str(k)] for k in range(4, KMAX + 1)])))

    s2, _ = experiment2()
    print("[实验二] Cp 定模：k* 均值 %.2f（真支撑 4），恰好 %.3f / 过选 %.3f / 欠选 %.3f"
          % (s2["k_star_mean"], s2["p_exact"], s2["p_over"], s2["p_under"]))

    s3, _ = experiment3()
    crit = s3["criteria"]
    sv = s3["sigma2_variants"]
    print("[实验三] 五准则均值 k*：Cp %.2f / AIC %.2f / BIC %.2f / PRESS %.2f / GCV %.2f；σ̂² 变体：full %.2f / selected %.2f / true %.2f / 2x %.2f"
          % (crit["cp"]["mean"], crit["aic"]["mean"], crit["bic"]["mean"],
             crit["press"]["mean"], crit["gcv"]["mean"],
             sv["full"]["mean"], sv["selected"]["mean"],
             sv["true"]["mean"], sv["2x_true"]["mean"]))

    print("实验完成：3 张图已存至 figures/，数值结果已存至 results/")


if __name__ == "__main__":
    main()
