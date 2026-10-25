"""全子集选择与分支定界可运行实验。

三组实验：
  实验一  合成稀疏数据：最优子集 vs 逐步法 vs 全模型/oracle（统计性质）
  实验二  多实例对照：分支定界与穷举的结果一致性及评估次数（工程性质）
  实验三  适用边界：评估次数随 p 的增长 + 重抽样不稳定性（规模与稳定性）

运行方式（在仓库根目录执行）：
    python methods/04-best-subset-selection/experiment.py
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from lib import make_synthetic, mse, set_style, support_recovery, save_fig  # noqa: E402
from best_subset import (BestSubsetSelector, _center, exhaustive_best_subsets,  # noqa: E402
                         branch_and_bound, fit_subset_rss)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")


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


def _forward_select(X, y, p_in=0.05):
    """精简版前向选择（实验一对照基线；完整实现见 03-stepwise-selection 模块）。

    与 03 模块使用同一判据（偏 F 检验）与同一停止规则，仅保留本实验所需部分。
    """
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    selected = []
    for _ in range(p):
        _, rss_base, rank_base = fit_subset_rss(Xc, yc, selected)
        best = None
        for j in range(p):
            if j in selected:
                continue
            _, rss_full, rank_full = fit_subset_rss(Xc, yc, selected + [j])
            df_res = n - len(selected) - 2
            if rank_full <= rank_base or df_res <= 0:
                continue
            if rss_full >= rss_base - 1e-12:
                continue
            F = (rss_base - rss_full) / (rss_full / df_res)
            pval = float(stats.f.sf(F, 1, df_res))
            if pval < p_in and (best is None or pval < best[1]):
                best = (j, pval)
        if best is None:
            break
        selected.append(best[0])
    return sorted(selected)


def _ols_on(X, y, cols):
    """给定列子集的 OLS 拟合（中心化实现），返回预测函数与 R^2。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xc, yc = _center(X, y)
    beta_s, rss, _ = fit_subset_rss(Xc, yc, cols)
    beta = np.zeros(X.shape[1])
    if len(cols):
        beta[list(cols)] = beta_s
    xbar, ybar = X.mean(axis=0), y.mean()
    intercept = float(ybar - xbar @ beta)

    def predict(Xnew):
        return np.asarray(Xnew, dtype=float) @ beta + intercept

    ss_tot = float(yc @ yc)
    return predict, 1.0 - rss / ss_tot


def experiment1():
    """实验一：合成稀疏数据上的三种方法对照（统计性质）。

    数据与 01-least-squares / 03-stepwise-selection 模块实验一完全相同
    （n=100, p=20, 非零 5，rho=0.3, 噪声 1.0，训练 seed=0，测试 seed=1），
    保证跨模块可比。对照：全模型 OLS、真支撑集 OLS（oracle）。
    """
    X, y, beta_true = make_synthetic(n_samples=100, n_features=20,
                                     n_informative=5, rho=0.3,
                                     noise_std=1.0, seed=0)
    Xte, yte, _ = make_synthetic(n_samples=2000, n_features=20,
                                 n_informative=5, rho=0.3,
                                 noise_std=1.0, seed=1)
    true_cols = [j for j in range(20) if beta_true[j] != 0]

    results = {}
    # 最优子集（Cp 定模）
    bs = BestSubsetSelector(method="branch_bound").fit(X, y)
    results["best_subset"] = {
        "k": bs.k_, "selected": bs.selected_,
        "support": support_recovery(beta_true, bs.coef_),
        "test_mse": mse(yte, bs.predict(Xte)),
        "train_mse": bs.rss_ / len(y), "r2": bs.r2_,
        "n_evals": bs.n_evals_}
    # 前向选择（03 模块的对照基线）
    fs_cols = _forward_select(X, y, p_in=0.05)
    fs_pred, fs_r2 = _ols_on(X, y, fs_cols)
    coef_fs = np.zeros(20)
    Xc, yc = _center(X, y)
    b_fs, rss_fs, _ = fit_subset_rss(Xc, yc, fs_cols)
    coef_fs[fs_cols] = b_fs
    results["forward_select"] = {
        "k": len(fs_cols), "selected": fs_cols,
        "support": support_recovery(beta_true, coef_fs),
        "test_mse": mse(yte, fs_pred(Xte)), "r2": fs_r2,
        "train_mse": rss_fs / len(y)}
    # 基线：全模型与 oracle
    full_pred, _ = _ols_on(X, y, list(range(20)))
    results["full_ols"] = {"k": 20, "test_mse": mse(yte, full_pred(Xte))}
    orc_pred, _ = _ols_on(X, y, true_cols)
    results["oracle"] = {"k": 5, "test_mse": mse(yte, orc_pred(Xte))}

    summary = {"setting": {"n_samples": 100, "n_features": 20,
                           "n_informative": 5, "rho": 0.3,
                           "noise_std": 1.0, "seeds": "train=0, test=1"},
               "cp_table": {str(k): v for k, v in bs.cp_by_k_.items()},
               "variants": results}
    _write_json("exp1_variants.json", summary)
    _write_csv("exp1_selection.csv",
               ["variant", "k", "tp", "fp", "fn", "jaccard", "test_mse"],
               [["best_subset", results["best_subset"]["k"],
                 results["best_subset"]["support"]["tp"],
                 results["best_subset"]["support"]["fp"],
                 results["best_subset"]["support"]["fn"],
                 results["best_subset"]["support"]["jaccard"],
                 results["best_subset"]["test_mse"]],
                ["forward_select", results["forward_select"]["k"],
                 results["forward_select"]["support"]["tp"],
                 results["forward_select"]["support"]["fp"],
                 results["forward_select"]["support"]["fn"],
                 results["forward_select"]["support"]["jaccard"],
                 results["forward_select"]["test_mse"]]])

    # 图 (a)：Cp-k 曲线（最小值即选定规模）；图 (b)：支撑集恢复对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ks = sorted(bs.cp_by_k_)
    cps = [bs.cp_by_k_[k] for k in ks]
    axes[0].plot(ks, cps, marker="o", ms=4, color="C0")
    axes[0].axvline(bs.k_, color="C3", linestyle="--", linewidth=1.2,
                    label="Cp 最小：k=%d" % bs.k_)
    axes[0].set_xlabel("子集规模 $k$")
    axes[0].set_ylabel("Mallows $C_p$")
    axes[0].set_title("(a) $C_p$ 曲线与选定规模")
    axes[0].legend(loc="upper right", fontsize=9)

    labels = ["tp（选中真非零）", "fp（误选）", "fn（漏选）"]
    x = np.arange(3)
    width = 0.35
    for i, (key, label, color) in enumerate(
            [("best_subset", "最优子集（Cp）", "C0"),
             ("forward_select", "前向选择（03 模块）", "C2")]):
        s = results[key]["support"]
        axes[1].bar(x + (i - 0.5) * width, [s["tp"], s["fp"], s["fn"]],
                    width, color=color, alpha=0.85, label=label)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("变量个数")
    axes[1].set_title("(b) 支撑集恢复（真支撑 = 5 个）")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle("实验一：合成数据上的全子集选择（n=100, p=20, 非零=5）", y=1.03)
    path = save_fig(fig, "fig1_selection_behavior", FIG_DIR)
    return summary, [path]


def experiment2():
    """实验二：分支定界 vs 穷举（工程性质）。

    多组随机实例上验证两者给出的每层最优 RSS 完全一致（正确性），
    并统计 RSS 评估次数（效率）。评估次数跨 k 去重：每个子集只算一次拟合。
    """
    settings = [(12, 4), (12, 2), (14, 3), (14, 5)]
    n_per_setting = 5
    rows = []
    all_consistent = True
    max_rss_diff = 0.0
    ratios = []
    for p, n_inf in settings:
        for rep in range(n_per_setting):
            X, y, _ = make_synthetic(n_samples=100, n_features=p,
                                     n_informative=n_inf, rho=0.3,
                                     noise_std=1.0, seed=100 + rep)
            be, ne, _ = exhaustive_best_subsets(X, y)
            bb, nb, _ = branch_and_bound(X, y)
            diff = max(abs(be[k][1] - bb[k][1]) for k in be)
            max_rss_diff = max(max_rss_diff, diff)
            consistent = diff < 1e-9
            all_consistent &= consistent
            ratios.append(ne / nb)
            rows.append([p, n_inf, rep, ne, nb, ne / nb,
                         1 if consistent else 0, diff])

    summary = {"setting": {"instances": len(rows),
                           "configs": "%s，各 %d 个实例"
                                      % (str(settings), n_per_setting),
                           "n_samples": 100, "rho": 0.3, "noise_std": 1.0},
               "all_consistent": bool(all_consistent),
               "max_rss_abs_diff": max_rss_diff,
               "eval_ratio": {"mean": float(np.mean(ratios)),
                              "min": float(np.min(ratios)),
                              "max": float(np.max(ratios))}}
    _write_json("exp2_branchbound.json", summary)
    _write_csv("exp2_eval_counts.csv",
               ["p", "n_informative", "rep", "evals_exhaustive",
                "evals_branch_bound", "ratio", "consistent", "max_rss_diff"],
               rows)

    # 图 (a)：按配置聚合的评估次数（对数轴，柱=均值，点=个体实例）；
    # 图 (b)：各实例各层 RSS 一致性
    configs = sorted({(r[0], r[1]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(configs))
    width = 0.35
    ex_means = [np.mean([r[3] for r in rows if (r[0], r[1]) == c])
                for c in configs]
    bb_means = [np.mean([r[4] for r in rows if (r[0], r[1]) == c])
                for c in configs]
    axes[0].bar(x - width / 2, ex_means, width, color="C3", alpha=0.85,
                label="穷举")
    axes[0].bar(x + width / 2, bb_means, width, color="C0", alpha=0.85,
                label="分支定界")
    rng = np.random.default_rng(0)
    for i, c in enumerate(configs):
        ex_pts = [r[3] for r in rows if (r[0], r[1]) == c]
        bb_pts = [r[4] for r in rows if (r[0], r[1]) == c]
        axes[0].scatter(x[i] - width / 2 + rng.uniform(-0.06, 0.06, len(ex_pts)),
                        ex_pts, s=14, color="C3", edgecolor="k", linewidth=0.3,
                        zorder=3)
        axes[0].scatter(x[i] + width / 2 + rng.uniform(-0.06, 0.06, len(bb_pts)),
                        bb_pts, s=14, color="C0", edgecolor="k", linewidth=0.3,
                        zorder=3)
        axes[0].text(x[i], max(ex_means) * 1.6,
                     "%.0fx" % (ex_means[i] / bb_means[i]),
                     ha="center", fontsize=9, color="C3")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["p=%d\n非零=%d" % c for c in configs], fontsize=9)
    axes[0].set_ylabel("RSS 评估次数（对数轴）")
    axes[0].set_ylim(top=max(ex_means) * 4)
    axes[0].set_title("(a) 评估次数：穷举 vs 分支定界（标注为缩减倍数）")
    axes[0].legend(loc="lower right", fontsize=9)

    # 重新跑一遍以取得逐层 RSS 对（数据量小，直接复算）
    xs, ys = [], []
    for p, n_inf in settings[:2]:
        for rep in range(2):
            X, y, _ = make_synthetic(n_samples=100, n_features=p,
                                     n_informative=n_inf, rho=0.3,
                                     noise_std=1.0, seed=100 + rep)
            be, _, _ = exhaustive_best_subsets(X, y)
            bb, _, _ = branch_and_bound(X, y)
            for k in be:
                xs.append(be[k][1])
                ys.append(bb[k][1])
    axes[1].scatter(xs, ys, s=22, color="C0", alpha=0.8)
    lim = [0, max(xs + ys) * 1.05]
    axes[1].plot(lim, lim, color="C3", linestyle="--", linewidth=1.0,
                 label="y = x")
    axes[1].set_xlabel("穷举的最优 RSS")
    axes[1].set_ylabel("分支定界的最优 RSS")
    axes[1].set_title("(b) 逐层最优 RSS 一致性（%d 个点）" % len(xs))
    axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle("实验二：分支定界的正确性与效率", y=1.03)
    path = save_fig(fig, "fig2_branchbound_vs_exhaustive", FIG_DIR)
    return summary, [path]


def experiment3():
    """实验三：适用边界（规模增长 + 不稳定性）。

    (a) 评估次数随 p 的增长：穷举指数增长 vs 分支定界的实际增长；
    (b) 重抽样不稳定性：B=50 次重拟合（噪声 3.0），考察 Cp 选定规模、
        支撑集恢复、变量选择频率与两两 Jaccard——全局最优同样不稳定
        （与 03-stepwise-selection 模块 §5.3 的结论并列）。
    """
    # (a) 规模增长
    scale_rows = []
    for p in [6, 8, 10, 12, 14, 16]:
        X, y, _ = make_synthetic(n_samples=100, n_features=p,
                                 n_informative=5, rho=0.3,
                                 noise_std=1.0, seed=7)
        _, ne, _ = exhaustive_best_subsets(X, y)
        _, nb, _ = branch_and_bound(X, y)
        scale_rows.append([p, 2 ** p - 1, ne, nb, ne / nb])

    # (b) 重抽样不稳定性（低信噪比）
    B = 50
    n, p, n_inf = 100, 20, 5
    freq = np.zeros(p)
    k_stars = []
    sels = []
    for b in range(B):
        Xb, yb, _ = make_synthetic(n_samples=n, n_features=p,
                                   n_informative=n_inf, rho=0.3,
                                   noise_std=3.0, seed=b)
        m = BestSubsetSelector(method="branch_bound", kmax=12).fit(Xb, yb)
        k_stars.append(m.k_)
        sel = set(m.selected_)
        sels.append(sel)
        for j in sel:
            freq[j] += 1
    freq /= B
    true_set = set(range(n_inf))
    # 选择结果两两 Jaccard 与对真支撑集的平均 Jaccard
    jaccards = []
    for i in range(B):
        for j in range(i + 1, B):
            u = len(sels[i] | sels[j])
            jaccards.append(len(sels[i] & sels[j]) / u if u else 1.0)
    vs_true = [len(s & true_set) / len(s | true_set) for s in sels]

    # 对照组：尺度不变性核查——同一 X 与同一噪声实现，噪声放大 3 倍，
    # k >= 真支撑规模段的 Cp 逐点不变、选择不变：不稳定性不是信噪比问题
    X0, y1, _ = make_synthetic(n_samples=n, n_features=p, n_informative=n_inf,
                               rho=0.3, noise_std=1.0, seed=0)
    _, y3, _ = make_synthetic(n_samples=n, n_features=p, n_informative=n_inf,
                              rho=0.3, noise_std=3.0, seed=0)
    m1 = BestSubsetSelector(method="branch_bound", kmax=12).fit(X0, y1)
    m3 = BestSubsetSelector(method="branch_bound", kmax=12).fit(X0, y3)
    cp_identical = all(abs(m1.cp_by_k_[k] - m3.cp_by_k_[k]) < 1e-9
                       for k in m1.cp_by_k_ if k >= n_inf)
    same_selection = m1.selected_ == m3.selected_

    summary = {
        "setting": {"scale": {"p_list": [6, 8, 10, 12, 14, 16],
                              "n_samples": 100, "n_informative": 5,
                              "rho": 0.3, "noise_std": 1.0, "seed": 7},
                    "instability": {"B": B, "n_samples": n, "n_features": p,
                                    "n_informative": n_inf, "rho": 0.3,
                                    "noise_std": 3.0, "seeds": "0..49",
                                    "kmax": 12}},
        "scale": [{"p": r[0], "exhaustive_evals": r[1],
                   "branch_bound_evals": r[3], "ratio": r[4]} for r in scale_rows],
        "instability": {"k_star_mean": float(np.mean(k_stars)),
                        "k_star_hist": {str(k): int(k_stars.count(k))
                                        for k in sorted(set(k_stars))},
                        "resample_freq": freq.tolist(),
                        "freq_on_true_support": [float(freq[j])
                                                 for j in range(n_inf)],
                        "pairwise_jaccard": {"mean": float(np.mean(jaccards)),
                                             "min": float(np.min(jaccards)),
                                             "max": float(np.max(jaccards))},
                        "jaccard_vs_true": {"mean": float(np.mean(vs_true)),
                                            "std": float(np.std(vs_true))},
                        "control_scale_invariance": {
                            "cp_identical_for_k_ge_true": bool(cp_identical),
                            "same_selection": bool(same_selection),
                            "k_star_noise1": m1.k_, "k_star_noise3": m3.k_,
                            "max_cp_abs_diff_k_ge_true": float(max(
                                abs(m1.cp_by_k_[k] - m3.cp_by_k_[k])
                                for k in m1.cp_by_k_ if k >= n_inf))}}}
    _write_json("exp3_scale.json", summary)
    _write_csv("exp3_frequencies.csv",
               ["variable", "is_true_support", "freq_resample"],
               [[j, int(j in true_set), freq[j]] for j in range(p)])

    # 图 (a)：评估次数随 p 的增长；图 (b)：重抽样选择频率
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ps = [r[0] for r in scale_rows]
    axes[0].plot(ps, [r[1] for r in scale_rows], marker="s", ms=4,
                 color="C3", label="穷举（$2^p-1$）")
    axes[0].plot(ps, [r[3] for r in scale_rows], marker="o", ms=4,
                 color="C0", label="分支定界（实际）")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("候选变量数 $p$")
    axes[0].set_ylabel("RSS 评估次数（对数轴）")
    axes[0].set_title("(a) 计算规模：指数 vs 分支定界")
    axes[0].legend(loc="upper left", fontsize=9)

    colors = ["C3" if j in true_set else "C0" for j in range(p)]
    axes[1].bar(range(p), freq, color=colors, alpha=0.85)
    axes[1].set_xlabel("变量编号（0-4 为真实非零，红色）")
    axes[1].set_ylabel("被选中频率（B=50）")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) 抽样不稳定性：同一过程、不同种子（噪声 3.0）")

    fig.suptitle("实验三：全子集选择的适用边界", y=1.03)
    path = save_fig(fig, "fig3_scale_and_stability", FIG_DIR)
    return summary, [path]


def main():
    # 以下三行是控制台快速摘要；正式结果以 results/ 下的 JSON（汇总）与
    # CSV（逐行明细）为准，供文档引用与再分析
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    s1, _ = experiment1()
    v = s1["variants"]
    print("[实验一] 最优子集 k=%d(J=%.2f, 测试MSE=%.3f) 前向 k=%d(J=%.2f, 测试MSE=%.3f) | 对照 全模型=%.3f oracle=%.3f | Cp 最小 k=%d"
          % (v["best_subset"]["k"], v["best_subset"]["support"]["jaccard"],
             v["best_subset"]["test_mse"], v["forward_select"]["k"],
             v["forward_select"]["support"]["jaccard"],
             v["forward_select"]["test_mse"], v["full_ols"]["test_mse"],
             v["oracle"]["test_mse"], int(min(s1["cp_table"],
                                               key=lambda k: float(s1["cp_table"][k])))))

    s2, _ = experiment2()
    print("[实验二] %d 个实例 RSS 全部一致=%s（最大差 %.1e），评估缩减倍数 mean=%.1fx min=%.1fx max=%.1fx"
          % (s2["setting"]["instances"], s2["all_consistent"],
             s2["max_rss_abs_diff"], s2["eval_ratio"]["mean"],
             s2["eval_ratio"]["min"], s2["eval_ratio"]["max"]))

    s3, _ = experiment3()
    ins = s3["instability"]
    ctrl = ins["control_scale_invariance"]
    print("[实验三] 规模：p=16 时穷举 %d 次 vs B&B %d 次（%.0fx）；不稳定性（B=50 重抽样）：真支撑平均频率=%.3f，两两 Jaccard 均值=%.3f，Cp 选定 k 均值=%.2f；尺度不变性核查：噪声放大 3 倍后 Cp(k>=5) 相同=%s、选择相同=%s（k*=%d vs %d）"
          % (s3["scale"][-1]["exhaustive_evals"],
             s3["scale"][-1]["branch_bound_evals"], s3["scale"][-1]["ratio"],
             np.mean(ins["freq_on_true_support"]),
             ins["pairwise_jaccard"]["mean"], ins["k_star_mean"],
             ctrl["cp_identical_for_k_ge_true"], ctrl["same_selection"],
             ctrl["k_star_noise1"], ctrl["k_star_noise3"]))

    print("实验完成：3 张图已存至 figures/，数值结果已存至 results/")


if __name__ == "__main__":
    main()
