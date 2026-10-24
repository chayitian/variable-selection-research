"""Fisher 判别分析可运行实验。

三组实验：
  实验一  合成等协方差高斯数据：判别方向、分类精度与 Bayes 最优的关系
  实验二  iris 真实数据：与 sklearn LinearDiscriminantAnalysis 对拍
  实验三  S_w 病态扫描：不截断 / 默认截断 / 收缩三种策略的数值稳定性

运行方式（在仓库根目录执行）：
    python methods/02-fisher-discriminant/experiment.py
"""

import csv
import json
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from lib import (accuracy, balanced_accuracy, confusion_matrix,  # noqa: E402
                 load_iris, make_gaussian_classes, save_fig, set_style)
from fda import FisherDiscriminant, solve_directions  # noqa: E402

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


def experiment1():
    """实验一：合成等协方差高斯数据（统计性质）。

    两类、p=5、共享协方差 Σ=I；以马氏距离 Δ=||μ1-μ2||_Σ 控制可分度。
    等协方差高斯下 Bayes 判别即 LDA，错误率有解析形式 Φ(-Δ/2)，可与
    实测错误率对照；Σ=I 时总体 Fisher 方向与 (μ1-μ2) 同向。
    """
    p = 5
    rng_mu = np.random.default_rng(11)
    base = rng_mu.standard_normal(p)
    base = base / np.linalg.norm(base)
    settings = [("well_separated", 4.0), ("hard", 1.0)]

    summary = {"setting": {"p": p, "n_per_class": 200, "n_test": 2000,
                           "cov": "I", "seeds": "train=0, test=1"},
               "cases": []}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, (name, delta) in zip(axes, settings):
        mu1, mu2 = base * delta / 2, -base * delta / 2
        mus = np.stack([mu1, mu2])
        X, y, _, _ = make_gaussian_classes(200, mus, np.eye(p), seed=0)
        Xte, yte, _, _ = make_gaussian_classes(2000, mus, np.eye(p), seed=1)

        model = FisherDiscriminant().fit(X, y)
        w = model.scalings_[:, 0]

        # 判别方向与 (mu1 - mu2) 的夹角（总体应同向，样本下接近 0）
        d = mu1 - mu2
        cos = float(w @ d / (np.linalg.norm(w) * np.linalg.norm(d)))
        angle = float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))

        train_acc = accuracy(y, model.predict(X))
        test_acc = accuracy(yte, model.predict(Xte))
        bayes_err = float(norm.cdf(-delta / 2))

        # 投影直方图 + 精确决策阈值（δ1=δ2 在投影直线上的位置）
        s_tr = model.transform(X)[:, 0]
        w_hat = w / np.linalg.norm(w)
        a = model.coef_ @ w_hat
        s_star = float(-(model.intercept_[1] - model.intercept_[0])
                       / (a[1] - a[0]))

        for k, color in [(0, "C0"), (1, "C3")]:
            ax.hist(s_tr[y == k], bins=30, density=True, alpha=0.55,
                    color=color, label="类别 %d（n=200）" % k)
        ax.axvline(s_star, color="k", linestyle="--", linewidth=1,
                   label="决策阈值")
        ax.set_xlabel("判别方向上的投影得分")
        ax.set_ylabel("密度")
        ax.set_title("%s：Δ=%.0f，测试错误率 %.3f（Bayes %.3f）"
                     % ("可分设置" if delta > 2 else "难分设置", delta,
                        1 - test_acc, bayes_err))
        ax.legend(loc="upper right", fontsize=9)

        summary["cases"].append({
            "case": name, "delta": delta, "angle_deg": angle,
            "train_acc": train_acc, "test_acc": test_acc,
            "test_err": 1 - test_acc, "bayes_err": bayes_err,
            "threshold": s_star,
        })

    fig.suptitle("实验一：等协方差高斯数据上的 Fisher 判别", y=1.03)
    path = save_fig(fig, "fig1_projection_histograms", FIG_DIR)
    _write_json("exp1_gaussian.json", summary)
    _write_csv("exp1_metrics.csv",
               ["case", "delta", "angle_deg", "train_acc", "test_acc",
                "test_err", "bayes_err", "threshold"],
               [[c["case"], c["delta"], c["angle_deg"], c["train_acc"],
                 c["test_acc"], c["test_err"], c["bayes_err"], c["threshold"]]
                for c in summary["cases"]])
    return summary, [path]


def experiment2():
    """实验二：iris 真实数据与 sklearn 对拍（工程性质）。"""
    X, y, feature_names, target_names = load_iris()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    n_train = 112
    tr, te = order[:n_train], order[n_train:]
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    model = FisherDiscriminant().fit(X_tr, y_tr)

    # 对拍：以 sklearn 为独立 oracle——两者在机器精度内一致即证明实现正确。
    # sklearn 内部用 SVD 求解，与本模块的白化特征值分解是独立代码路径
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    sk = LinearDiscriminantAnalysis().fit(X_tr, y_tr)

    pred_ours = model.predict(X_te)
    pred_sk = sk.predict(X_te)
    agreement = float(np.mean(pred_ours == pred_sk))

    # 方向对比：特征向量只确定到符号，取 |cos| 应接近 1
    scalings_cos = []
    for j in range(model.scalings_.shape[1]):
        v1 = model.scalings_[:, j]
        v2 = sk.scalings_[:, j]
        scalings_cos.append(float(abs(v1 @ v2
                                      / (np.linalg.norm(v1)
                                         * np.linalg.norm(v2)))))
    evr_diff = float(np.max(np.abs(model.explained_variance_ratio_
                                   - sk.explained_variance_ratio_)))
    coef_diff = (float(np.max(np.abs(model.coef_ - sk.coef_)))
                 if model.coef_.shape == sk.coef_.shape else None)

    cm, labels = confusion_matrix(y_te, pred_ours)
    summary = {
        "setting": {"dataset": "sklearn iris", "n_train": int(n_train),
                    "n_test": int(len(te)), "seed": 42},
        "agreement_with_sklearn": agreement,
        "scalings_abs_cos": scalings_cos,
        "explained_variance_ratio_ours": model.explained_variance_ratio_.tolist(),
        "evr_max_diff": evr_diff,
        "coef_max_diff": coef_diff,
        "train_acc": accuracy(y_tr, model.predict(X_tr)),
        "test_acc": accuracy(y_te, pred_ours),
        "test_balanced_acc": balanced_accuracy(y_te, pred_ours),
        "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(),
        "target_names": [str(t) for t in target_names],
    }
    _write_json("exp2_iris.json", summary)
    _write_csv("exp2_predictions.csv",
               ["sample_idx", "y_true", "pred_ours", "pred_sklearn"],
               [[int(te[i]), int(y_te[i]), int(pred_ours[i]), int(pred_sk[i])]
                for i in range(len(te))])

    # 图 (a)：前两个判别方向的得分散点（训练=圆点，测试=叉号）
    scores = model.transform(X)
    scores_te = model.transform(X_te)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["C0", "C3", "C2"]
    for k in range(3):
        m_tr = y_tr == k
        axes[0].scatter(scores[tr][m_tr, 0], scores[tr][m_tr, 1], s=22,
                        alpha=0.75, color=colors[k],
                        label="%s（训练）" % target_names[k])
        m_te = y_te == k
        axes[0].scatter(scores_te[m_te, 0], scores_te[m_te, 1], s=28,
                        marker="x", color=colors[k],
                        label="%s（测试）" % target_names[k])
    axes[0].set_xlabel("判别方向 1")
    axes[0].set_ylabel("判别方向 2")
    axes[0].set_title("iris 判别空间（训练 %.3f / 测试 %.3f）"
                      % (summary["train_acc"], summary["test_acc"]))
    axes[0].legend(loc="upper right", fontsize=8)

    # 图 (b)：测试集混淆矩阵热力图
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels([t[:8] for t in target_names], fontsize=9)
    axes[1].set_yticklabels([t[:8] for t in target_names], fontsize=9)
    axes[1].set_xlabel("预测类别")
    axes[1].set_ylabel("真实类别")
    axes[1].set_title("测试集混淆矩阵（与 sklearn 一致率 %.3f）" % agreement)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=11)
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle("实验二：iris 数据集拟合（自实现 vs sklearn）", y=1.03)
    path = save_fig(fig, "fig2_iris_discriminant", FIG_DIR)
    return summary, [path]


def _direction_error(w, w_star):
    """方向误差：对齐符号并归一化后的相对差（对尺度不变）。

    判别方向作为特征向量只确定到符号与尺度（白化空间单位范数），
    与解析解的欧氏尺度不同，直接相减会混入尺度伪影，故先归一化。
    """
    w = np.asarray(w, dtype=float)
    if float(w @ w_star) < 0:
        w = -w
    w = w / np.linalg.norm(w)
    ws = w_star / np.linalg.norm(w_star)
    return float(np.linalg.norm(w - ws))


def experiment3():
    """实验三：S_w 的数值稳定性（数值性质）。

    两个场景，对应两种失效模式：
      (a) 病态（满秩但条件数极大）：矩阵级实验——直接构造 S_w 与均值差
          向量，无抽样噪声，总体方向有解析解，误差纯来自数值计算
          （与 OLS 模块实验三同法）；对比"不截断"与"默认截断"。
      (b) 秩亏（p>n）：数据级实验——S_w 必然奇异，对比三种策略，
          并对照理论极限（真方向在 S_w 行空间之外的分量不可恢复）。
    """
    # ---- 场景 (a)：矩阵级病态扫描 ----
    p, n = 10, 200
    rng = np.random.default_rng(5)
    V, _ = np.linalg.qr(rng.standard_normal((p, p)))
    # d 仅取前 3 个特征方向的分量：保证真方向在扫描范围内始终可恢复
    u = np.zeros(p)
    u[:3] = 1.0 / np.sqrt(3.0)
    d = V @ u

    kappas = np.logspace(0, 8, 9)
    rows = []
    for kappa in kappas:
        s = np.logspace(0, np.log10(1.0 / kappa), p)
        Sw = 2.0 * n * (V * s ** 2) @ V.T
        w_star = V @ (u / (2.0 * n * s ** 2))     # 解析解：S_w^{-1} d
        Sb = np.outer(d, d)                       # 两分类时 S_b 秩为 1
        row = {"kappa": float(kappa), "cond_Sw": float(np.linalg.cond(Sw))}
        for name, rcond in [("no_trunc", 0.0), ("truncate", None)]:
            W, _ = solve_directions(Sw, Sb, rcond=rcond)
            row["err_" + name] = _direction_error(W[:, 0], w_star)
        rows.append(row)

    _write_csv("exp3_conditioning.csv",
               ["kappa", "cond_Sw", "err_no_trunc", "err_truncate"],
               [[r["kappa"], r["cond_Sw"], r["err_no_trunc"], r["err_truncate"]]
                for r in rows])

    # ---- 场景 (b)：p > n 秩亏 ----
    p2, n2 = 100, 30
    rng2 = np.random.default_rng(7)
    base2 = rng2.standard_normal(p2)
    base2 = base2 / np.linalg.norm(base2)
    mus2 = np.stack([base2, -base2])
    X2, y2, _, _ = make_gaussian_classes(n2, mus2, np.eye(p2), seed=0)
    w_star2 = mus2[0] - mus2[1]

    Sw2 = np.zeros((p2, p2))
    for k in range(2):
        Xc = X2[y2 == k] - X2[y2 == k].mean(axis=0)
        Sw2 += Xc.T @ Xc
    d2, V2 = np.linalg.eigh(Sw2)
    rank_sw = int(np.sum(d2 > p2 * np.finfo(float).eps * d2.max()))
    # 理论极限：真方向在 S_w 行空间（最大 rank 个特征向量张成）之外的分量
    row_basis = V2[:, p2 - rank_sw:]
    proj = row_basis @ (row_basis.T @ w_star2)
    limit_err = float(np.linalg.norm(w_star2 - proj) / np.linalg.norm(w_star2))

    # 数值可靠性：对特征加 1e-10 量级微小扰动后重新拟合。
    # 不截断解由舍入噪声驱动（白化放大 ~1e8），方向应剧烈变化；
    # 截断/收缩解应保持稳定
    rng3 = np.random.default_rng(99)
    X2p = X2 + 1e-10 * rng3.standard_normal(X2.shape)

    highdim = {"setting": {"p": p2, "n_per_class": n2, "seed": 0,
                           "perturbation": "1e-10"},
               "rank_Sw": rank_sw, "full_rank": p2,
               "rank_limit_err": limit_err,
               "errors": {}, "perturb_angle_deg": {}}
    for name, kw in [("no_trunc", {"rcond": 0.0}),
                     ("truncate", {"rcond": None}),
                     ("shrink", {"reg": 0.3})]:
        m1 = FisherDiscriminant(**kw).fit(X2, y2)
        m2 = FisherDiscriminant(**kw).fit(X2p, y2)
        w1, w2 = m1.scalings_[:, 0], m2.scalings_[:, 0]
        err = _direction_error(w1, w_star2)
        cos = float(w1 @ w2 / (np.linalg.norm(w1) * np.linalg.norm(w2)))
        angle = float(np.degrees(np.arccos(np.clip(abs(cos), -1.0, 1.0))))
        highdim["errors"][name] = err if np.isfinite(err) else None
        highdim["perturb_angle_deg"][name] = (angle if np.isfinite(angle)
                                              else None)
    _write_json("exp3_highdim.json", highdim)

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    kappas_have = [r["kappa"] for r in rows]
    for name, color, label in [("no_trunc", "C3", "不截断（rcond=0）"),
                               ("truncate", "C0", "默认截断")]:
        axes[0].loglog(kappas_have, [r["err_" + name] for r in rows],
                       marker="o", ms=4, color=color, label=label)
    axes[0].set_xlabel("协方差条件数 κ")
    axes[0].set_ylabel("判别方向相对误差")
    axes[0].set_title("(a) 病态扫描：矩阵级无噪实验（p=10）")
    axes[0].legend(loc="upper left", fontsize=9)

    # 图 (b)：秩亏场景的数值可靠性——扰动后方向变化（度）
    names = ["no_trunc", "truncate", "shrink"]
    labels_cn = ["不截断", "默认截断", "收缩"]
    angs = [highdim["perturb_angle_deg"][nm] for nm in names]
    bars = axes[1].bar(labels_cn, angs, color=["C3", "C0", "C2"], alpha=0.85)
    axes[1].set_ylabel("微小扰动后的方向变化（度）")
    axes[1].set_title("(b) 秩亏场景：p=100 > n=60（rank=58）")
    finite = [a for a in angs if a is not None]
    axes[1].set_ylim(0, max(finite) * 1.25 if finite else 100)
    for bar, v in zip(bars, angs):
        text = "NaN" if v is None else "%.2e°" % v
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     text, ha="center", va="bottom", fontsize=9)

    fig.suptitle("实验三：S_w 病态与秩亏下的数值稳定性", y=1.03)
    path = save_fig(fig, "fig3_sw_conditioning", FIG_DIR)
    return {"conditioning": rows, "highdim": highdim}, [path]


def main():
    # 以下三行是控制台快速摘要；正式结果以 results/ 下的 JSON（汇总）与
    # CSV（逐行明细）为准，供文档引用与再分析
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    s1, _ = experiment1()
    c = s1["cases"]
    print("[实验一] 可分：夹角%.2f° 测试错误率%.3f(Bayes %.3f) | 难分：夹角%.2f° 测试错误率%.3f(Bayes %.3f)"
          % (c[0]["angle_deg"], c[0]["test_err"], c[0]["bayes_err"],
             c[1]["angle_deg"], c[1]["test_err"], c[1]["bayes_err"]))

    s2, _ = experiment2()
    print("[实验二] iris：与sklearn一致率%.3f，训练准确率=%.4f 测试准确率=%.4f 平衡准确率=%.4f，EVR最大差=%.2e"
          % (s2["agreement_with_sklearn"], s2["train_acc"], s2["test_acc"],
             s2["test_balanced_acc"], s2["evr_max_diff"]))

    s3, _ = experiment3()
    last = s3["conditioning"][-1]
    hd = s3["highdim"]
    pa = hd["perturb_angle_deg"]
    print("[实验三] 病态κ=%.0e：不截断=%.2e 截断=%.2e | 秩亏p>n 扰动角：不截断=%s 截断=%.2e° 收缩=%.2e°"
          % (last["kappa"], last["err_no_trunc"], last["err_truncate"],
             ("NaN" if pa["no_trunc"] is None else "%.2e°" % pa["no_trunc"]),
             pa["truncate"], pa["shrink"]))

    print("实验完成：3 张图已存至 figures/，数值结果已存至 results/")


if __name__ == "__main__":
    main()
