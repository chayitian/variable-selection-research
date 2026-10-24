"""最小二乘可运行实验。

三组实验：
  实验一  合成稀疏数据上的系数恢复（支撑集指标 + 系数误差）
  实验二  diabetes 真实数据拟合，并与 sklearn LinearRegression 对拍
  实验三  数值稳定性：条件数扫描下三种解法的相对系数误差

运行方式（在仓库根目录执行）：
    python methods/01-least-squares/experiment.py
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from lib import load_diabetes, make_synthetic, mse, set_style, support_recovery, save_fig  # noqa: E402
from ols import OLS, fit_normal_equation, fit_qr, fit_svd  # noqa: E402

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
    """实验一：合成稀疏数据上的系数恢复。

    两部分：
    A. 单次实现（seed=0）：展示 OLS 的支撑集行为——满支撑、不稀疏；
    B. 蒙特卡洛（B=200，seeds 0..199）：重复生成数据并拟合，以平均系数
       偏差验证无偏性。无偏性是重复实验的均值性质，单次模拟无法展示，
       必须依赖 B 次重复的均值行为（见 README §5.1）。
    """
    setting = {"n_samples": 100, "n_features": 20, "n_informative": 5,
               "rho": 0.3, "noise_std": 1.0}
    # 参数理由：p=20 而真非零仅 5 个，制造"信号可恢复但解不稀疏"的典型场景；
    # rho=0.3 为温和特征相关（后续 Elastic Net / Group Lasso 模块会加大难度）；
    # noise_std=1.0 使信噪比适中；所有种子固定，结果完全可复现

    # ---- A. 单次实现（seed=0）：支撑集行为 ----
    X, y, beta_true = make_synthetic(seed=0, **setting)
    # 本实验关注统计性质，数值稳定性由实验三专测，故用默认 QR
    beta_hat = fit_qr(X, y)
    support = support_recovery(beta_true, beta_hat)

    # ---- B. 蒙特卡洛（B=200）：无偏性验证 ----
    # 注意 make_synthetic 的 beta_true 由 n_informative 唯一确定、与种子无关，
    # 各次重复共享同一真值，平均偏差直接估计 E[beta_hat] - beta
    B = 200
    beta_hats = np.zeros((B, setting["n_features"]))
    for b in range(B):
        Xb, yb, _ = make_synthetic(seed=b, **setting)
        beta_hats[b] = fit_qr(Xb, yb)
    mc_mean = beta_hats.mean(axis=0)
    mc_sd = beta_hats.std(axis=0, ddof=1)
    bias = mc_mean - beta_true

    summary = {
        "setting": dict(setting, seed=0, solver="qr"),
        "single_seed0": {
            "support": support,
            "coef_mse": mse(beta_true, beta_hat),
            "pred_mse": mse(y, X @ beta_hat),
        },
        "monte_carlo": {
            "B": B,
            "seeds": "0..199",
            "mean_abs_bias": float(np.mean(np.abs(bias))),
            "max_abs_bias": float(np.max(np.abs(bias))),
            "mean_coef_sd": float(np.mean(mc_sd)),
            "mc_se_scale": float(np.mean(mc_sd) / np.sqrt(B)),
        },
    }
    _write_json("exp1_coefficient_recovery.json", summary)
    _write_csv("exp1_coefficients.csv",
               ["j", "beta_true", "beta_hat"],
               [[j, float(beta_true[j]), float(beta_hat[j])]
                for j in range(len(beta_true))])
    _write_csv("exp1_mc_bias.csv",
               ["j", "beta_true", "mc_mean", "mc_sd", "bias"],
               [[j, float(beta_true[j]), float(mc_mean[j]), float(mc_sd[j]),
                 float(bias[j])] for j in range(len(beta_true))])

    # 图 (a)：单次实现的系数对比。stem 图是稀疏系数对比的标准画法：
    # 真实非零位置与估计非零位置同屏一目了然
    fig, ax = plt.subplots(figsize=(7.5, 4))
    idx = np.arange(len(beta_true))
    ax.stem(idx, beta_true, linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label="真实系数")
    ax.stem(idx, beta_hat, linefmt="C3-", markerfmt="C3s", basefmt=" ",
            label="估计系数（QR，seed=0）")
    ax.set_xlabel("特征编号 j")
    ax.set_ylabel("系数值")
    ax.set_title("实验一(a)：单次实现的系数恢复（n=100, p=20, 非零系数=5）")
    ax.legend(loc="upper right")
    path_a = save_fig(fig, "fig1_coefficient_recovery", FIG_DIR)

    # 图 (b)：蒙特卡洛均值 vs 真值，误差棒为 2 倍标准误——若均值与真值
    # 的偏差在标准误量级内，即实证支持无偏性
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.stem(idx, beta_true, linefmt="C0-", markerfmt="C0o", basefmt=" ",
            label="真实系数")
    ax.errorbar(idx + 0.15, mc_mean, yerr=2 * mc_sd / np.sqrt(B), fmt="C3s",
                ms=4, capsize=2, label="MC 均值 ± 2×标准误（B=200）")
    ax.set_xlabel("特征编号 j")
    ax.set_ylabel("系数值")
    ax.set_title("实验一(b)：蒙特卡洛平均系数 vs 真值")
    ax.legend(loc="upper right")
    path_b = save_fig(fig, "fig1b_mc_bias", FIG_DIR)
    return summary, [path_a, path_b]


def experiment2():
    """实验二：diabetes 数据拟合与 sklearn 对拍。"""
    X, y, names = load_diabetes()
    # 350/92 约 8:2 划分；seed=42 固定，保证与 README §5.2 的数字一致
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    n_train = 350
    tr, te = order[:n_train], order[n_train:]
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    model = OLS(solver="qr", fit_intercept=True).fit(X_tr, y_tr)

    # 对拍：以 sklearn 为独立 oracle——两者在机器精度内一致即证明实现正确。
    # sklearn 内部走 scipy.linalg.lstsq（基于 SVD），与本模块的 QR 是独立代码路径
    from sklearn.linear_model import LinearRegression
    sk = LinearRegression().fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    summary = {
        "setting": {"dataset": "sklearn diabetes", "n_train": int(n_train),
                    "n_test": int(len(te)), "seed": 42, "solver": "qr"},
        "train_r2": model.score(X_tr, y_tr),
        "test_r2": model.score(X_te, y_te),
        "test_mse": mse(y_te, y_pred),
        "sklearn_max_coef_diff": float(np.max(np.abs(model.coef_ - sk.coef_))),
        "sklearn_intercept_diff": float(abs(model.intercept_ - sk.intercept_)),
    }
    _write_json("exp2_diabetes_fit.json", summary)
    _write_csv("exp2_coefficients.csv",
               ["feature", "coef_ours", "coef_sklearn"],
               [[names[j], float(model.coef_[j]), float(sk.coef_[j])]
                for j in range(len(names))])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_te, y_pred, s=18, alpha=0.75, color="C0")
    lo = float(min(y_te.min(), y_pred.min()))
    hi = float(max(y_te.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1, label="理想线 y = x")
    axes[0].set_xlabel("真实值")
    axes[0].set_ylabel("预测值")
    axes[0].set_title("测试集预测 vs 真实（R2 = %.3f）" % summary["test_r2"])
    axes[0].legend(loc="upper left")

    # 残差 = 真实 - 预测；理想情形应围绕 0 随机分布、无喇叭口等系统性模式
    resid = y_te - y_pred
    axes[1].scatter(y_pred, resid, s=18, alpha=0.75, color="C1")
    axes[1].axhline(0.0, color="k", linewidth=1, linestyle="--")
    axes[1].set_xlabel("预测值")
    axes[1].set_ylabel("残差")
    axes[1].set_title("残差图（MSE = %.1f）" % summary["test_mse"])

    fig.suptitle("实验二：diabetes 数据集拟合（自实现 vs sklearn）", y=1.03)
    path = save_fig(fig, "fig2_diabetes_fit", FIG_DIR)
    return summary, path


def experiment3():
    """实验三：条件数扫描下的数值稳定性（无噪声，误差纯来自数值计算）。"""
    n, p = 200, 10
    rng = np.random.default_rng(1)
    # 用两个正交矩阵夹出设计矩阵：X = U diag(s) V^T，其奇异值恰为 s，
    # 从而把条件数 kappa = s_max / s_min 变成精确受控变量
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    V, _ = np.linalg.qr(rng.standard_normal((p, p)))
    beta_true = rng.standard_normal(p)

    kappas = np.logspace(0, 8, 13)
    solvers = [("normal", fit_normal_equation), ("qr", fit_qr), ("svd", fit_svd)]
    rows = []
    for kappa in kappas:
        s = np.linspace(1.0, 1.0 / kappa, p)
        X = U @ np.diag(s) @ V.T
        # 无噪声设计：y 精确落在 X 的列空间中，系数误差 100% 来自数值计算，
        # 从而把"分解方式的影响"从统计误差中干净隔离出来
        y = X @ beta_true
        # 记录实测 cond：构造值 1/kappa 与实测值存在微小浮点差异，作图应使用实测值
        row = {"cond": float(np.linalg.cond(X))}
        for name, fn in solvers:
            beta_hat = fn(X, y)
            row[name] = float(np.linalg.norm(beta_hat - beta_true)
                              / np.linalg.norm(beta_true))
        rows.append(row)

    _write_csv("exp3_numerical_stability.csv",
               ["cond", "err_normal", "err_qr", "err_svd"],
               [[r["cond"], r["normal"], r["qr"], r["svd"]] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    conds = [r["cond"] for r in rows]
    # 条件数跨 8 个数量级，线性坐标下前 7 个点会挤在原点，必须用对数坐标
    for name, color in [("normal", "C3"), ("qr", "C0"), ("svd", "C2")]:
        ax.loglog(conds, [r[name] for r in rows], marker="o", ms=4,
                  color=color, label=name + " 解法")
    ax.set_xlabel("设计矩阵条件数 cond(X)")
    ax.set_ylabel("相对系数误差")
    ax.set_title("实验三：条件数扫描下三种解法的数值稳定性（无噪声）")
    ax.legend(loc="upper left")
    path = save_fig(fig, "fig3_numerical_stability", FIG_DIR)
    return rows, path


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # 以下三行是控制台快速摘要；正式结果以 results/ 下的 JSON（汇总）与
    # CSV（逐行明细）为准，供文档引用与再分析
    s1, _ = experiment1()
    print("[实验一] 单次(seed=0)：tp=%d fp=%d fn=%d Jaccard=%.3f 系数MSE=%.4f 预测MSE=%.4f | MC(B=200)：平均|偏差|=%.4f 最大|偏差|=%.4f 系数平均SD=%.4f"
          % (s1["single_seed0"]["support"]["tp"],
             s1["single_seed0"]["support"]["fp"],
             s1["single_seed0"]["support"]["fn"],
             s1["single_seed0"]["support"]["jaccard"],
             s1["single_seed0"]["coef_mse"],
             s1["single_seed0"]["pred_mse"],
             s1["monte_carlo"]["mean_abs_bias"],
             s1["monte_carlo"]["max_abs_bias"],
             s1["monte_carlo"]["mean_coef_sd"]))

    s2, _ = experiment2()
    print("[实验二] 训练R2=%.4f，测试R2=%.4f，测试MSE=%.2f，与sklearn最大系数差=%.2e"
          % (s2["train_r2"], s2["test_r2"], s2["test_mse"],
             s2["sklearn_max_coef_diff"]))

    rows, _ = experiment3()
    last = rows[-1]
    print("[实验三] cond=%.0e 时相对系数误差：normal=%.3e，qr=%.3e，svd=%.3e"
          % (last["cond"], last["normal"], last["qr"], last["svd"]))

    print("实验完成：4 张图已存至 figures/，数值结果已存至 results/")


if __name__ == "__main__":
    main()
