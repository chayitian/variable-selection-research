"""Mallows Cp 与模型定模准则（Mallows Cp & Model Selection Criteria）从零实现。

对应文档：methods/05-mallows-cp/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。

定位：本模块回答"选哪个模型规模"（04-best-subset-selection 回答"每个规模下
哪个子集最优"）。子集枚举只含教学规模穷举；完整分支定界见 04 模块。
"""

import numpy as np

_EPS = 1e-12


# ═════════════════════════════ 第 1 段：问题设定与公共内核 ═════════════════════════════
# 【做什么】对候选变量的子集 S 拟合带截距的 OLS（截距经中心化实现），返回
#          系数、残差平方和 RSS 与数值秩；并提供教学规模的穷举最优子集表。
# 【为什么】所有准则都建立在一个量上：子集 RSS。定模准则的差异只在于
#          "如何把 RSS 与模型复杂度权衡"（§2.2–2.4 的五种答案）。
# 【对应文档】README §2.1、公式 (1)


def _center(X, y):
    """按列中心化 X、整体中心化 y（截距由此实现）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    return X - X.mean(axis=0), y - y.mean()


def fit_subset_rss(Xc, yc, cols):
    """中心化数据上子集 OLS 的系数、RSS 与数值秩（README 公式 (1)）。

    Parameters
    ----------
    Xc, yc : ndarray，已中心化的设计矩阵与响应
    cols : sequence of int，子集列索引（可为空，即仅截距模型）

    Returns
    -------
    beta : ndarray, shape (len(cols),)
    rss : float
    rank : int，子集设计矩阵的数值秩
    """
    cols = list(cols)
    if not cols:
        return np.zeros(0), float(yc @ yc), 0
    A = Xc[:, cols]
    # 秩亏子集（共线变量）用最小范数解：RSS 仍可计算，方向不唯一
    beta, _, rank, _ = np.linalg.lstsq(A, yc, rcond=None)
    resid = yc - A @ beta
    return beta, float(resid @ resid), int(rank)


def sigma2_hat_full(Xc, yc):
    """全模型残差方差估计（README 公式 (2)）：RSS(全模型)/(n-p-1)。"""
    n, p = Xc.shape
    if n - p - 1 < 1:
        raise ValueError("全模型不可估计：n - p - 1 >= 1 不成立（n=%d, p=%d）" % (n, p))
    _, rss, _ = fit_subset_rss(Xc, yc, list(range(p)))
    return rss / (n - p - 1)


def exhaustive_best_subsets(X, y, kmax=None):
    """教学规模穷举：遍历全部子集，返回每层最优子集表。

    完整的分支定界搜索见 04-best-subset-selection 模块；本模块的定模准则
    既可用穷举表，也可用任何搜索算法给出的子集表（接口一致）。

    Parameters
    ----------
    X, y : ndarray
    kmax : int or None，最大规模；None 时取 min(p, n-2)

    Returns
    -------
    best : dict，k -> (cols, rss)
    n_evals : int，RSS 评估次数
    """
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    if kmax is None:
        kmax = min(p, n - 2)
    if not 1 <= kmax <= min(p, n - 2):
        raise ValueError("kmax 须满足 1 <= kmax <= min(p, n-2)（当前 kmax=%r）" % kmax)
    best = {0: ([], float(yc @ yc))}
    n_evals = 0
    for mask in range(1, 1 << p):
        cols = [j for j in range(p) if (mask >> j) & 1]
        k = len(cols)
        if k > kmax:
            continue
        _, rss, _ = fit_subset_rss(Xc, yc, cols)
        n_evals += 1
        if k not in best or rss < best[k][1] - _EPS:
            best[k] = (cols, rss)
    return best, n_evals


# ═════════════════════════════ 第 2 段：Mallows Cp ═════════════════════════════
# 【做什么】Mallows Cp 的标准形式：Cp = RSS/σ̂² - n + 2q（q = 预测变量数 + 1，
#          含截距）；对预先指定的模型 S，E[Cp(S)] = q + 偏差项。
# 【为什么】Cp 把"选规模"转化为预测误差估计：RSS/σ̂² 估计训练点上的总
#          失配（噪声 + 估计方差），减去 n 再补回 2q 的估计方差代价，
#          恰好是无偏的预测误差代理。这是 Mallows (1973) 的核心洞察。
# 【对应文档】README §2.2、公式 (3)(4)


def mallows_cp(rss, q, n, sigma2_hat):
    """Mallows Cp（README 公式 (3)）。

    Parameters
    ----------
    rss : float，子集残差平方和
    q : int，模型参数个数（= 预测变量数 + 1，含截距）
    n : int，样本数
    sigma2_hat : float，误差方差估计

    Returns
    -------
    float
    """
    if sigma2_hat <= 0:
        raise ValueError("sigma2_hat 须为正数（当前 %r）" % sigma2_hat)
    return rss / sigma2_hat - (n - 2 * q)


def cp_curve(best, n, sigma2_hat):
    """对每层最优子集表计算 Cp 曲线（dict k -> Cp）。"""
    return {k: mallows_cp(best[k][1], k + 1, n, sigma2_hat)
            for k in sorted(best)}


# ═════════════════════════════ 第 3 段：交叉验证族（PRESS 与 GCV） ═════════════════════════════
# 【做什么】留一交叉验证的两种形式：PRESS（hat 矩阵恒等式）与 GCV（其加权版本）。
# 【为什么】PRESS/GCV 与 Cp 估计同一个目标（训练点上的预测误差），但走
#          "计算"路线而非"解析修正"路线；GCV 用有效参数个数 tr(H) 替代逐个
#          杠杆值，对高杠杆点更稳健。三者的对照是实验三的第一组。
# 【对应文档】README §2.3、公式 (5)


def hat_diagonal(Xc, cols):
    """子集模型（含截距）的 hat 矩阵对角线（杠杆值 h_ii）。

    h = diag(A(A'A)^{-1}A')，A = [1, X_cols]；用 QR 分解数值稳定地计算。
    """
    cols = list(cols)
    A = np.column_stack([np.ones(len(Xc)), Xc[:, cols]]) if cols \
        else np.ones((len(Xc), 1))
    Q, _ = np.linalg.qr(A)
    return np.sum(Q ** 2, axis=1)


def _fitted(Xc, yc, cols):
    """中心化空间中的拟合值（内部辅助）。"""
    cols = list(cols)
    if not cols:
        return np.zeros(len(yc))
    beta, _, _ = fit_subset_rss(Xc, yc, cols)
    return Xc[:, cols] @ beta


def press_rss(X, y, cols):
    """PRESS：留一交叉验证残差平方和 Σ(e_i/(1-h_ii))^2（README 公式 (5)）。"""
    Xc, yc = _center(X, y)
    if not list(cols):
        return float(yc @ yc)  # 仅截距模型：每个留一拟合都等于总均值
    h = hat_diagonal(Xc, cols)
    if np.any(h >= 1.0):
        raise ValueError("存在杠杆值 >= 1 的点（共线或重复样本），PRESS 不可计算")
    e = yc - _fitted(Xc, yc, cols)
    return float(np.sum((e / (1.0 - h)) ** 2))


def gcv_rss(X, y, cols):
    """GCV：RSS/n / (1 - q/n)^2 的 n 倍形式（README 公式 (6)）。

    返回与 PRESS 同量纲的值 n*RSS/(n-q)^2，便于与 PRESS 直接对照。
    """
    Xc, yc = _center(X, y)
    q = len(list(cols)) + 1
    n = len(yc)
    _, rss, _ = fit_subset_rss(Xc, yc, cols)
    return float(n * rss / (n - q) ** 2)


# ═════════════════════════════ 第 4 段：信息准则族（AIC 与 BIC） ═════════════════════════════
# 【做什么】同一 RSS 表上的另外两种权衡：AIC（惩罚 2q）与 BIC（惩罚 q·ln n）。
# 【为什么】Cp 的惩罚 2q 与 AIC 相同、比 BIC 的 q·ln n 温和；BIC 在大样本
#          下一致性选择真实规模（若存在），Cp/AIC 则倾向于保留略大的模型
#          以降低预测误差。实验三第二组直接对照五者的定模行为。
# 【对应文档】README §2.4、公式 (7)


def aic_value(rss, q, n):
    """高斯线性模型的 AIC（差常数不影响定模）：n·ln(RSS/n) + 2q。"""
    return n * np.log(rss / n) + 2 * q


def bic_value(rss, q, n):
    """高斯线性模型的 BIC（差常数不影响定模）：n·ln(RSS/n) + q·ln n。"""
    return n * np.log(rss / n) + q * np.log(n)


# ═════════════════════════════ 第 5 段：统一估计器接口 ═════════════════════════════
# 【做什么】封装 ModelCriteriaSelector 类：给定子集表与准则，输出选定规模、
#          选定模型与整条准则曲线；接口与 04 模块 BestSubsetSelector 对齐。
# 【为什么】定模是"可审计"的——整条 Cp/AIC/BIC 曲线与最小点位置应该能直接
#          读出，而不是只给一个 k*；σ̂² 的估计方式也必须是显式参数（实验三
#          将证明它显著影响定模）。
class ModelCriteriaSelector:
    """模型定模准则估计器（教学版）。

    Parameters
    ----------
    criterion : {"cp", "aic", "bic", "press", "gcv"}, default="cp"
        定模准则。press/gcv 直接由数据计算；cp 需要 σ̂²；aic/bic 不需要。
    sigma2 : {"full", "selected", float}, default="full"
        Cp 的误差方差估计方式：全模型残差方差 / 选定模型残差方差（循环，
        有偏）/ 用户给定值（实验中可传真实噪声方差作 oracle）。
    kmax : int or None，搜索的最大规模；None 时取 min(p, n-2)。

    Attributes
    ----------
    subsets_ : dict，k -> list of int，每层最优子集
    rss_by_k_ : dict，k -> float
    criterion_curve_ : dict，k -> float，选定准则的曲线
    k_ : int，选定规模
    selected_ : list of int，选定子集（升序）
    coef_ : ndarray, shape (p,)，全长度系数
    intercept_ : float
    rss_ : float，最终模型 RSS
    r2_ : float
    sigma2_hat_ : float或None，实际使用的 σ̂²
    n_evals_ : int，子集枚举的 RSS 评估次数
    """

    _CRITERIA = ("cp", "aic", "bic", "press", "gcv")

    def __init__(self, criterion="cp", sigma2="full", kmax=None):
        if criterion not in self._CRITERIA:
            raise ValueError("未知准则 %r，可选：%s"
                             % (criterion, list(self._CRITERIA)))
        if not (isinstance(sigma2, str) and sigma2 in ("full", "selected")) \
                and not (isinstance(sigma2, (int, float)) and sigma2 > 0):
            raise ValueError("sigma2 须为 'full'、'selected' 或正数")
        self.criterion = criterion
        self.sigma2 = sigma2
        self.kmax = kmax

    def fit(self, X, y):
        """拟合并返回 self。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 须为二维数组 (n_samples, n_features)")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y 须为长度与 X 行数一致的一维数组")
        n, p = X.shape
        Xc, yc = _center(X, y)

        best, n_evals = exhaustive_best_subsets(X, y, kmax=self.kmax)
        self.subsets_ = {k: best[k][0] for k in sorted(best)}
        self.rss_by_k_ = {k: best[k][1] for k in sorted(best)}
        self.n_evals_ = n_evals

        # σ̂²：全模型（标准）/ 用户给定；'selected' 需先有一轮定模，故延后
        if isinstance(self.sigma2, (int, float)):
            self.sigma2_hat_ = float(self.sigma2)
        elif self.sigma2 == "full":
            self.sigma2_hat_ = sigma2_hat_full(Xc, yc)
        else:
            self.sigma2_hat_ = None  # selected：延后到曲线计算后

        if self.criterion == "press":
            curve = {k: press_rss(X, y, best[k][0]) for k in sorted(best)}
        elif self.criterion == "gcv":
            curve = {k: gcv_rss(X, y, best[k][0]) for k in sorted(best)}
        else:
            if self.sigma2_hat_ is None:  # selected：用全模型 σ̂² 先定一模
                s2 = sigma2_hat_full(Xc, yc)
                curve0 = cp_curve(best, n, s2)
                k0 = min(curve0, key=lambda k: (curve0[k], k))
                _, rss0, _ = fit_subset_rss(Xc, yc, best[k0][0])
                self.sigma2_hat_ = rss0 / (n - k0 - 1)
            if self.criterion == "cp":
                curve = cp_curve(best, n, self.sigma2_hat_)
            elif self.criterion == "aic":
                curve = {k: aic_value(best[k][1], k + 1, n) for k in sorted(best)}
            else:
                curve = {k: bic_value(best[k][1], k + 1, n) for k in sorted(best)}
        self.criterion_curve_ = curve

        k_star = min(curve, key=lambda k: (curve[k], k))
        self.k_ = k_star
        self.selected_ = list(self.subsets_[k_star])

        beta, rss, _ = fit_subset_rss(Xc, yc, self.selected_)
        self.coef_ = np.zeros(p)
        if self.selected_:
            self.coef_[self.selected_] = beta
        self.intercept_ = float(y.mean() - X.mean(axis=0) @ self.coef_)
        self.rss_ = rss
        ss_tot = float(yc @ yc)
        self.r2_ = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
        return self

    def predict(self, X):
        """预测：X @ coef_ + intercept_（未选中变量的系数为 0）。"""
        if not hasattr(self, "coef_"):
            raise RuntimeError("模型尚未拟合，请先调用 fit")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X, y):
        """决定系数 R^2（定义与 01/04 模块一致）。"""
        y = np.asarray(y, dtype=float)
        ss_res = float(np.sum((y - self.predict(X)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot
