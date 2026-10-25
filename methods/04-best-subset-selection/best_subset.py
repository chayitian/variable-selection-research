"""全子集选择与分支定界（Best Subset Selection & Branch and Bound）从零实现。

对应文档：methods/04-best-subset-selection/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。

算法思想依据：Furnival & Wilson (1974), *Technometrics* 16(4): 499–511, §3
——RSS 单调性给出的剪枝原理。本实现是借鉴该思想的教学变体：加法树 +
最大完成集界，并非原文逆树的逐行复现；两者关系与选型理由见 README §2.3。
"""

import warnings

import numpy as np

_EPS = 1e-12


# ═════════════════════════════ 第 1 段：问题设定与公共内核 ═════════════════════════════
# 【做什么】对候选变量的子集 S 拟合带截距的 OLS（截距经中心化实现：对 X、y
#          去均值后拟合无截距模型，再以 y_bar - x_bar^T beta 还原），返回
#          系数、残差平方和 RSS 与数值秩。
# 【为什么】全子集选择的目标函数就是子集 RSS；三种变体（穷举、分支定界、
#          后续信息准则）全部建立在这一个内核上，与 01/03 模块同构。
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


def _importance_order(Xc, yc):
    """按与响应的绝对边际相关降序给出变量重要性排序（README §3 步骤 2）。

    Furnival & Wilson (1974) 原文建议用全模型 |t| 排序；本实现改用边际
    |corr|，使 p >= n（全模型不可估计）时仍可排序。该替代只改变搜索顺序
    与剪枝效率，不改变最优结果；实际剪枝效果见 README §5.2 实测。

    Returns
    -------
    order : list of int，重要性降序的变量索引
    cors : ndarray，各变量的绝对边际相关
    """
    ynorm = float(np.linalg.norm(yc))
    cors = np.zeros(Xc.shape[1])
    for j in range(Xc.shape[1]):
        xj = Xc[:, j]
        denom = float(np.linalg.norm(xj)) * ynorm
        cors[j] = abs(float(xj @ yc)) / denom if denom > 0 else 0.0
    order = sorted(range(Xc.shape[1]), key=lambda j: (-cors[j], j))
    return order, cors


# ═════════════════════════════ 第 2 段：穷举全子集 ═════════════════════════════
# 【做什么】遍历全部 2^p 个子集（规模 1..kmax），记录每个规模的最优子集。
# 【为什么】穷举是分支定界的正确性基准：实验二用两者结果的一致性来验证
#          剪枝没有漏掉最优解。规模上限 kmax 之外的子集无需拟合。
# 【对应文档】README §2.2、公式 (2)


def exhaustive_best_subsets(X, y, kmax=None):
    """穷举全部子集，返回每层最优子集表。

    Parameters
    ----------
    X, y : ndarray
    kmax : int or None，最大规模；None 时取 min(p, n-2)

    Returns
    -------
    best : dict，k -> (cols, rss)
    n_evals : int，RSS 评估次数（= 被拟合的子集数）
    cors : ndarray，边际相关（供诊断）
    """
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    if kmax is None:
        kmax = min(p, n - 2)
    if not 1 <= kmax <= min(p, n - 2):
        raise ValueError("kmax 须满足 1 <= kmax <= min(p, n-2)（当前 kmax=%r）" % kmax)
    _, cors = _importance_order(Xc, yc)

    best = {0: ([], float(yc @ yc))}
    n_evals = 0
    # 位掩码遍历全部非空子集；popcount 超限的跳过（不拟合）
    for mask in range(1, 1 << p):
        cols = [j for j in range(p) if (mask >> j) & 1]
        k = len(cols)
        if k > kmax:
            continue
        _, rss, _ = fit_subset_rss(Xc, yc, cols)
        n_evals += 1
        if k not in best or rss < best[k][1] - _EPS:
            best[k] = (cols, rss)
    return best, n_evals, cors


# ═════════════════════════════ 第 3 段：分支定界（加法树 + 最大完成集界） ═════════════════════════════
# 【做什么】对每个目标规模 k = 1..kmax 做一次"纳入/排除"二叉搜索（加法树），
#           用 RSS 单调性得到的下界剪掉不可能包含更优 k 子集的分支。
# 【为什么】穷举要 2^p 次拟合。单调性引理（S ⊆ T ⟹ RSS(S) ≥ RSS(T)）给出
#           两种建树方向：逆树（FW 原文，从全集删变量）中节点 RSS 是其子孙
#           （更小子集）RSS 的下界；但逆树对小规模目标存在结构性缺陷——最优
#           k 子集的所有超集都满足 RSS ≤ best[k]，永不被剪枝。本实现改用
#           加法树配"最大完成集"界：节点 (i, S)（已处理前 i 个变量、纳入 S）
#           的任何一个完成集 T 都满足 T ⊆ U := S ∪ {未决定变量}，于是
#           RSS(T) ≥ RSS(U)——一旦 RSS(U) ≥ best[k]，整枝剪掉。变量按重要性
#           降序、优先纳入，使首个叶子（最重要的 k 个变量）立刻成为紧 incumbent。
# 【对应文档】README §2.3、公式 (3)(4)


def branch_and_bound(X, y, kmax=None):
    """分支定界：对每个 k = 1..kmax 搜索最优 k 子集，汇总每层最优子集表。

    Parameters
    ----------
    X, y : ndarray
    kmax : int or None，最大规模；None 时取 min(p, n-2)

    Returns
    -------
    best : dict，k -> (cols, rss)，与 exhaustive_best_subsets 结构一致
    n_evals : int，RSS 评估次数（跨 k 去重：每个子集只拟合一次）
    cors : ndarray，边际相关（供诊断）

    Notes
    -----
    正确性：节点 (i, S) 的完成集 T 都满足 S ⊆ T ⊆ U（U = S ∪ 未决定变量），
    由单调性 RSS(T) ≥ RSS(U)；若 RSS(U) ≥ best[k]，则该子树中不存在比
    incumbent 更优的 k 子集，剪枝不丢最优解。纳入/排除树中每个 k 子集对应
    唯一路径，未剪路径的叶子逐一评估，故最终 best[k] 为全局最优。
    """
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    if kmax is None:
        kmax = min(p, n - 2)
    if not 1 <= kmax <= min(p, n - 2):
        raise ValueError("kmax 须满足 1 <= kmax <= min(p, n-2)（当前 kmax=%r）" % kmax)
    order, cors = _importance_order(Xc, yc)
    seq = order  # 位置 i 对应的变量（重要性降序：先纳入最重要的）

    best = {0: ([], float(yc @ yc))}
    cache = {}  # frozenset -> RSS：跨 k 共享，每个子集只拟合一次
    n_evals = 0

    def rss_of(cols):
        """子集 RSS（带缓存）；返回缓存未命中计数之外的本次拟合计数。"""
        nonlocal n_evals
        key = frozenset(cols)
        if key not in cache:
            _, r, _ = fit_subset_rss(Xc, yc, list(key))
            cache[key] = r
            n_evals += 1
        return cache[key]

    for k in range(1, kmax + 1):
        best_cols, best_rss = None, np.inf

        def dfs(i, S):
            """深度优先遍历：i = 下一个待决定变量位置，S = 已纳入变量。"""
            nonlocal best_cols, best_rss
            if len(S) == k:
                rss = rss_of(S)       # 叶子：唯一完成集就是 S 本身
                if rss < best_rss - _EPS:
                    best_rss, best_cols = rss, list(S)
                return
            if i == p:
                return                # 变量已用尽仍凑不够 k 个
            U = S + seq[i:]           # 最大完成集
            if len(U) < k:
                return
            if rss_of(U) >= best_rss - _EPS:
                return                # 剪枝：所有完成集 T ⊆ U，RSS(T) ≥ RSS(U)
            dfs(i + 1, S + [seq[i]])   # 纳入 seq[i]（U 不变）
            dfs(i + 1, S)              # 排除 seq[i]（U 缩小，界收紧）

        dfs(0, [])
        best[k] = (sorted(best_cols), best_rss)
    return best, n_evals, cors


# ═════════════════════════════ 第 4 段：Mallows Cp 与模型定模 ═════════════════════════════
# 【做什么】对每层最优子集计算 Mallows Cp，取最小 Cp 的规模为选定模型。
# 【为什么】"每层最优子集表"只回答"每个规模下谁最好"；选哪个规模需要
#          拟合优度与复杂度的权衡准则。Cp = RSS/σ̂² - (n - 2k) 在真实模型
#          维度不超过 k 时期望约为 k（加常数），最小 Cp 的 k 即所选规模
#          （Mallows 1973；FW 1974 的算法正是配合 Cp 使用）。
# 【对应文档】README §2.4、公式 (5)


def mallows_cp(rss, k, n, sigma2_hat):
    """Mallows Cp（README 公式 (5)）。

    Parameters
    ----------
    rss : float，该子集的残差平方和
    k : int，子集规模
    n : int，样本数
    sigma2_hat : float，误差方差估计（取全模型残差方差）

    Returns
    -------
    float
    """
    if sigma2_hat <= 0:
        raise ValueError("sigma2_hat 须为正数")
    return rss / sigma2_hat - (n - 2 * k)


def cp_table(best, n, sigma2_hat, kmax):
    """对最优子集表计算各规模 Cp，返回 dict k -> Cp。"""
    return {k: mallows_cp(best[k][1], k, n, sigma2_hat)
            for k in range(0, kmax + 1) if k in best}


# ═════════════════════════════ 第 5 段：统一估计器接口 ═════════════════════════════
# 【做什么】封装 BestSubsetSelector 类，对外暴露与 03 模块 StepwiseSelector
#          一致的 fit/predict/score 接口，并输出可审计的选择过程。
# 【为什么】接口一致便于跨模块对照实验（实验一即与逐步法同参数对比）；
#          subsets_ 与 cp_by_k_ 把"每层最优子集 + 定模依据"变成可检查的
#          对象——定模是否合理可以直接从表里读出来。
class BestSubsetSelector:
    """全子集选择估计器（教学版）。

    Parameters
    ----------
    method : {"exhaustive", "branch_bound"}, default="branch_bound"
        搜索算法：穷举全部子集 / 分支定界（结果一致，评估次数不同）。
    k : int or None, default=None
        选定模型规模；None 时按 Mallows Cp 最小自动选择。
    kmax : int or None, default=None
        搜索的最大规模；None 时取 min(p, n-2)。

    Attributes
    ----------
    subsets_ : dict，k -> list of int，每层最优子集
    rss_by_k_ : dict，k -> float
    cp_by_k_ : dict，k -> float
    n_evals_ : int，本次拟合的 RSS 评估次数
    k_ : int，选定规模
    selected_ : list of int，选定子集（升序）
    coef_ : ndarray, shape (p,)，全长度系数（未选中为 0）
    intercept_ : float
    rss_ : float，最终模型在训练数据上的 RSS
    r2_ : float
    """

    _METHODS = {"exhaustive": exhaustive_best_subsets,
                "branch_bound": branch_and_bound}

    def __init__(self, method="branch_bound", k=None, kmax=None):
        if method not in self._METHODS:
            raise ValueError("未知方法 %r，可选：%s"
                             % (method, sorted(self._METHODS)))
        self.method = method
        self.k = k
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
        kmax = self.kmax if self.kmax is not None else min(p, n - 2)
        if self.k is not None and not 0 <= self.k <= kmax:
            raise ValueError("k 须满足 0 <= k <= kmax=%d（当前 k=%r）" % (kmax, self.k))

        # 全模型残差方差作为 σ̂²（Cp 的需要；要求全模型可估计）
        Xc, yc = _center(X, y)
        if np.linalg.matrix_rank(Xc) < p or n - p - 1 < 1:
            raise ValueError("Mallows Cp 需要全模型可估计（满列秩且 n-p-1>=1）；"
                             "p 接近或超过 n 时请改用固定规模 k 或其他准则")
        _, rss_full, _ = fit_subset_rss(Xc, yc, list(range(p)))
        sigma2_hat = rss_full / (n - p - 1)

        best, n_evals, _ = self._METHODS[self.method](X, y, kmax=kmax)
        self.subsets_ = {k: best[k][0] for k in sorted(best)}
        self.rss_by_k_ = {k: best[k][1] for k in sorted(best)}
        self.cp_by_k_ = cp_table(best, n, sigma2_hat, kmax)
        self.n_evals_ = n_evals

        # 定模：指定 k 或 Cp 最小（并列时取较小规模，偏向简单模型）
        if self.k is not None:
            k_star = self.k
        else:
            k_star = min(self.cp_by_k_, key=lambda k: (self.cp_by_k_[k], k))
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
        """决定系数 R^2（定义与 01 模块 OLS.score、03 模块 StepwiseSelector.score 一致）。"""
        y = np.asarray(y, dtype=float)
        ss_res = float(np.sum((y - self.predict(X)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot
