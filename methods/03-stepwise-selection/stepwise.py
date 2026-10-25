"""逐步选择法（Stepwise Selection）从零实现。

对应文档：methods/03-stepwise-selection/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。
"""

import numpy as np
from scipy import stats

_EPS = 1e-12


# ═════════════════════════════ 第 1 段：问题设定与公共内核 ═════════════════════════════
# 【做什么】三种变体共享同一个内核：在候选变量的某个子集 S 上拟合 OLS
#          （截距经中心化实现，与 01 模块一致），用残差平方和 RSS(S) 与
#          偏 F 检验度量单个候选变量的"进入/退出资格"。
# 【为什么】逐步法的每一步都是 OLS，筛选逻辑完全建立在"加入或剔除某变量
#          能使 RSS 下降多少"之上，判据必须可计算、可比较。
# 【对应文档】README §2.1、公式 (1)(2)


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
    beta, _, rank, _ = np.linalg.lstsq(A, yc, rcond=None)
    resid = yc - A @ beta
    return beta, float(resid @ resid), int(rank)


def partial_f(Xc, yc, base_cols, cand_col):
    """把 cand_col 加入 base 模型的偏 F 检验（README 公式 (2)）。

    原假设：该变量系数为零。F = (RSS_base - RSS_full) / (RSS_full / df_res)，
    其中 df_res = n - |base| - 2 为加入后模型的残差自由度。

    Parameters
    ----------
    base_cols : sequence of int，进入前的子集
    cand_col : int，候选变量

    Returns
    -------
    F : float，F 统计量；候选不可用时返回 0.0
    p : float，上尾概率；候选不可用时返回 1.0
    """
    n = len(yc)
    base = list(base_cols)
    _, rss_base, rank_base = fit_subset_rss(Xc, yc, base)
    _, rss_full, rank_full = fit_subset_rss(Xc, yc, base + [cand_col])
    # 边界 1：秩不增（共线或重复列）——该候选不带来新信息，判据不可用
    if rank_full <= rank_base:
        return 0.0, 1.0
    # 边界 2：数值上 RSS 无改善——同样视为不可用
    if rss_full >= rss_base - _EPS * max(1.0, rss_base):
        return 0.0, 1.0
    df_res = n - len(base) - 2
    # 边界 3：残差自由度耗尽（样本太少）——无法构造检验
    if df_res <= 0:
        return 0.0, 1.0
    F = (rss_base - rss_full) / (rss_full / df_res)
    return float(F), float(stats.f.sf(F, 1, df_res))


# ═════════════════════════════ 第 2 段：前向选择 ═════════════════════════════
# 【做什么】从空模型开始，每步在未入选候选中选偏 F 检验 p 值最小者，
#          p < p_in 则引入，否则停止（Efroymson 1960 的原始算法）。
# 【为什么】前向是三种变体中最廉价的一种：起点不要求全模型可估计，
#          天然适合候选多、样本少的场景；代价是早期误入的变量无法被纠正。
# 【对应文档】README §2.2、公式 (3)
def forward_select(X, y, p_in=0.05, max_steps=None):
    """前向选择（Forward Selection）。

    Parameters
    ----------
    X, y : ndarray
    p_in : float，进入阈值（F 检验 p 值）
    max_steps : int or None，最多引入步数；None 表示不超过变量总数

    Returns
    -------
    selected : list of int，按进入顺序的列索引
    history : list of dict，每步记录（step/action/variable/F/p_value/rss）
    """
    Xc, yc = _center(X, y)
    p = Xc.shape[1]
    if max_steps is None:
        max_steps = p
    selected, history = [], []
    for step in range(1, max_steps + 1):
        best = None
        for j in range(p):
            if j in selected:
                continue
            F, pval = partial_f(Xc, yc, selected, j)
            if pval < p_in and (best is None or pval < best[2]):
                best = (j, F, pval)
        if best is None:
            break                       # 无候选达到进入阈值，正常停止
        j, F, pval = best
        selected.append(j)
        rss = fit_subset_rss(Xc, yc, selected)[1]
        history.append({"step": step, "action": "enter", "variable": j,
                        "F": F, "p_value": pval, "rss": rss})
    return selected, history


# ═════════════════════════════ 第 3 段：后向消除 ═════════════════════════════
# 【做什么】从全模型开始，每步剔除偏 F 检验 p 值最大者（p > p_out），
#          否则停止。
# 【为什么】后向的起点是全模型，能看到每个变量在"其余都在"条件下的净贡献，
#          不易被单个变量的边际表现误导；代价是要求 n - p - 1 >= 1 且
#          设计矩阵满列秩，候选多于样本时不可用（README §3 边界条件）。
# 【对应文档】README §2.3、公式 (4)
def backward_eliminate(X, y, p_out=0.10, min_cols=1, max_steps=None):
    """后向消除（Backward Elimination）。

    Parameters
    ----------
    X, y : ndarray
    p_out : float，剔除阈值
    min_cols : int，至少保留的预测变量数（默认 1；置 0 允许退到仅截距模型）
    max_steps : int or None

    Returns
    -------
    selected, history：同 forward_select
    """
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    # 边界：全模型必须可估计，否则后向无从起步
    if n - p - 1 < 1:
        raise ValueError(
            "后向消除要求全模型可估计：n - p - 1 >= 1（当前 n=%d, p=%d）" % (n, p))
    if np.linalg.matrix_rank(Xc) < p:
        raise ValueError("后向消除要求设计矩阵满列秩（存在共线列时请改用前向/逐步）")
    if max_steps is None:
        max_steps = p
    selected, history = list(range(p)), []
    for step in range(1, max_steps + 1):
        if len(selected) <= min_cols:
            break
        worst = None
        for j in selected:
            others = [c for c in selected if c != j]
            F, pval = partial_f(Xc, yc, others, j)
            if worst is None or pval > worst[2]:
                worst = (j, F, pval)
        j, F, pval = worst
        if pval <= p_out:
            break                       # 最差变量仍显著，正常停止
        selected.remove(j)
        rss = fit_subset_rss(Xc, yc, selected)[1]
        history.append({"step": step, "action": "remove", "variable": j,
                        "F": F, "p_value": pval, "rss": rss})
    return selected, history


# ═════════════════════════════ 第 4 段：逐步回归（双向变体） ═════════════════════════════
# 【做什么】每步依次检查两个动作（Draper & Smith 式标准流程）：
#          先看剔除——已入选者中 p 值最大者是否超过剔除阈值 p_out；
#          再看进入——未入选者中 p 值最小者是否达到进入阈值 p_in；
#          两者都不满足则收敛。
# 【为什么】双向是逐步回归的狭义形态：早期误入的变量还有机会被剔除，
#          比单向更稳健。p_in < p_out 的滞回设计避免变量"进了又出、
#          出了又进"的震荡（README 公式 (5)）。
#          注意：不要改成"进入者须比最差成员更显著"的变体——那会让模型
#          在早期就因成员过于显著而拒绝一切新变量，过早收敛。
# 【对应文档】README §2.4、公式 (5)
def stepwise_select(X, y, p_in=0.05, p_out=0.10, max_steps=None, min_cols=1):
    """逐步回归（Stepwise Regression，双向变体）。

    每步流程：(1) 最差已入选变量 p > p_out 则剔除；(2) 否则最佳未入选
    变量 p < p_in 则进入；(3) 否则停止。

    Parameters
    ----------
    p_in : float，进入阈值
    p_out : float，剔除阈值（须 >= p_in，即滞回条件）
    max_steps : int or None；None 时取 2p + 10 作为防御性上限
    min_cols : int，至少保留的预测变量数

    Returns
    -------
    selected, history：同 forward_select
    """
    if p_out < p_in:
        raise ValueError("逐步回归要求 p_out >= p_in（滞回条件，见 README §2.4）")
    Xc, yc = _center(X, y)
    n, p = Xc.shape
    if max_steps is None:
        max_steps = 2 * p + 10
    selected, history = [], []
    for step in range(1, max_steps + 1):
        # 动作 1：剔除——最差已入选变量是否超过剔除阈值
        worst_out = None
        for j in selected:
            others = [c for c in selected if c != j]
            F, pval = partial_f(Xc, yc, others, j)
            if worst_out is None or pval > worst_out[2]:
                worst_out = (j, F, pval)
        if (worst_out is not None and worst_out[2] > p_out
                and len(selected) > min_cols):
            j, F, pval = worst_out
            selected.remove(j)
            action = "remove"
        else:
            # 动作 2：进入——最佳未入选变量是否达到进入阈值
            best_in = None
            for j in range(p):
                if j in selected:
                    continue
                F, pval = partial_f(Xc, yc, selected, j)
                if pval < p_in and (best_in is None or pval < best_in[2]):
                    best_in = (j, F, pval)
            if best_in is None:
                break                   # 无可进入者，收敛
            j, F, pval = best_in
            selected.append(j)
            action = "enter"
        rss = fit_subset_rss(Xc, yc, selected)[1]
        history.append({"step": step, "action": action, "variable": j,
                        "F": F, "p_value": pval, "rss": rss})
    return selected, history


# ═════════════════════════════ 第 5 段：统一估计器接口 ═════════════════════════════
# 【做什么】封装 StepwiseSelector 类，对外暴露与 01 模块 OLS 类一致的
#          fit / predict / score 接口，并输出选择过程的可审计信息。
# 【为什么】接口一致便于跨模块对照实验（实验一即与 01 模块同参数对比）；
#          history_ 与 rss_path_ 把"选择过程"变成可检查的对象——逐步法
#          的核心缺陷（不稳定性）正是通过审视过程而非仅看结果暴露的。
class StepwiseSelector:
    """逐步选择法估计器（教学版）。

    Parameters
    ----------
    method : {"forward", "backward", "stepwise"}, default="stepwise"
        变体：前向选择 / 后向消除 / 逐步回归（双向）。
    p_in : float in (0, 1), default=0.05
        进入阈值（F 检验 p 值）。
    p_out : float in (0, 1), default=0.10
        剔除阈值（后向/逐步使用；须 >= p_in）。
    max_steps : int or None, default=None
        最多步数；None 时按各变体默认（前向/后向为 p，逐步为 2p + 10）。
    min_cols : int, default=1
        至少保留的预测变量数（后向/逐步）。

    Attributes
    ----------
    selected_ : list of int，选中的列索引（升序）
    coef_ : ndarray, shape (p,)，全长度系数（未选中位置为 0）
    intercept_ : float
    rss_ : float，最终模型在训练数据上的 RSS
    r2_ : float，最终模型的决定系数
    rss_path_ : list of float，每步之后的 RSS（首元素为仅截距模型的 TSS）
    history_ : list of dict，每步记录
    n_steps_ : int
    """

    _METHODS = {"forward": forward_select,
                "backward": backward_eliminate,
                "stepwise": stepwise_select}

    def __init__(self, method="stepwise", p_in=0.05, p_out=0.10,
                 max_steps=None, min_cols=1):
        if method not in self._METHODS:
            raise ValueError("未知方法 %r，可选：%s"
                             % (method, sorted(self._METHODS)))
        if not (0.0 < p_in < 1.0) or not (0.0 < p_out < 1.0):
            raise ValueError("p_in / p_out 须为 (0, 1) 内的概率阈值")
        if p_out < p_in:
            raise ValueError("p_out 须 >= p_in（滞回条件，见 README §2.4）")
        if min_cols < 0:
            raise ValueError("min_cols 须为非负整数")
        self.method = method
        self.p_in = p_in
        self.p_out = p_out
        self.max_steps = max_steps
        self.min_cols = min_cols

    def fit(self, X, y):
        """拟合模型，返回 self。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 须为二维数组 (n_samples, n_features)")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y 须为长度与 X 行数一致的一维数组")
        Xc, yc = _center(X, y)
        fn = self._METHODS[self.method]
        if self.method == "forward":
            selected, history = fn(X, y, p_in=self.p_in, max_steps=self.max_steps)
        elif self.method == "backward":
            selected, history = fn(X, y, p_out=self.p_out,
                                   max_steps=self.max_steps,
                                   min_cols=self.min_cols)
        else:
            selected, history = fn(X, y, p_in=self.p_in, p_out=self.p_out,
                                   max_steps=self.max_steps,
                                   min_cols=self.min_cols)
        self.selected_ = sorted(selected)
        self.history_ = history
        self.n_steps_ = len(history)
        self.rss_path_ = [float(yc @ yc)] + [h["rss"] for h in history]
        # 最终系数：选中子集上的 OLS（与算法内核一致，便于审计）
        beta, rss, _ = fit_subset_rss(Xc, yc, self.selected_)
        self.coef_ = np.zeros(X.shape[1])
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
        """决定系数 R^2（定义与 01 模块 OLS.score 一致）。"""
        y = np.asarray(y, dtype=float)
        ss_res = float(np.sum((y - self.predict(X)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot
