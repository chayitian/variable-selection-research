"""最小二乘（Ordinary Least Squares, OLS）从零实现。

对应文档：methods/01-least-squares/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。
"""

import numpy as np


# ═════════════════════════════ 第 1 段：问题设定 ═════════════════════════════
# 【做什么】给定设计矩阵 X（n×p）与响应 y（n 维），估计线性系数 beta。
# 【为什么】Legendre (1805) 面对的天文学问题正是"方程多于未知数"：
#          n 个观测方程无法被 p 个参数同时精确满足，需要一条最优准则。
# 【对应文档】README §2 公式 (2)：S(beta) = ||y - X beta||^2


# ═════════════════════════════ 第 2 段：正规方程解法 ═════════════════════════════
# 【做什么】直接解正规方程 X^T X beta = X^T y。
# 【为什么】S(beta) 是凸二次函数，驻点即全局最小；对 beta 求梯度令为零，
#          恰好得到这组线性方程（README 公式 (3)-(4)）。
# 【注意】X^T X 的条件数是 X 的条件数的平方，病态时数值误差被平方级放大。
def fit_normal_equation(X, y):
    """正规方程闭式解：beta_hat = (X^T X)^{-1} X^T y。

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)

    Returns
    -------
    beta_hat : ndarray, shape (n_features,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # 解 X^T X beta = X^T y（README 公式 (4)）。
    # 用 solve 而非显式构造 (X^T X)^{-1}：求逆会额外放大条件数。
    # 陷阱：X^T X 的条件数是 cond(X) 的平方，病态时应改用 fit_qr / fit_svd；
    # X 秩亏时 X^T X 数值奇异，solve 会直接抛 LinAlgError（见 README §5.3）。
    return np.linalg.solve(X.T @ X, X.T @ y)


# ═════════════════════════════ 第 3 段：QR 分解解法 ═════════════════════════════
# 【做什么】先做 QR 分解 X = Q R，再解 R beta = Q^T y。
# 【为什么】QR 不放大条件数，数值稳定性优于正规方程；R 为上三角矩阵，
#          可用回代高效求解（README 公式 (8)）。
def fit_qr(X, y):
    """QR 分解法：X = Q R，beta_hat = R^{-1} Q^T y。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Q, R = np.linalg.qr(X)
    # Q 正交：||y - Xb|| = ||Q^T y - R b||，问题化为解上三角方程 R b = Q^T y。
    # 严格做法是回代（scipy.linalg.solve_triangular，更快且专为三角结构设计）；
    # 此处用通用 solve 是教学优先，两者数值结果一致。
    # 前提：X 满列秩（R 非奇异）；秩亏时应改用 fit_svd。
    return np.linalg.solve(R, Q.T @ y)


# ═════════════════════════════ 第 4 段：SVD 解法 ═════════════════════════════
# 【做什么】做 SVD X = U diag(s) V^T，得 beta = V diag(1/s) U^T y。
# 【为什么】SVD 对秩亏/近秩亏矩阵也能给出最小范数解，是三种解法中数值
#          最稳的（README 公式 (9)）；截断过小奇异值还可进一步抑制病态。
def fit_svd(X, y, rcond=None):
    """SVD 法：beta_hat = V diag(1/s) U^T y（README 公式 (9)）。

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    rcond : float or None
        奇异值截断阈值（相对最大奇异值）。None 表示默认截断：与
        np.linalg.lstsq 一致，按 cutoff = max(n, p) * eps * s.max()
        剔除数值零奇异值，使秩亏输入也能安全返回最小范数解；
        传正数使用该相对阈值；传 0 表示不截断（调用者自负其责）。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if rcond is None:
        # 默认截断：机器精度级别的数值零奇异值置 inf，避免除零得到 inf/nan
        cutoff = max(X.shape) * np.finfo(s.dtype).eps * s.max()
        s = np.where(s > cutoff, s, np.inf)
    elif rcond > 0:
        # 显式相对阈值：把小于阈值的历史奇异值置为 inf，对应项在 1/s 中
        # 归零，等价于截断伪逆，秩亏时给出最小范数解
        cutoff = rcond * s.max()
        s = np.where(s > cutoff, s, np.inf)
    # beta = V diag(1/s) U^T y（README 公式 (9)），逐项对应右侧的矩阵乘积
    return Vt.T @ ((U.T @ y) / s)


# ═════════════════════════════ 第 5 段：统一估计器接口 ═════════════════════════════
# 【做什么】封装 OLS 类，支持三种解法与截距项，接口对齐 sklearn 惯例
#          （fit / predict / score），便于后续方法与 sklearn 对拍。
# 【为什么】截距通过对 X、y 中心化实现，等效于在设计矩阵中加一列 1，
#          但数值上更稳（README §3 步骤 4）。
class OLS:
    """最小二乘线性回归估计器。

    Parameters
    ----------
    solver : {"qr", "normal", "svd"}, default="qr"
        数值解法；默认 QR（精度与速度的平衡）。
    fit_intercept : bool, default=True
        是否估计截距项（通过对数据中心化实现）。

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)
    intercept_ : float
    """

    # 名称 → 解法函数的分发表；新增解法只需在此注册一行
    _SOLVERS = {"normal": fit_normal_equation, "qr": fit_qr, "svd": fit_svd}

    def __init__(self, solver="qr", fit_intercept=True):
        # 默认 QR：不放大条件数（优于 normal），又无需 SVD 的完整分解开销
        if solver not in self._SOLVERS:
            raise ValueError("未知解法 %r，可选：%s" % (solver, sorted(self._SOLVERS)))
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self._x_mean = None

    def fit(self, X, y):
        """拟合模型，返回 self。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            # 中心化后拟合等效于设计矩阵加一列 1，但避免显式构造该列、条件数更小
            self._x_mean = X.mean(axis=0)
            y_mean = y.mean()
            Xc = X - self._x_mean
            yc = y - y_mean
        else:
            self._x_mean = np.zeros(X.shape[1])
            y_mean = 0.0
            Xc, yc = X, y
        self.coef_ = self._SOLVERS[self.solver](Xc, yc)
        # 还原截距：b = mean(y) - mean(X) @ coef（代入中心化后模型的期望即得）
        self.intercept_ = float(y_mean - self._x_mean @ self.coef_)
        return self

    def predict(self, X):
        """预测：X @ coef_ + intercept_。"""
        if self.coef_ is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit。")
        X = np.asarray(X, dtype=float)
        # 线性预测 X @ coef_ + intercept_（README §3 步骤 4）
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """决定系数 R^2 = 1 - SS_res / SS_tot。"""
        y = np.asarray(y, dtype=float)
        # R^2 = 1 - SS_res / SS_tot（未校正定义，与 sklearn r2_score 一致）
        ss_res = float(np.sum((y - self.predict(X)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot
