"""Fisher 判别分析（Fisher Discriminant Analysis, FDA）从零实现。

对应文档：methods/02-fisher-discriminant/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。
"""

import numpy as np


# ═════════════════════════════ 第 1 段：问题设定 ═════════════════════════════
# 【做什么】给定 n 个样本 X（n×p）与类别标签 y（取值于 K 个类），估计各类
#          先验 π_k、类均值 μ_k 与总均值 μ，并定义两类散度矩阵。
# 【为什么】Fisher (1936) 的问题：p 个测量如何组合成一个"最有利于分类"的
#          投影？判据是类间散度尽量大、类内散度尽量小——分类误差本质上
#          取决于类间差异与类内变差的对比。
# 【对应文档】README §2 公式 (1)(2)：
#           S_w = Σ_k Σ_{i∈C_k} (x_i - μ_k)(x_i - μ_k)^T   （类内散度）
#           S_b = Σ_k n_k (μ_k - μ)(μ_k - μ)^T             （类间散度）


def class_statistics(X, y):
    """各类先验、均值与总均值（第 1 段的代码化）。

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)

    Returns
    -------
    classes : ndarray, shape (K,)
    priors : ndarray, shape (K,)
    means : ndarray, shape (K, p)
    mu : ndarray, shape (p,)，总均值
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    K = len(classes)
    priors = np.zeros(K)
    means = np.zeros((K, X.shape[1]))
    for k, c in enumerate(classes):
        mask = y == c
        priors[k] = float(mask.mean())
        means[k] = X[mask].mean(axis=0)
    return classes, priors, means, X.mean(axis=0)


def scatter_matrices(X, y, classes, means, mu):
    """计算类内散度 S_w 与类间散度 S_b（README 公式 (2)）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    p = X.shape[1]
    Sw = np.zeros((p, p))
    Sb = np.zeros((p, p))
    for k, c in enumerate(classes):
        mask = y == c
        Xc = X[mask] - means[k]
        Sw += Xc.T @ Xc
        d = (means[k] - mu).reshape(-1, 1)
        Sb += int(mask.sum()) * (d @ d.T)
    return Sw, Sb


# ═════════════════════════════ 第 2 段：两分类 Fisher 准则 ═════════════════════════════
# 【做什么】定义广义瑞利商 J(w) = (w'S_b w)/(w'S_w w)，给出两分类闭式解。
# 【为什么】J(w) 对 w 的尺度不变、只依赖方向；最大化 J 是经典广义特征值
#          问题。两分类时 S_b 秩为 1，解可以直接写出，无需数值迭代。
# 【对应文档】README §2 公式 (3)(4)
def fit_two_class(X1, X2):
    """两分类 Fisher 判别方向：w = S_w^{-1}(mu_1 - mu_2)。

    Parameters
    ----------
    X1, X2 : ndarray, shape (n_k, p)，两类的样本矩阵

    Returns
    -------
    w : ndarray, shape (p,)，判别方向（未归一化）
    mu1, mu2 : ndarray, shape (p,)
    Sw : ndarray, shape (p, p)，合并类内散度矩阵
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
    Sw = (X1 - mu1).T @ (X1 - mu1) + (X2 - mu2).T @ (X2 - mu2)
    # S_b = (n1 n2 / n)(mu1 - mu2)(mu1 - mu2)^T 秩为 1，代入瑞利商即得
    w = np.linalg.solve(Sw, mu1 - mu2)
    return w, mu1, mu2, Sw


# ═════════════════════════════ 第 3 段：多类推广：广义特征值 ═════════════════════════════
# 【做什么】解广义特征值问题 S_b v = λ S_w v，取前 m 个特征向量构成投影
#          矩阵 W（p×m），即判别方向。
# 【为什么】K 类时单一方向不够用；S_b 的秩至多为 K-1，故有效判别方向至多
#          K-1 个——这就是"判别分析最多 K-1 维"的由来（README 公式 (5)）。
#          数值实现不直接求 S_w 的逆：先白化为标准对称特征值问题
#          (S_w^{-1/2} S_b S_w^{-1/2}) u = λ u，再变换回 v = S_w^{-1/2} u。
#          对称化保证数值稳定，且截断数值零特征值即可处理秩亏的 S_w。
# 【对应文档】README §2 公式 (5)
def solve_directions(Sw, Sb, n_components=None, rcond=None):
    """解广义特征值问题，返回判别方向与特征值。

    Parameters
    ----------
    Sw, Sb : ndarray, shape (p, p)
    n_components : int or None，保留方向数；None 表示全部
    rcond : float or None，特征值截断阈值（相对最大特征值）；
        None 表示按机器精度默认截断（与 OLS 模块 fit_svd 默认策略一致），
        0 表示不截断（调用者自负其责）

    Returns
    -------
    W : ndarray, shape (p, m)，判别方向（相互正交）
    lam : ndarray, shape (m,)，对应的广义特征值（降序）
    """
    d, V = np.linalg.eigh(Sw)
    tol = (max(Sw.shape) * np.finfo(d.dtype).eps * d.max()
           if rcond is None else rcond * d.max())
    keep = d > tol
    # 截断版 S_w^{-1/2}：被剔除的方向权重置 0（而非 inf），天然得到最小范数行为
    d_inv_sqrt = np.where(keep, 1.0 / np.sqrt(np.where(keep, d, 1.0)), 0.0)
    Wh = (V * d_inv_sqrt) @ V.T
    M = Wh @ Sb @ Wh                       # 对称半正定
    lam, U = np.linalg.eigh(M)
    order = np.argsort(lam)[::-1]          # eigh 返回升序，此处转为降序
    lam, U = lam[order], U[:, order]
    m = len(lam) if n_components is None else min(len(lam), n_components)
    return Wh @ U[:, :m], lam[:m]


# ═════════════════════════════ 第 4 段：决策规则与病态处理 ═════════════════════════════
# 【做什么】在正态、等协方差假设下，后验概率的对数是 x 的线性函数，由此
#          给出各类线性判别函数 δ_k(x)；并提供 S_w 病态/奇异时的收缩稳定化。
# 【为什么】投影方向回答"往哪投"，判别函数回答"投完怎么判"。S_w 在 p>n 或
#          特征强相关时奇异/病态，直接求逆会失败或放大误差——与 OLS 模块
#          实验三的正规方程教训同源（README §5.3）。
# 【对应文档】README §2 公式 (6)、§3 步骤 4
def shrink_sw(Sw, reg):
    """对 S_w 做对角线收缩：S_w_reg = (1-reg) S_w + reg * mean(diag(S_w)) * I。

    reg=0 不收缩；reg>0 时结果正定（只要原 S_w 半正定）。
    S_w 已近似各向同性时该操作近似恒等。
    说明：这是教学用的各向同性收缩（数值稳定化目的）；作为统计方法，
    收缩判别分析（RDA）的系统阐述见 README §7 文献 [63]。
    """
    Sw = np.asarray(Sw, dtype=float)
    if reg <= 0:
        return Sw
    return (1.0 - reg) * Sw + reg * np.diag(Sw).mean() * np.eye(Sw.shape[0])


def linear_scores(X, means, cov, priors):
    """各类线性判别函数 δ_k(x) = x' Σ^{-1} μ_k - ½ μ_k' Σ^{-1} μ_k + log π_k。

    用 solve 逐类求解而非显式构造 Σ^{-1}。返回 (n, K) 得分矩阵。
    """
    X = np.asarray(X, dtype=float)
    scores = np.empty((X.shape[0], len(means)))
    for k in range(len(means)):
        w_k = np.linalg.solve(cov, means[k])
        scores[:, k] = X @ w_k - 0.5 * (means[k] @ w_k) + np.log(priors[k])
    return scores


# ═════════════════════════════ 第 5 段：统一估计器接口 ═════════════════════════════
# 【做什么】封装 FisherDiscriminant 类，接口与属性名对齐 sklearn 的
#          LinearDiscriminantAnalysis（fit / transform / decision_function /
#          predict / score），使实验二可以直接对拍。
# 【为什么】接口一致是"对拍"的前提；属性名对齐还方便读者对照 sklearn
#          官方文档学习（README §4）。
class FisherDiscriminant:
    """Fisher 判别分析估计器（教学版）。

    Parameters
    ----------
    reg : float in [0, 1], default=0.0
        对 S_w 的收缩强度；0 为不收缩。
    rcond : float or None, default=None
        方向求解的特征值截断阈值（相对最大特征值）；None 表示按机器精度
        默认截断，0 表示不截断。
    n_components : int or None, default=None
        保留的判别方向数；None 表示 min(p, K-1)。

    Attributes
    ----------
    classes_ : ndarray, shape (K,)
    priors_ : ndarray, shape (K,)
    means_ : ndarray, shape (K, p)
    scalings_ : ndarray, shape (p, m)，判别方向（相互正交）
    explained_variance_ratio_ : ndarray, shape (m,)
    coef_ : ndarray, shape (K, p)
    intercept_ : ndarray, shape (K,)
    """

    def __init__(self, reg=0.0, rcond=None, n_components=None):
        self.reg = reg
        self.rcond = rcond
        self.n_components = n_components

    def fit(self, X, y):
        """拟合模型，返回 self。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape
        # 1) 类别统计与散度矩阵
        self.classes_, self.priors_, self.means_, mu = class_statistics(X, y)
        K = len(self.classes_)
        Sw, Sb = scatter_matrices(X, y, self.classes_, self.means_, mu)
        # 2) 收缩稳定化（reg>0 时 S_w 正定）
        Sw_used = shrink_sw(Sw, self.reg)
        # 3) 判别方向：白化后标准特征值问题的特征向量（内部按 rcond 截断）
        m = self.n_components if self.n_components else min(p, K - 1)
        W, lam = solve_directions(Sw_used, Sb, n_components=m,
                                  rcond=self.rcond)
        self.scalings_ = W
        total = lam.sum()
        self.explained_variance_ratio_ = (lam / total if total > 0
                                          else np.zeros_like(lam))
        # 4) 线性判别函数：合并协方差 = S_w_used / (n - K)
        cov = Sw_used / (n - K)
        self.coef_ = np.linalg.solve(cov, self.means_.T).T
        self.intercept_ = (-0.5 * np.sum(self.means_ * self.coef_, axis=1)
                           + np.log(self.priors_))
        return self

    def transform(self, X):
        """投影到判别方向，返回 (n, m) 判别得分。"""
        return np.asarray(X, dtype=float) @ self.scalings_

    def decision_function(self, X):
        """各类线性判别函数值，返回 (n, K)。"""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X):
        """预测类别：判别函数取值最大的类。"""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        """分类准确率。"""
        y = np.asarray(y)
        return float(np.mean(y == self.predict(X)))
