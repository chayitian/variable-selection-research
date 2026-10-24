"""{方法名}从零实现（模板）。

对应文档：methods/{编号}-{方法名}/README.md
本文为教学代码：按"分段讲解"组织，每段注释回答三件事——
做什么 / 为什么 / 对应 README 中哪一条公式。追求可读性而非性能。
"""

import numpy as np


# ═════════════════════════════ 第 1 段：{段落标题} ═════════════════════════════
# 【做什么】……
# 【为什么】……
# 【对应文档】README §2 公式 (1)


def fit_example(X, y):
    """示例函数（替换为真实实现）。

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
    raise NotImplementedError("请在此实现")
