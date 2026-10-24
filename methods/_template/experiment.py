"""{方法名}可运行实验（模板）。

标准流程：准备数据 → 调用实现 → 评估指标 → 生成图表 → 落盘结果。

运行方式（在仓库根目录执行）：
    python methods/{编号}-{方法名}/experiment.py
"""

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt  # noqa: E402

from lib import make_synthetic, mse, set_style, support_recovery, save_fig  # noqa: E402
# from method import fit_example  # noqa: E402  （取消注释并改为真实函数）

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")


def _write_json(name, payload):
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


def main():
    set_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # ── 实验一：{实验名称} ──────────────────────────────
    # 1) 准备数据（种子固定，保证可复现）
    # 2) 调用 method.py 中的实现
    # 3) 用 lib.metrics 计算指标
    # 4) 出图并保存到 FIG_DIR
    # 5) 数值结果写入 RES_DIR（json 存汇总，csv 存逐行数据）

    print("实验完成：图表见 figures/，数值结果见 results/")


if __name__ == "__main__":
    main()
