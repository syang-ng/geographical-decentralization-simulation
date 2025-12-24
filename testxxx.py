#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比两种方法（精确 Poisson-Binomial vs 正态近似）的运行时间与结果。

依赖：
  pip install numpy scipy tqdm
  以及你当前用的 Poisson-Binomial 包（提供 poisson_binom 或类似接口）

运行：
  python compare_threshold_methods.py

说明：
- “精确法”用你当前的 poisson_binom(probabilities).sf(k-1) 逻辑。
- “近似法”用正态近似（带连续性修正）计算 P(S >= k)，速度 O(n)。
- 两者都用同样的二分搜索找到最小 threshold，使得概率 >= target_prob。
"""

import time
import numpy as np
from scipy.stats import lognorm, norm
from functools import lru_cache
from tqdm import tqdm

# -----------------------------
# 1) 这里请按你的环境修改/导入 Poisson-Binomial
# -----------------------------


from scipy.stats import norm, lognorm, poisson_binom


# -----------------------------
# 2) 你的“精确法”原始实现（做了最小改动）
#    注意：这里保留 lru_cache 只是为了复现你现状；后面会在对比里展示它的代价
# -----------------------------
@lru_cache(maxsize=1024)
def evaluate_threshold_exact(
    broadcast_latencies,  # tuple
    broadcast_stds,       # tuple
    threshold,
    required_attesters
):
    if poisson_binom is None:
        raise RuntimeError(
            "没有导入 poisson_binom。请在脚本顶部修改导入语句，使其指向你实际使用的 Poisson-Binomial 包。"
        )

    if not broadcast_latencies:
        return 0.0

    latencies = np.array(broadcast_latencies, dtype=np.float64)
    stds = np.array(broadcast_stds, dtype=np.float64)

    probabilities = np.zeros_like(latencies)

    zero_latency_mask = (latencies <= 0)
    zero_std_mask = (stds <= 0) & ~zero_latency_mask
    valid_mask = ~zero_latency_mask & ~zero_std_mask

    probabilities[zero_latency_mask] = 1.0
    probabilities[zero_std_mask] = np.where(latencies[zero_std_mask] < threshold, 1.0, 0.0)

    if np.any(valid_mask):
        valid_latencies = latencies[valid_mask]
        std_dev = valid_latencies * stds[valid_mask]

        mean_sq = valid_latencies**2
        std_dev_sq = std_dev**2

        mu = np.log(mean_sq / np.sqrt(mean_sq + std_dev_sq))
        sigma = np.sqrt(np.log(1 + (std_dev_sq / mean_sq)))

        probabilities[valid_mask] = lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))

    pb = poisson_binom(probabilities.tolist())
    return pb.sf(required_attesters - 1)


def find_min_threshold_exact(
    broadcast_latencies,  # tuple
    broadcast_stds,       # tuple
    required_attesters,
    target_prob=0.99,
    threshold_low=0.0,
    threshold_high=4000.0,
    tolerance=1.0
):
    lo, hi = threshold_low, threshold_high
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if mid <= 0:
            lo = tolerance
            continue

        prob = evaluate_threshold_exact(
            broadcast_latencies,
            broadcast_stds,
            threshold=mid,
            required_attesters=required_attesters
        )

        if prob >= target_prob:
            hi = mid
        else:
            lo = mid

    return (hi + lo) / 2.0


# -----------------------------
# 3) “近似法”：预计算 lognormal 参数 + 正态近似计算 P(S >= k)
# -----------------------------
class ThresholdEvaluatorApprox:
    def __init__(self, broadcast_latencies, broadcast_stds):
        lat = np.asarray(broadcast_latencies, dtype=np.float64)
        stdr = np.asarray(broadcast_stds, dtype=np.float64)

        self.lat = lat
        self.stdr = stdr

        self.zero_latency_mask = (lat <= 0)
        self.zero_std_mask = (stdr <= 0) & ~self.zero_latency_mask
        self.valid_mask = ~self.zero_latency_mask & ~self.zero_std_mask

        vlat = lat[self.valid_mask]
        vstdr = stdr[self.valid_mask]

        std_dev = vlat * vstdr
        mean_sq = vlat**2
        std_dev_sq = std_dev**2

        mu = np.log(mean_sq / np.sqrt(mean_sq + std_dev_sq))
        sigma = np.sqrt(np.log(1.0 + (std_dev_sq / mean_sq)))

        self._ln_sigma = sigma
        self._ln_scale = np.exp(mu)

        # 复用 buffer，避免每次分配 10k 数组
        self._p = np.empty_like(lat, dtype=np.float64)

    def survival_ge_k(self, threshold, k):
        p = self._p

        # 条件 1
        p[self.zero_latency_mask] = 1.0

        # 条件 2
        if np.any(self.zero_std_mask):
            lat_zs = self.lat[self.zero_std_mask]
            p[self.zero_std_mask] = (lat_zs < threshold).astype(np.float64)

        # 条件 3
        if np.any(self.valid_mask):
            p[self.valid_mask] = lognorm.cdf(
                threshold,
                s=self._ln_sigma,
                scale=self._ln_scale
            )

        mu = p.sum()
        var = np.sum(p * (1.0 - p))
        if var <= 0.0:
            return 1.0 if mu >= k else 0.0

        std = np.sqrt(var)

        # 简单剪枝：远离边界就直接返回 0/1（可按需要调）
        if mu + 8.0 * std < k:
            return 0.0
        if mu - 8.0 * std >= k:
            return 1.0

        # 连续性修正：P(S >= k) ≈ 1 - Phi((k - 0.5 - mu)/std)
        z = ((k - 0.5) - mu) / std
        return norm.sf(z)


def find_min_threshold_approx(
    evaluator: ThresholdEvaluatorApprox,
    required_attesters,
    target_prob=0.99,
    threshold_low=0.0,
    threshold_high=4000.0,
    tolerance=1.0
):
    lo, hi = threshold_low, threshold_high
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if mid <= 0:
            lo = tolerance
            continue

        prob = evaluator.survival_ge_k(mid, required_attesters)
        if prob >= target_prob:
            hi = mid
        else:
            lo = mid

    return (hi + lo) / 2.0


# -----------------------------
# 4) 生成一组可复现实验数据（你也可以替换成真实数据）
# -----------------------------
def make_synthetic_data(n=10_000, seed=0):
    rng = np.random.default_rng(seed)

    # latency: 以 ms 为单位，带一些小概率的 0/负值模拟异常
    lat = rng.lognormal(mean=np.log(200.0), sigma=0.6, size=n)
    anomaly = rng.random(n)
    lat[anomaly < 0.002] = 0.0
    lat[(anomaly >= 0.002) & (anomaly < 0.004)] = -1.0

    # std_ratio: 你代码里把 std_dev = latency * std_ratio
    stdr = rng.uniform(0.05, 0.50, size=n)
    # 一点 std<=0 的情况
    stdr[rng.random(n) < 0.002] = 0.0

    return tuple(lat.tolist()), tuple(stdr.tolist())


# -----------------------------
# 5) 对比主程序
# -----------------------------
def main():
    n = 10_000
    target_prob = 0.99
    tolerance = 1.0

    # 这里调 required_attesters；例如 10k 中要 7000 人
    required_attesters = 7000

    lat_tup, std_tup = make_synthetic_data(n=n, seed=42)

    print(f"n={n}, required_attesters={required_attesters}, target_prob={target_prob}, tolerance={tolerance}")

    # 近似法：预计算
    t0 = time.perf_counter()
    evaluator = ThresholdEvaluatorApprox(lat_tup, std_tup)
    t1 = time.perf_counter()
    thr_approx = find_min_threshold_approx(
        evaluator,
        required_attesters=required_attesters,
        target_prob=target_prob,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=tolerance
    )
    t2 = time.perf_counter()

    print("\n[Approx] 正态近似")
    print(f"  build evaluator: {(t1 - t0)*1000:.2f} ms")
    print(f"  threshold: {thr_approx:.3f}")
    print(f"  total time: {(t2 - t0):.3f} s")

    # 精确法：注意 evaluate_threshold_exact 带 lru_cache，会对大 tuple 做哈希
    if poisson_binom is None:
        print("\n[Exact] 跳过：未导入 poisson_binom（请修改脚本顶部 import）")
        return

    # 清空 cache，避免上次跑影响
    evaluate_threshold_exact.cache_clear()

    t3 = time.perf_counter()
    thr_exact = find_min_threshold_exact(
        lat_tup, std_tup,
        required_attesters=required_attesters,
        target_prob=target_prob,
        threshold_low=0.0,
        threshold_high=4000.0,
        tolerance=tolerance
    )
    t4 = time.perf_counter()

    print("\n[Exact] Poisson-Binomial 精确法（复现你当前逻辑）")
    print(f"  threshold: {thr_exact:.3f}")
    print(f"  total time: {(t4 - t3):.3f} s")

    # 结果差异
    print("\n[Diff]")
    print(f"  abs(thr_exact - thr_approx) = {abs(thr_exact - thr_approx):.3f}")

    # 可选：在阈值点再对比一次概率（近似 vs 精确）
    p_approx = evaluator.survival_ge_k(thr_exact, required_attesters)
    p_exact = evaluate_threshold_exact(lat_tup, std_tup, thr_exact, required_attesters)
    print("\n[Prob at thr_exact]")
    print(f"  approx prob: {p_approx:.6f}")
    print(f"  exact  prob: {p_exact:.6f}")
    print(f"  abs diff:     {abs(p_exact - p_approx):.6f}")


if __name__ == "__main__":
    main()
