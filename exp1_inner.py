"""
1단계. 내적 실험 (합성 벡터)

질문: 같은 비트폭 b에서 TurboQuant_mse와 TurboQuant_prod로 압축한 키로
      쿼리와의 내적을 추정할 때, 편향과 분산은 각각 얼마인가?
      그리고 그 값은 실험 전에 고정한 예측표(STEP 0)와 맞는가?

설계
  - 독립변수(2범주): 압축 방식 mse / prod
  - 조건: d = 64, 128  ×  b = 2, 3, 4  (6개 조건)
  - 반복 단위: 난수 시드 30개. 시드마다 회전행렬 Π와 QJL 행렬 S를 새로 뽑고,
    같은 시드 안에서는 두 방식이 같은 Π와 같은 데이터를 쓴다 (대응 설계).
  - 시드 하나에서 쓰는 데이터
      (가) 무작위 쌍 4,000개: 서로 독립인 단위벡터 키 k와 쿼리 q (실제 내적 ≈ 0 근처)
           → 오차 분산 × d,  평균제곱오차 × d
      (나) 상관 쌍 4,000개: 실제 내적이 정확히 ρ = 0.5인 키와 쿼리
           → 상대 편향 = 평균(추정 − 실제) / 실제
      (다) 고정 수직 쌍 1,000개: 시드가 바뀌어도 같은 쌍(실제 내적 0)을 계속 쓴다
           → 쌍마다 '회전이 바뀔 때의 흩어짐'(시드 30개에 걸친 분산)을 구해 평균 × d
              (편향 성분이 섞이지 않은 순수한 흩어짐. 예측표의 mse 분산 D(1−D)와 같은 정의)
  - 시드마다 지표 하나씩 → 방식별로 30개 값
  - 통계: 평균과 95% 신뢰구간(t분포, 자유도 29),
          방식 간 대응표본 t검정 + 윌콕슨 부호순위 검정, 효과크기 d_z,
          본페로니 보정 (조건 6개 → 유의수준 0.05/6)
  - 예측표와 비교: 실측 평균이 예측값의 ±10% 이내인지
"""
import json
import math
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy import stats

from tq import TurboQuantMSE, TurboQuantProd, random_rotation
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style

apply_style()

DT = torch.float64
DIMS = (64, 128)
BITS = (2, 3, 4)
N_SEED = 30
N_PAIRS = 4000
N_FIX = 1000
RHO = 0.5
SEED_BASE = 100_000          # 0단계와 겹치지 않는 시드 범위
ALPHA = 0.05
N_COND = len(DIMS) * len(BITS)
T_CRIT = stats.t.ppf(0.975, N_SEED - 1)   # 2.045


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def make_data(d, gen):
    """(가) 무작위 쌍, (나) 실제 내적이 RHO인 상관 쌍."""
    k_rand = unit(torch.randn(N_PAIRS, d, generator=gen, dtype=DT))
    q_rand = unit(torch.randn(N_PAIRS, d, generator=gen, dtype=DT))
    k_corr = unit(torch.randn(N_PAIRS, d, generator=gen, dtype=DT))
    w = torch.randn(N_PAIRS, d, generator=gen, dtype=DT)
    w = unit(w - (w * k_corr).sum(-1, keepdim=True) * k_corr)
    q_corr = RHO * k_corr + math.sqrt(1 - RHO ** 2) * w
    return k_rand, q_rand, k_corr, q_corr


def ip_estimate(quantizer, k, q):
    """압축한 키를 복원해 쿼리와의 내적을 추정 (prod도 복원 벡터와의 내적 = 논문의 추정식)."""
    k_hat = quantizer.dequantize(quantizer.quantize(k))
    return (k_hat * q).sum(-1)


def ci95(v):
    v = np.asarray(v)
    m, s = v.mean(), v.std(ddof=1)
    h = T_CRIT * s / math.sqrt(len(v))
    return float(m), float(m - h), float(m + h)


def run():
    preds = {(p["d"], p["bits"]): p for p in
             json.load(open("results/step0_predictions.json", encoding="utf-8"))["predictions"]}
    per_seed = {}            # (d, b, method) -> {"var_x_d": [...], "mse_x_d": [...], "bias_rel": [...]}
    hist = {}                # 그림 1용: d=128 상관 쌍의 오차 (시드 30개 전부)
    fixed_err = {}           # (d, b, method) -> [시드][쌍] 오차 (고정 수직 쌍)
    for d in DIMS:
        gfix = torch.Generator().manual_seed(SEED_BASE - d)       # 시드와 무관하게 고정
        k_f = unit(torch.randn(N_FIX, d, generator=gfix, dtype=DT))
        w = torch.randn(N_FIX, d, generator=gfix, dtype=DT)
        q_f = unit(w - (w * k_f).sum(-1, keepdim=True) * k_f)       # 키와 정확히 수직
        for s in range(N_SEED):
            gen = torch.Generator().manual_seed(SEED_BASE + 1000 * d + s)
            Pi = random_rotation(d, gen, DT)
            S = torch.randn(d, d, generator=gen, dtype=DT)
            k_r, q_r, k_c, q_c = make_data(d, gen)
            true_r = (k_r * q_r).sum(-1)
            for b in BITS:
                for method, Q in (("mse", TurboQuantMSE(d, b, rotation=Pi)),
                                  ("prod", TurboQuantProd(d, b, rotation=Pi, qjl=S))):
                    e_r = ip_estimate(Q, k_r, q_r) - true_r
                    e_c = ip_estimate(Q, k_c, q_c) - RHO
                    rec = per_seed.setdefault((d, b, method), {"var_x_d": [], "mse_x_d": [], "bias_rel": []})
                    rec["var_x_d"].append(float(e_r.var(unbiased=True) * d))
                    rec["mse_x_d"].append(float((e_r ** 2).mean() * d))
                    rec["bias_rel"].append(float(e_c.mean() / RHO))
                    fixed_err.setdefault((d, b, method), []).append(ip_estimate(Q, k_f, q_f).numpy())
                    if d == 128:
                        hist.setdefault((b, method), []).append(e_c.numpy())

    rows = []
    for d in DIMS:
        for b in BITS:
            p = preds[(d, b)]
            m, pr = per_seed[(d, b, "mse")], per_seed[(d, b, "prod")]
            row = {"d": d, "bits": b}
            for method, rec in (("mse", m), ("prod", pr)):
                for key in ("var_x_d", "mse_x_d", "bias_rel"):
                    mean, lo, hi = ci95(rec[key])
                    row[f"{method}_{key}"] = {"mean": mean, "ci95": [lo, hi]}
                # 고정 수직 쌍: 쌍마다 시드 30개에 걸친 분산 → 쌍 1,000개 평균 (표준오차는 쌍 기준)
                E = np.stack(fixed_err[(d, b, method)])            # (시드, 쌍)
                per_pair = E.var(axis=0, ddof=1) * d
                row[f"{method}_spread_x_d"] = {"mean": float(per_pair.mean()),
                                               "se": float(per_pair.std(ddof=1) / math.sqrt(N_FIX))}
            # 예측표와 비교 (±10%)
            #  - 분산 예측(mse D(1−D), prod (π/2)D_(b−1))은 '편향을 뺀 흩어짐' → 고정 수직 쌍과 비교
            #  - 예측표의 mse 평균제곱오차 D는 무작위 쌍 전체 오차와 비교
            row["pred"] = {"mse_bias_rel": p["mse_bias_rel"], "mse_var_x_d": p["mse_var_x_d"],
                           "prod_var_x_d": p["prod_var_x_d"], "mse_total_mse_x_d": p["mse_total_mse_x_d"]}
            row["pred_check"] = {
                "mse_bias_rel": row["mse_bias_rel"]["mean"] / p["mse_bias_rel"] - 1,
                "mse_spread_vs_D(1-D)": row["mse_spread_x_d"]["mean"] / p["mse_var_x_d"] - 1,
                "prod_spread_vs_pi/2*D(b-1)": row["prod_spread_x_d"]["mean"] / p["prod_var_x_d"] - 1,
                "mse_random_pair_mse_vs_D": row["mse_mse_x_d"]["mean"] / p["mse_total_mse_x_d"] - 1,
                "prod_random_pair_mse_vs_pi/2*D(b-1)": row["prod_mse_x_d"]["mean"] / p["prod_var_x_d"] - 1,
            }
            # 처음(사전) 비교 방식: 무작위 쌍 분산을 D(1−D)와 비교 → 기록으로 남김
            row["initial_comparison_mse_random_var_vs_D(1-D)"] = row["mse_var_x_d"]["mean"] / p["mse_var_x_d"] - 1
            row["pred_within_10pct"] = {k: abs(v) <= 0.10 for k, v in row["pred_check"].items()}
            # 방식 비교: 분산 (prod − mse), 대응표본
            diff = np.array(pr["var_x_d"]) - np.array(m["var_x_d"])
            t = stats.ttest_rel(pr["var_x_d"], m["var_x_d"])
            w = stats.wilcoxon(pr["var_x_d"], m["var_x_d"])
            dm, dlo, dhi = ci95(diff)
            row["var_compare"] = {
                "diff_prod_minus_mse": {"mean": dm, "ci95": [dlo, dhi]},
                "ratio_prod_over_mse": float(np.mean(pr["var_x_d"]) / np.mean(m["var_x_d"])),
                "t": float(t.statistic), "p_t": float(t.pvalue), "p_wilcoxon": float(w.pvalue),
                "d_z": float(diff.mean() / diff.std(ddof=1)),
                "significant_bonferroni": bool(t.pvalue < ALPHA / N_COND and w.pvalue < ALPHA / N_COND),
            }
            # 편향: 각 방식의 상대 편향이 0과 다른가 (단일표본 t검정)
            for method, rec in (("mse", m), ("prod", pr)):
                tt = stats.ttest_1samp(rec["bias_rel"], 0.0)
                row[f"{method}_bias_test"] = {"t": float(tt.statistic), "p": float(tt.pvalue),
                                              "zero_in_ci": row[f"{method}_bias_rel"]["ci95"][0] <= 0 <= row[f"{method}_bias_rel"]["ci95"][1]}
            rows.append(row)
    return rows, per_seed, hist


def fig1_hist(hist):
    """그림 1. 내적 오차 분포 겹치기 (d=128, 실제 내적 0.5)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), dpi=200, sharey=False)
    for ax, b in zip(axes, BITS):
        em, ep = np.concatenate(hist[(b, "mse")]), np.concatenate(hist[(b, "prod")])
        lim = np.percentile(np.abs(ep), 99.5)
        bins = np.linspace(-lim, lim, 90)
        ax.hist(ep, bins=bins, density=True, color=C_PROD, alpha=0.45, label="prod")
        ax.hist(em, bins=bins, density=True, color=C_MSE, alpha=0.55, label="mse")
        ax.axvline(0, color=INK, lw=1.2)
        ax.axvline(em.mean(), color=C_MSE, lw=1.6, ls="--")
        ax.axvline(ep.mean(), color=C_PROD, lw=1.6, ls="--")
        ax.set_title(f"b = {b}비트", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("내적 오차 (추정 − 실제)")
        ax.text(0.98, 0.95, f"mse  평균 {em.mean():+.4f}\n        표준편차 {em.std():.4f}\n"
                            f"prod 평균 {ep.mean():+.4f}\n        표준편차 {ep.std():.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2,
                family=matplotlib.rcParams["font.family"])
    axes[0].set_ylabel("확률밀도")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("그림 1. 편향 vs 분산: 내적 오차 분포 (d=128, 실제 내적 0.5, 시드 30개 × 4,000쌍)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig("figures/fig1_inner_error_hist.png")
    plt.close(fig)


def fig1b_pred_vs_meas(rows):
    """그림 1-b. 예측값 vs 실측값 (오차 분산 × d, 로그 눈금)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), dpi=200, sharey=True)
    for ax, d in zip(axes, DIMS):
        for method, col, off in (("mse", C_MSE, -0.08), ("prod", C_PROD, 0.08)):
            sel = [r for r in rows if r["d"] == d]
            xs = np.array([r["bits"] for r in sel]) + off
            mean = np.array([r[f"{method}_spread_x_d"]["mean"] for r in sel])
            err = 1.96 * np.array([r[f"{method}_spread_x_d"]["se"] for r in sel])
            pred = np.array([r["pred"][f"{method}_var_x_d"] for r in sel])
            ax.errorbar(xs, mean, yerr=err, fmt="o", ms=7, color=col, capsize=3,
                        lw=1.5, label=f"{method} 실측 (95% CI)")
            ax.scatter(xs, pred, s=90, facecolors="none", edgecolors=col, linewidths=1.5,
                       label=f"{method} 예측")
        ax.set_yscale("log")
        ax.set_xticks(BITS)
        ax.set_xlabel("비트폭 b")
        ax.set_title(f"d = {d}", loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("흩어짐: 오차 분산 × d")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("그림 1-b. 예측표 vs 실측: 편향을 뺀 흩어짐 (고정 수직 쌍 1,000개, 시드 30개)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig("figures/fig1b_pred_vs_measured.png")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t0 = time.time()
    rows, per_seed, hist = run()
    fig1_hist(hist)
    fig1b_pred_vs_meas(rows)
    out = {
        "experiment": "1단계 내적 실험",
        "design": {"dims": DIMS, "bits": BITS, "n_seed": N_SEED, "n_pairs_per_seed": N_PAIRS, "rho": RHO,
                   "seed_rule": f"{SEED_BASE} + 1000*d + s", "alpha": ALPHA,
                   "bonferroni_alpha": ALPHA / N_COND, "t_crit_df29": float(T_CRIT)},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__},
        "rows": rows,
        "per_seed": {f"d{d}_b{b}_{m}": v for (d, b, m), v in per_seed.items()},
        "runtime_sec": time.time() - t0,
    }
    with open("results/exp1_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"완료: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
