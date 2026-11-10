"""
0단계(E0). 재현 검증

직접 구현한 TurboQuant(mse, prod)가 논문의 알고리즘과 이론값을 재현하는지 확인한다.
판정 기준(원래 계획서 E0 기준을 그대로 따름):
  [A] 회전 후 좌표 분포가 이론 분포(베타 분포)와 맞는가  → 히스토그램 + KS 검정
  [B] QR 부호 보정 여부에 따라 회전행렬과 MSE가 달라지는가
  [C] 실측 MSE가 직접 계산한 이론값의 ±5% 이내인가
  [D] 비편향성: 고정된 벡터-쿼리 쌍에 대해 시드 200개로 반복했을 때
        prod는 평균 오차 ± 2×표준오차 범위에 0이 들어가는가 (불편)
        mse(b=1)은 추정값이 실제 내적의 약 2/π배로 치우치는가
  [E] 실제 저장 크기가 메모리 계산식과 맞는가 (비트 묶기 후 바이트 수 측정)

모든 계산은 64비트 실수. 사용한 시드는 결과 JSON에 기록한다.
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

from tq import TurboQuantMSE, TurboQuantProd, lloyd_max, random_rotation, coord_pdf
from tq.codebook import _cdf

DT = torch.float64
DIMS = (64, 128)
BITS = (1, 2, 3, 4)
PAPER_D = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style

apply_style()


def unit_vectors(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=DT)
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------- [A] 좌표 분포
def check_A(d=128, n_vec=4000, n_seed=10):
    coords = []
    for s in range(n_seed):
        Pi = random_rotation(d, torch.Generator().manual_seed(1000 + s), DT)
        u = unit_vectors(n_vec, d, 2000 + s)
        coords.append((u @ Pi.T).flatten().numpy())
    y = np.concatenate(coords)
    ks_beta = stats.kstest(y, lambda t: _cdf(t, d))
    ks_norm = stats.kstest(y, lambda t: stats.norm.cdf(t, 0, 1 / math.sqrt(d)))

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    ax.hist(y * math.sqrt(d), bins=120, density=True, color=C_MSE, alpha=0.35,
            label=f"실측: 회전 후 좌표 × √d  (좌표 {len(y):,}개)")
    t = np.linspace(-4.2, 4.2, 600)
    ax.plot(t, coord_pdf(t / math.sqrt(d), d) / math.sqrt(d), color=INK, lw=2, label=f"이론: 베타 분포 (d={d})")
    ax.plot(t, stats.norm.pdf(t), color=C_PROD, lw=1.6, ls="--", label="참고: 표준정규분포 근사")
    ax.set_xlabel("좌표값 × √d")
    ax.set_ylabel("확률밀도")
    ax.set_title(f"[A] 무작위 회전 후 좌표 분포 (d={d})", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/e0_A_coord_hist.png")
    plt.close(fig)
    return {"d": d, "n_coords": int(len(y)),
            "mean": float(y.mean()), "var_x_d": float(y.var() * d),
            "ks_vs_beta": {"stat": float(ks_beta.statistic), "p": float(ks_beta.pvalue)},
            "ks_vs_normal": {"stat": float(ks_norm.statistic), "p": float(ks_norm.pvalue)}}


# ---------------------------------------------------------------- [B] QR 부호 보정
def check_B(d=128, n_seed=300, bits=2, n_vec=2000):
    res = {}
    for fix in (True, False):
        diag_pos, first_entry, mses = [], [], []
        for s in range(n_seed):
            Pi = random_rotation(d, torch.Generator().manual_seed(5000 + s), DT, sign_fix=fix)
            diag_pos.append(float((torch.diagonal(Pi) > 0).double().mean()))
            first_entry.append(float(Pi[0, 0]))
            if s < 30:
                q = TurboQuantMSE(d, bits, rotation=Pi)
                u = unit_vectors(n_vec, d, 6000 + s)
                mses.append(float(((q.dequantize_unit(q.quantize_unit(u)) - u) ** 2).sum(-1).mean()))
        res["with_fix" if fix else "without_fix"] = {
            "diag_positive_ratio": float(np.mean(diag_pos)),
            "Pi00_mean": float(np.mean(first_entry)),
            "Pi00_se": float(np.std(first_entry, ddof=1) / math.sqrt(n_seed)),
            f"mse_b{bits}": float(np.mean(mses)),
            f"mse_b{bits}_se": float(np.std(mses, ddof=1) / math.sqrt(len(mses))),
        }
    return res


# ---------------------------------------------------------------- [C] MSE 재현
def check_C(n_seed=20, n_vec=2000):
    rows = []
    for d in DIMS:
        for b in BITS:
            D_th = lloyd_max(d, b)[2]
            per_seed = []
            for s in range(n_seed):
                q = TurboQuantMSE(d, b, seed=10_000 + s, dtype=DT)
                u = unit_vectors(n_vec, d, 20_000 + s)
                per_seed.append(float(((q.dequantize(q.quantize(u)) - u) ** 2).sum(-1).mean()))
            m, se = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1) / math.sqrt(n_seed))
            rows.append({"d": d, "bits": b, "mse_measured": m, "mse_se": se,
                         "D_theory_exact": D_th, "D_paper": PAPER_D[b],
                         "rel_diff_vs_theory": m / D_th - 1,
                         "pass_5pct": abs(m / D_th - 1) <= 0.05})
    return rows


# ---------------------------------------------------------------- [D] 비편향성
def make_pair(d, rho, seed):
    """길이 1인 키 x와 쿼리 q, 실제 내적이 rho가 되도록."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(d, generator=g, dtype=DT); x /= x.norm()
    w = torch.randn(d, generator=g, dtype=DT); w -= (w @ x) * x; w /= w.norm()
    q = rho * x + math.sqrt(1 - rho ** 2) * w
    return x, q


def check_D(K=200, rhos=(0.1, 0.5, 0.9)):
    rows, raw = [], {}
    for d in DIMS:
        pairs = {rho: make_pair(d, rho, 777 + int(rho * 100)) for rho in rhos}
        est = {(m, b, rho): [] for m in ("mse", "prod") for b in BITS for rho in rhos}
        for k in range(K):
            g = torch.Generator().manual_seed(90_000 + k)
            Pi = random_rotation(d, g, DT)
            S = torch.randn(d, d, generator=g, dtype=DT)
            for b in BITS:
                qm = TurboQuantMSE(d, b, rotation=Pi)
                qp = TurboQuantProd(d, b, rotation=Pi, qjl=S)
                for rho, (x, q) in pairs.items():
                    est[("mse", b, rho)].append(float(q @ qm.dequantize(qm.quantize(x))))
                    est[("prod", b, rho)].append(float(q @ qp.dequantize(qp.quantize(x))))
        for (m, b, rho), v in est.items():
            v = np.array(v)
            err = v - rho
            mean, se = float(err.mean()), float(err.std(ddof=1) / math.sqrt(K))
            D_b = lloyd_max(d, b)[2]
            rows.append({"d": d, "method": m, "bits": b, "true_ip": rho,
                         "mean_err": mean, "se": se, "z": mean / se,
                         "zero_in_2se": abs(mean) <= 2 * se,
                         "ratio_est_over_true": float(v.mean() / rho),
                         "ratio_theory_mse": 1 - D_b if m == "mse" else 1.0,
                         "var_x_d": float(err.var(ddof=1) * d)})
            raw[f"d{d}_{m}_b{b}_rho{rho}"] = v.tolist()
    return rows, raw


# ---------------------------------------------------------------- [E] 저장 크기
def pack_bits(idx: np.ndarray, bits: int) -> bytes:
    """정수 번호들을 bits비트씩 이어 붙여 바이트로 묶는다."""
    if bits == 0:
        return b""
    flat = idx.astype(np.uint8).ravel()
    bitplanes = ((flat[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.uint8)
    return np.packbits(bitplanes.ravel()).tobytes()


def check_E(n_vec=4096):
    rows = []
    for d in DIMS:
        u = unit_vectors(n_vec, d, 31337)
        for b in (2, 3, 4):
            qm, qp = TurboQuantMSE(d, b, seed=1), TurboQuantProd(d, b, seed=1)
            cm, cp = qm.quantize(u), qp.quantize(u)
            m_bytes = len(pack_bits(cm["idx"].numpy(), b)) + 2 * n_vec
            z01 = (cp["z"].numpy() > 0).astype(np.uint8)
            p_bytes = len(pack_bits(cp["idx"].numpy(), b - 1)) + len(np.packbits(z01.ravel())) + 2 * 2 * n_vec
            rows.append({"d": d, "bits": b,
                         "mse_bits_per_vec_formula": qm.bits_per_vector(),
                         "mse_bits_per_vec_packed": m_bytes * 8 / n_vec,
                         "prod_bits_per_vec_formula": qp.bits_per_vector(),
                         "prod_bits_per_vec_packed": p_bytes * 8 / n_vec,
                         "fp16_bits_per_vec": 16 * d,
                         "compress_ratio_mse": 16 * d / qm.bits_per_vector(),
                         "compress_ratio_prod": 16 * d / qp.bits_per_vector()})
    return rows


def plot_D(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), dpi=200, sharey=True)
    for ax, d in zip(axes, DIMS):
        for m, col, off in (("mse", C_MSE, -0.12), ("prod", C_PROD, 0.12)):
            sel = [r for r in rows if r["d"] == d and r["method"] == m and r["true_ip"] == 0.5]
            xb = [r["bits"] + off for r in sel]
            ax.errorbar(xb, [r["mean_err"] for r in sel], yerr=[2 * r["se"] for r in sel],
                        fmt="o", ms=7, color=col, capsize=3, lw=1.6, label=m)
        th = [-(lloyd_max(d, b)[2]) * 0.5 for b in BITS]
        ax.plot(BITS, th, color=INK2, lw=1, ls=":", label="mse 이론 편향 (−D×0.5)")
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(BITS)
        ax.set_xlabel("비트폭 b")
        ax.set_title(f"d = {d}", loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("내적 오차 평균 ± 2×표준오차")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("[D] 비편향성 검증 (실제 내적 0.5, 시드 200개)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig("figures/e0_D_unbiasedness.png")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    t0 = time.time()
    A = check_A(); print("[A] done", A)
    B = check_B(); print("[B] done", B)
    C = check_C(); print("[C] done")
    D, raw = check_D(); print("[D] done")
    E = check_E(); print("[E] done")
    plot_D(D)

    out = {
        "experiment": "E0 재현 검증",
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "machine": platform.machine(), "cpu_threads": torch.get_num_threads()},
        "seeds": {"A_rotation": "1000+s (s<10)", "A_vectors": "2000+s", "B_rotation": "5000+s (s<300)",
                  "B_vectors": "6000+s", "C_quantizer": "10000+s (s<20)", "C_vectors": "20000+s",
                  "D_pairs": "777+int(100*rho)", "D_rotation_and_S": "90000+k (k<200)", "E": "31337, quantizer seed 1"},
        "A_coord_distribution": A, "B_qr_sign_fix": B, "C_mse_reproduction": C,
        "D_unbiasedness": D, "E_storage": E, "runtime_sec": time.time() - t0,
    }
    with open("results/exp0_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open("results/exp0_D_raw_estimates.json", "w", encoding="utf-8") as f:
        json.dump(raw, f)

    print("\n[C] MSE 재현")
    for r in C:
        print(f"  d={r['d']:3d} b={r['bits']}  실측 {r['mse_measured']:.5f} ± {r['mse_se']:.5f}  이론 {r['D_theory_exact']:.5f}"
              f"  차이 {r['rel_diff_vs_theory']*100:+.2f}%  {'PASS' if r['pass_5pct'] else 'FAIL'}")
    print("\n[D] 비편향성 (평균오차, z, ±2SE 안에 0?)")
    for r in D:
        print(f"  d={r['d']:3d} {r['method']:4s} b={r['bits']} ρ={r['true_ip']}  평균오차 {r['mean_err']:+.5f}"
              f"  z={r['z']:+6.2f}  0포함={r['zero_in_2se']}  추정/실제={r['ratio_est_over_true']:.4f}"
              f" (mse이론 {r['ratio_theory_mse']:.4f})")
    print("\n[E] 저장 크기")
    for r in E:
        print("  ", r)
    print(f"\n총 실행 시간 {time.time()-t0:.1f}초")


if __name__ == "__main__":
    main()
