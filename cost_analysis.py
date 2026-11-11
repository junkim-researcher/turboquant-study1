"""
비용 측정 (계산형): 저장 크기, 연산 수, 메모리 읽기량

정확도 비교(0~4단계)에 이어, 두 방식이 그 정확도를 얻는 데 드는 '자원'을 비교한다.
실제 시간은 재지 않는다(구현과 장비에 따라 달라지므로). 대신
  ① 저장 크기: 식으로 계산하고, 실제로 비트를 묶어 바이트 수를 재서 식과 맞는지 확인
  ② 연산 수:  식으로 계산하고, 곱셈과 덧셈을 하나하나 세는 '세기용 구현'을 작은 크기로 돌려
              (a) 센 횟수가 식과 정확히 같은지, (b) 계산 결과가 실제 구현(tq/)과 같은지 확인
  ③ 메모리 읽기량: 새 토큰 하나를 만들 때 KV 캐시 전체를 한 번 읽으므로 = KV 캐시 크기

기준: Qwen2.5-0.5B (24층, 쿼리 헤드 14개, 키/값 헤드 2개, 헤드 차원 64), 문맥 n = 4,096토큰.

점수(쿼리와 키의 내적) 계산 방식 두 가지
  [복원 후 계산] 키마다 압축을 풀어(역회전 Πᵀ) 원래 공간에서 쿼리와 내적
  [쿼리 회전 + 표] ⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩ 이므로 쿼리를 한 번만 회전하고,
                  표 T[j][c] = (Πq)_j × 눈금값_c 를 만들어 두면 키마다 '표에서 꺼내 더하기'만 하면 된다
                  (원래 계획서의 H1 쿼리 쪽 회전 + H2 표(LUT) 아이디어)
  prod의 QJL 부분은 z가 +1 또는 −1이라서 Sᵀz나 ⟨Sq, z⟩에 곱셈이 필요 없다(부호를 바꿔 더하기만).

실행: python cost_analysis.py   (CPU, 몇 초)
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

from tq import TurboQuantMSE, TurboQuantProd, random_rotation
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style
from exp0_verify import pack_bits

apply_style()
C_FP16 = "#52514e"
QWEN = {"n_layers": 24, "n_heads": 14, "n_kv_heads": 2, "head_dim": 64}
N_CTX = 4096
BITS = (2, 3, 4)


# =============================================================== ① 저장 크기
def bits_per_vector(kind, d, b):
    return {"fp16": 16 * d, "mse": d * b + 16, "prod": d * b + 32}[kind]


def kv_cache_bytes(key_kind, b, m=QWEN, n=N_CTX, centered=False):
    """키는 key_kind, 값은 (fp16이면 fp16, 아니면 mse b비트)."""
    d = m["head_dim"]
    val_kind = "fp16" if key_kind == "fp16" else "mse"
    per_token = bits_per_vector(key_kind, d, b) + bits_per_vector(val_kind, d, b)
    total_bits = m["n_layers"] * m["n_kv_heads"] * n * per_token
    if centered:
        total_bits += m["n_layers"] * m["n_kv_heads"] * d * 16      # 평균 c: 층, 헤드마다 FP16 64개
    return total_bits / 8


def measure_packed_bytes(d=64, n=4096, b=3, seed=0):
    """실제로 압축하고 비트를 묶어서 잰 바이트 수 (키/값 헤드 하나 분량)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64) * 7.0
    out = {}
    qm = TurboQuantMSE(d, b, seed=seed)
    cm = qm.quantize(x)
    out["mse"] = len(pack_bits(cm["idx"].numpy(), b)) + 2 * n                       # 번호 + 노름(FP16)
    qp = TurboQuantProd(d, b, seed=seed)
    cp = qp.quantize(x)
    z01 = (cp["z"].numpy() > 0).astype(np.uint8)
    out["prod"] = len(pack_bits(cp["idx"].numpy(), b - 1)) + len(np.packbits(z01.ravel())) + 2 * 2 * n
    out["fp16"] = x.to(torch.float16).numpy().nbytes
    return out


# =============================================================== ② 연산 수 (식)
def score_ops_formula(kind, method, d, n, b):
    """쿼리 헤드 하나가 키 n개와의 점수를 구할 때 (곱셈, 덧셈) 횟수."""
    if kind == "fp16":
        return n * d, n * (d - 1)
    if method == "dequant":                                   # 복원 후 계산
        if kind == "mse":
            mul = n * (d * d + d + d)                         # Πᵀỹ (d²) + 노름 곱 (d) + 내적 (d)
            add = n * (d * (d - 1) + (d - 1))
        else:                                                 # prod: mse(b−1) 복원 + QJL 복원 + 합 + 노름 + 내적
            mse_mul = d * d if b - 1 > 0 else 0
            mse_add = d * (d - 1) if b - 1 > 0 else 0
            mul = n * (mse_mul + d + d + d)                   # + (계수×γ)·Sᵀz (d) + 노름 곱 (d) + 내적 (d)
            two_part = d if b - 1 > 0 else 0                  # 두 부분 합 (b=1이면 mse 부분이 없음)
            add = n * (mse_add + d * (d - 1) + two_part + (d - 1))   # + Sᵀz (부호 더하기) + 내적
        return mul, add
    # 쿼리 회전 + 표
    if kind == "mse":
        mul = d * d + d * 2 ** b + n                          # Πq + 표 만들기 + 키마다 노름 곱
        add = d * (d - 1) + n * (d - 1)                       # Πq + 키마다 표에서 꺼내 더하기
        return mul, add
    mul = d * d + d                                           # Sq + 계수 접어 넣기
    add = d * (d - 1) + n * (d - 1)                           # Sq + 키마다 부호 더하기
    if b - 1 > 0:
        mul += d * d + d * 2 ** (b - 1)
        add += d * (d - 1) + n * (d - 1)
    mul += n * 2                                              # 키마다 γ 곱, 노름 곱
    add += n * (1 if b - 1 > 0 else 0)                        # 두 부분 합
    return mul, add


def compress_ops_formula(kind, d, b):
    """새 키 하나를 압축할 때 (곱셈, 덧셈, 비교) 횟수."""
    if kind == "mse":
        return d + d * d, (d - 1) + d * (d - 1), d * b           # 노름², 회전, 눈금 찾기(이진 탐색)
    mul = d + d * d + (d * d if b - 1 > 0 else 0) + d + d * d     # 노름², 회전, 복원(잔차용), 잔차 노름², Sr
    add = (d - 1) + d * (d - 1) + (d * (d - 1) if b - 1 > 0 else 0) + d + (d - 1) + d * (d - 1)
    cmp = d * (b - 1) + d                                         # 눈금 찾기 + 부호
    return mul, add, cmp


# =============================================================== ② 연산 수 (직접 세기)
class Counter:
    def __init__(self):
        self.mul = self.add = 0

    def m(self, a, b):
        self.mul += 1
        return a * b

    def a(self, x, y):
        self.add += 1
        return x + y

    def dot(self, u, v):
        s = self.m(u[0], v[0])
        for i in range(1, len(u)):
            s = self.a(s, self.m(u[i], v[i]))
        return s

    def sum(self, xs):
        s = xs[0]
        for x in xs[1:]:
            s = self.a(s, x)
        return s

    def matvec(self, M, v):                       # M: 리스트의 리스트 (d×d)
        return [self.dot(row, v) for row in M]


def counted_scores(kind, method, q, codes, Q):
    """세기용 구현: 파이썬 반복문으로 하나씩 계산하며 곱셈과 덧셈을 센다."""
    C = Counter()
    q = q.tolist()
    d = len(q)
    if kind == "mse":
        Pi = Q.Pi.tolist(); PiT = Q.Pi.T.tolist(); cent = Q.centroids.tolist()
        idx = codes["idx"].tolist(); nrm = codes["norm"][:, 0].tolist()
        if method == "dequant":
            out = []
            for i in range(len(idx)):
                y = [cent[c] for c in idx[i]]
                u = C.matvec(PiT, y)
                k = [C.m(nrm[i], t) for t in u]
                out.append(C.dot(q, k))
            return out, C
        qr = C.matvec(Pi, q)
        T = [[C.m(qr[j], cent[c]) for c in range(len(cent))] for j in range(d)]
        return [C.m(nrm[i], C.sum([T[j][idx[i][j]] for j in range(d)])) for i in range(len(idx))], C
    # prod
    mq = Q.mse
    Pi = mq.Pi.tolist(); PiT = mq.Pi.T.tolist(); cent = mq.centroids.tolist()
    S = Q.S.tolist(); ST = Q.S.T.tolist()
    idx = codes["idx"].tolist(); z = codes["z"].tolist()
    gam = codes["gamma"][:, 0].tolist(); nrm = codes["norm"][:, 0].tolist()
    coef = Q.coef
    has_mse = Q.bits - 1 > 0

    def signed_sum(vals, signs):                  # 부호만 바꿔 더하기 (곱셈 없음)
        s = vals[0] if signs[0] > 0 else -vals[0]
        for v_, s_ in zip(vals[1:], signs[1:]):
            s = C.a(s, v_ if s_ > 0 else -v_)
        return s

    if method == "dequant":
        out = []
        for i in range(len(idx)):
            u_m = C.matvec(PiT, [cent[c] for c in idx[i]]) if has_mse else None
            stz = [signed_sum(col, z[i]) for col in ST]           # Sᵀz: 각 성분 = Σ_j z_j S[j][·]
            cg = coef * gam[i]                                     # 키마다 상수 (저장 시 미리 계산 가능)
            u_q = [C.m(cg, t) for t in stz]
            u = [C.a(a_, b_) for a_, b_ in zip(u_m, u_q)] if has_mse else u_q
            k = [C.m(nrm[i], t) for t in u]
            out.append(C.dot(q, k))
        return out, C
    sq = C.matvec(S, q)
    sq = [C.m(coef, t) for t in sq]                                # 계수를 쿼리 쪽에 접어 넣기
    if has_mse:
        qr = C.matvec(Pi, q)
        T = [[C.m(qr[j], cent[c]) for c in range(len(cent))] for j in range(d)]
    out = []
    for i in range(len(idx)):
        t = C.m(gam[i], signed_sum(sq, z[i]))                      # γ × (부호 더하기)
        if has_mse:
            t = C.a(C.sum([T[j][idx[i][j]] for j in range(d)]), t)  # 표에서 꺼내 더하기 + 두 부분 합
        out.append(C.m(nrm[i], t))                                  # 노름은 마지막에 한 번만
    return out, C


def verify_counts(d=8, n=6, seed=0):
    """작은 크기에서: 센 횟수 == 식, 계산 결과 == 실제 구현(복원 후 내적)."""
    rows = []
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, d, generator=g, dtype=torch.float64) * 3
    q = torch.randn(d, generator=g, dtype=torch.float64)
    for kind in ("mse", "prod"):
        for b in (1, 2, 3, 4) if kind == "prod" else (1, 2, 3, 4):
            Q = (TurboQuantMSE if kind == "mse" else TurboQuantProd)(d, b, seed=seed + b)
            codes = Q.quantize(keys)
            ref = (Q.dequantize(codes) @ q).numpy()
            for method in ("dequant", "rotate_lut"):
                vals, C = counted_scores(kind, method, q, codes, Q)
                fm, fa = score_ops_formula(kind, method, d, n, b)
                rows.append({"kind": kind, "bits": b, "method": method,
                             "counted_mul": C.mul, "formula_mul": fm, "counted_add": C.add, "formula_add": fa,
                             "count_match": C.mul == fm and C.add == fa,
                             "max_abs_diff_vs_tq": float(np.max(np.abs(np.array(vals) - ref)))})
    return rows


# =============================================================== 표와 그림
def per_token_tables(m=QWEN, n=N_CTX):
    d, L, H, Hkv = m["head_dim"], m["n_layers"], m["n_heads"], m["n_kv_heads"]
    rows = []
    for b in BITS:
        for kind in ("fp16", "mse", "prod"):
            for method in (("-",) if kind == "fp16" else ("dequant", "rotate_lut")):
                mul, add = score_ops_formula(kind, method, d, n, b)
                if method == "dequant" and kind != "fp16":
                    # 복원은 키/값 헤드마다 한 번이면 되고(쿼리 헤드 7개가 공유), 내적만 쿼리 헤드마다
                    dq_mul, dq_add = mul - n * d, add - n * (d - 1)
                    tot_mul = L * (Hkv * dq_mul + H * n * d)
                    tot_add = L * (Hkv * dq_add + H * n * (d - 1))
                else:
                    tot_mul, tot_add = L * H * mul, L * H * add
                rows.append({"bits": b, "key": kind, "method": method,
                             "mul_per_token": tot_mul, "add_per_token": tot_add})
    comp = []
    for b in BITS:
        for kind in ("mse", "prod"):
            mu, ad, cm = compress_ops_formula(kind, d, b)
            comp.append({"bits": b, "key": kind, "mul_per_token": L * Hkv * mu, "add_per_token": L * Hkv * ad,
                         "cmp_per_token": L * Hkv * cm})
    return rows, comp


def make_figure(storage, score_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=200)
    ax = axes[0]
    fp = storage["fp16"]["MB"]
    ax.axhline(fp, color=C_FP16, lw=1.2, ls="--")
    ax.text(4.45, fp, f"FP16\n{fp:.1f} MB", va="center", fontsize=8.5, color=INK2)
    for kind, col, off in (("mse", C_MSE, -0.16), ("prod", C_PROD, 0.16)):
        vals = [storage[f"{kind}_b{b}"]["MB"] for b in BITS]
        bars = ax.bar(np.array(BITS) + off, vals, width=0.3, color=col, label=f"키 {kind} + 값 mse")
        for x_, v_ in zip(np.array(BITS) + off, vals):
            ax.text(x_, v_ + 0.6, f"{v_:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(BITS); ax.set_xlabel("비트폭 b"); ax.set_ylabel("KV 캐시 크기 (MB)")
    ax.set_xlim(1.5, 4.9); ax.set_ylim(0, fp * 1.12)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    ax.set_title(f"(가) KV 캐시 크기 = 새 토큰마다 읽는 양 (n = {N_CTX:,})", loc="left", fontsize=11, color=INK)

    ax = axes[1]
    b = 3
    sel = {(r["key"], r["method"]): r for r in score_rows if r["bits"] == b}
    labels = ["FP16\n(압축 없음)", "mse\n복원 후 계산", "prod\n복원 후 계산", "mse\n쿼리 회전+표", "prod\n쿼리 회전+표"]
    keys = [("fp16", "-"), ("mse", "dequant"), ("prod", "dequant"), ("mse", "rotate_lut"), ("prod", "rotate_lut")]
    cols = [C_FP16, C_MSE, C_PROD, C_MSE, C_PROD]
    vals = [sel[k]["mul_per_token"] / 1e6 for k in keys]
    hatch = [None, "//", "//", None, None]
    for i, (v_, c_, h_) in enumerate(zip(vals, cols, hatch)):
        ax.bar(i, v_, color=c_, hatch=h_, edgecolor="white" if h_ else c_, width=0.7)
        ax.text(i, v_ * 1.08, f"{v_:,.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1, 2000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.4g}"))
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("곱셈 횟수 (백만 회, 로그 눈금)")
    ax.set_title(f"(나) 새 토큰 하나의 키 점수 계산에 드는 곱셈 (b = {b})", loc="left", fontsize=11, color=INK)
    fig.suptitle("그림 8. 비용 비교 (Qwen2.5-0.5B, 24층 전체, 식으로 계산하고 직접 세어 검증)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig("figures/fig8_cost.png"); plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    d = QWEN["head_dim"]

    # ① 저장 크기
    storage = {"fp16": {"bytes": kv_cache_bytes("fp16", 16)}}
    for b in BITS:
        for kind in ("mse", "prod"):
            storage[f"{kind}_b{b}"] = {"bytes": kv_cache_bytes(kind, b)}
        storage[f"mse_centered_b{b}"] = {"bytes": kv_cache_bytes("mse", b, centered=True)}
    for k, v in storage.items():
        v["MB"] = v["bytes"] / 2 ** 20
        v["ratio_vs_fp16"] = storage["fp16"]["bytes"] / v["bytes"]
    packed = {b: measure_packed_bytes(d=d, n=N_CTX, b=b) for b in BITS}
    pack_check = []
    for b in BITS:
        for kind in ("fp16", "mse", "prod"):
            formula = N_CTX * bits_per_vector(kind, d, b) / 8
            pack_check.append({"bits": b, "kind": kind, "formula_bytes": formula, "packed_bytes": packed[b][kind],
                               "match": formula == packed[b][kind]})

    # ② 연산 수
    verify = verify_counts()
    score_rows, comp_rows = per_token_tables()
    make_figure(storage, score_rows)

    out = {"model_dims": QWEN, "n_ctx": N_CTX, "storage": storage, "pack_check_one_head": pack_check,
           "op_count_verification_small": verify, "score_ops_per_token": score_rows,
           "compress_ops_per_token": comp_rows,
           "note": "연산 수는 식으로 계산했고, d=8, n=6에서 세기용 구현으로 직접 세어 식과 일치함을 확인. 실제 시간은 재지 않음."}
    with open("results/cost_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 출력
    print("=" * 84)
    print(f" 비용 측정 (Qwen2.5-0.5B, n = {N_CTX:,}토큰)")
    print("=" * 84)
    print(" [①] KV 캐시 크기 (= 새 토큰 하나를 만들 때 메모리에서 읽는 양)")
    print(f"   FP16 (압축 없음)       {storage['fp16']['MB']:7.2f} MB")
    for b in BITS:
        m_, p_, c_ = storage[f"mse_b{b}"], storage[f"prod_b{b}"], storage[f"mse_centered_b{b}"]
        print(f"   b={b}  키 mse + 값  {m_['MB']:6.2f} MB ({m_['ratio_vs_fp16']:.2f}배 작음)   "
              f"키 prod + 값 {p_['MB']:6.2f} MB ({p_['ratio_vs_fp16']:.2f}배)   "
              f"평균 뺀 mse {c_['MB']:6.3f} MB (평균 c 추가분 {(c_['bytes'] - m_['bytes']) / 1024:.1f} KB)")
    print(f"   비트 묶기 실측과 식 일치: {sum(r['match'] for r in pack_check)}/{len(pack_check)}")
    print("-" * 84)
    print(" [②-검증] 작은 크기(d=8, n=6)에서 직접 센 횟수 == 식 ?   계산 결과 == 실제 구현 ?")
    print(f"   횟수 일치 {sum(r['count_match'] for r in verify)}/{len(verify)},  "
          f"결과 최대 차이 {max(r['max_abs_diff_vs_tq'] for r in verify):.1e}")
    print("-" * 84)
    print(" [②] 새 토큰 하나의 키 점수 계산 (24층, 쿼리 헤드 14개 전체)  단위: 백만 회")
    print(f"   {'b':>2} {'방식':<22} {'곱셈':>10} {'덧셈':>10}")
    for r in score_rows:
        name = {"-": "FP16 (압축 없음)", "dequant": "복원 후 계산", "rotate_lut": "쿼리 회전 + 표"}[r["method"]]
        if r["key"] != "fp16":
            name = f"{r['key']} {name}"
        print(f"   {r['bits']:>2} {name:<22} {r['mul_per_token'] / 1e6:10.2f} {r['add_per_token'] / 1e6:10.2f}")
    print("-" * 84)
    print(" [②] 새 토큰 하나의 키 압축 (24층, 키/값 헤드 2개)  단위: 천 회")
    for r in comp_rows:
        print(f"   b={r['bits']} {r['key']:<5} 곱셈 {r['mul_per_token'] / 1e3:7.1f}   덧셈 {r['add_per_token'] / 1e3:7.1f}   "
              f"비교 {r['cmp_per_token'] / 1e3:5.1f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
