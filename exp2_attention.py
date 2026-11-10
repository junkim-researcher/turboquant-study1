"""
2단계. 어텐션 실험 (실제 언어모델의 Q, K, V)

질문: 1단계에서 확인한 성질(mse = 편향 있음 + 흩어짐 작음, prod = 편향 없음 + 흩어짐 큼)이
      softmax를 거친 실제 어텐션에서는 어느 쪽에 유리하게 작용하는가?
      (가설 H-b: b=3, 4에서는 mse가 더 정확, b=2는 뒤집힐 수 있음
       가설 H-c: 문맥 길이 n이 길수록 prod의 top-1 일치율이 mse보다 빨리 떨어짐)

설계
  - 모델: Qwen2.5-0.5B (기본값). 위치 정보(RoPE)가 들어간 뒤의 쿼리, 키, 값을 가로챈다.
  - 데이터: WikiText-2 test에서 겹치지 않는 4,096토큰 구간 4개, 층 3개(앞, 가운데, 뒤), 모든 헤드
  - 상황: 문맥 앞 n개 토큰 중 마지막 64개 쿼리가 자기 앞의 키들을 보는 상황 (= 새 토큰 생성 재현)
  - 독립변수(2범주): 키 압축 방식 mse / prod.  값(value)은 압축하지 않음(키 효과만 분리)
  - 조건: b = 2, 3, 4  ×  n = 1,024, 2,048, 4,096  (9개 조건)
  - 반복 단위: 시드 30개 (시드마다 층별 회전행렬 Π, QJL 행렬 S를 새로 뽑음, 두 방식이 같은 Π 공유)
  - 지표 (원본 FP32 어텐션 대비)
      KL 발산          : softmax 분포가 얼마나 달라졌나 (작을수록 좋음)
      top-1 일치율     : 가장 많이 본 키가 원본과 같은가 (클수록 좋음)
      출력 상대오차    : ||ô − o|| / ||o|| (작을수록 좋음)
      로짓 기울기      : 압축 후 점수를 원본 점수에 회귀한 기울기 (mse는 1 − D 예상, prod는 1 예상)
  - 통계: 시드별 차이(prod − mse) 대응표본 t검정 + 윌콕슨, 본페로니(0.05/9), 95% 신뢰구간(t, 자유도 29)
          H-c: 시드마다 '두 방식의 top-1 차이'를 log2 n에 대해 직선 맞춤한 기울기 → 0보다 큰가 (단일표본 t검정)

실행: python exp2_attention.py            (Colab T4 GPU, 약 5~10분)
      python exp2_attention.py --smoke    (인터넷 없이 무작위 소형 모델로 동작만 확인)
"""
import argparse
import json
import math
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from tq import TurboQuantMSE, TurboQuantProd, random_rotation, lloyd_max
from tq import capture
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style

apply_style()

BITS = (2, 3, 4)
METHODS = ("mse", "prod")
METRICS = ("kl", "top1", "rel_err", "logit_slope")
SEED_BASE = 200_000
ALPHA = 0.05


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--window-len", type=int, default=4096)
    ap.add_argument("--ns", default="1024,2048,4096")
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.windows, a.window_len, a.ns, a.queries = 4, 2, 1024, "256,512,1024", 32
    a.ns = tuple(int(x) for x in a.ns.split(","))
    return a


# ------------------------------------------------------------------ 데이터와 모델
def load_everything(a, device):
    if a.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
        cfg._attn_implementation = capture.IMPL
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg).to(device).eval()
        g = torch.Generator().manual_seed(1)
        windows = [torch.randint(0, 1000, (a.window_len,), generator=g) for _ in range(a.windows)]
        return model, windows, "무작위 토큰 (smoke)"
    from datasets import load_dataset
    tok, model = capture.load_model(a.model, device)
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    starts = np.linspace(0, len(ids) - a.window_len - 1, a.windows).astype(int)
    windows = [ids[s:s + a.window_len] for s in starts]
    return model, windows, f"WikiText-2 test, 전체 {len(ids):,}토큰 중 시작 위치 {starts.tolist()}"


def pick_layers(n_layers):
    return sorted({max(1, round(n_layers * 0.15)), n_layers // 2, n_layers - 2})


@torch.no_grad()
def capture_all(model, windows, layers, device):
    capture.CAPTURE["layers"] = set(layers)
    caps = []
    for w in windows:
        capture.CAPTURE["store"] = {}
        model(w[None].to(device))
        caps.append({li: tuple(t[0] for t in capture.CAPTURE["store"][li]) for li in layers})  # (H,T,d)
    capture.CAPTURE["layers"] = set()
    return caps


# ------------------------------------------------------------------ 지표 계산
def attn_metrics(q, k, v, k_hat, n, n_q, scaling, group):
    """q:(H,T,d) k,v,k_hat:(Hkv,T,d). 문맥 앞 n개 중 마지막 n_q개 쿼리."""
    qq = q[:, n - n_q:n]
    kk = k[:, :n].repeat_interleave(group, 0)
    kh = k_hat[:, :n].repeat_interleave(group, 0)
    vv = v[:, :n].repeat_interleave(group, 0)
    pos = torch.arange(n - n_q, n, device=q.device)[:, None]
    mask = torch.arange(n, device=q.device)[None, :] <= pos            # (n_q, n) 인과 마스크
    l = (qq @ kk.transpose(-1, -2)) * scaling
    lh = (qq @ kh.transpose(-1, -2)) * scaling
    neg = torch.finfo(l.dtype).min
    l, lh = l.masked_fill(~mask, neg), lh.masked_fill(~mask, neg)
    lp, lph = torch.log_softmax(l, -1), torch.log_softmax(lh, -1)
    p, ph = lp.exp(), lph.exp()
    kl = (p * (lp - lph)).masked_fill(~mask, 0).sum(-1)                  # (H, n_q)
    top1 = (l.argmax(-1) == lh.argmax(-1)).double()
    o, oh = p @ vv, ph @ vv
    rel = (oh - o).norm(dim=-1) / o.norm(dim=-1)
    lv, lhv = l[:, mask], lh[:, mask]                                    # 유효한 점수만
    lc = lv - lv.mean(-1, keepdim=True)
    slope = (lc * (lhv - lhv.mean(-1, keepdim=True))).sum(-1) / (lc * lc).sum(-1)
    return {"kl": kl.mean().item(), "top1": top1.mean().item(), "rel_err": rel.mean().item(),
            "logit_slope": slope.mean().item()}


def attn_descriptives(q, k, n, n_q, scaling, group):
    """원본 어텐션의 성격: 엔트로피, 첫 토큰(어텐션 싱크)이 1등인 비율, 1등 키의 가중치."""
    qq = q[:, n - n_q:n]
    kk = k[:, :n].repeat_interleave(group, 0)
    pos = torch.arange(n - n_q, n, device=q.device)[:, None]
    mask = torch.arange(n, device=q.device)[None, :] <= pos
    l = ((qq @ kk.transpose(-1, -2)) * scaling).masked_fill(~mask, torch.finfo(q.dtype).min)
    p = torch.softmax(l, -1)
    ent = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    return {"entropy": ent.mean().item(), "sink_top1_rate": (p.argmax(-1) == 0).double().mean().item(),
            "top1_weight": p.max(-1).values.mean().item()}


# ------------------------------------------------------------------ 통계
def ci95(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def compare(prod, mse, better_low, n_cond):
    prod, mse = np.asarray(prod), np.asarray(mse)
    diff = prod - mse
    t = stats.ttest_rel(prod, mse)
    try:
        w = stats.wilcoxon(prod, mse).pvalue
    except ValueError:            # 차이가 전부 0
        w = 1.0
    sig = bool(t.pvalue < ALPHA / n_cond and w < ALPHA / n_cond)
    if not sig:
        winner = "차이 없음"
    else:
        prod_better = diff.mean() < 0 if better_low else diff.mean() > 0
        winner = "prod" if prod_better else "mse"
    sd = diff.std(ddof=1)
    return {"mse_mean": float(mse.mean()), "mse_ci95": ci95(mse), "prod_mean": float(prod.mean()),
            "prod_ci95": ci95(prod), "diff_mean": float(diff.mean()), "diff_ci95": ci95(diff),
            "ratio_prod_over_mse": float(prod.mean() / mse.mean()) if mse.mean() != 0 else None,
            "t": float(t.statistic) if np.isfinite(t.statistic) else 0.0,
            "p_t": float(t.pvalue) if np.isfinite(t.pvalue) else 1.0, "p_wilcoxon": float(w),
            "d_z": float(diff.mean() / sd) if sd > 0 else 0.0, "significant": sig, "winner": winner}


# ------------------------------------------------------------------ 그림
def make_figures(per_seed, ns, prefix, D, n_seed):
    # 그림 2: 출력 상대오차와 KL (위: 출력 오차, 아래: KL), 열 = n
    fig, axes = plt.subplots(2, len(ns), figsize=(4.2 * len(ns), 7.2), dpi=200, sharey="row")
    for row, (metric, ylabel) in enumerate((("rel_err", "어텐션 출력 상대오차"), ("kl", "KL 발산"))):
        for col, n in enumerate(ns):
            ax = axes[row, col]
            data, pos, colors = [], [], []
            for i, b in enumerate(BITS):
                for j, (m, c) in enumerate(((("mse", C_MSE)), ("prod", C_PROD))):
                    data.append(per_seed[f"b{b}_n{n}_{m}"][metric]); pos.append(i * 3 + j); colors.append(c)
            bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=False,
                            medianprops={"color": INK, "lw": 1.2})
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor(c)
            ax.set_yscale("log")
            ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels([f"b={b}" for b in BITS])
            if row == 0:
                ax.set_title(f"n = {n:,}", loc="left", fontsize=11, color=INK)
            if col == 0:
                ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_MSE, alpha=0.75), plt.Rectangle((0, 0), 1, 1, color=C_PROD, alpha=0.75)]
    axes[0, 0].legend(handles, ["mse", "prod"], frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"그림 2. 어텐션 오차 (시드 {n_seed}개의 분포, 낮을수록 정확)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(f"figures/{prefix}fig2_attention_error.png")
    plt.close(fig)

    # 그림 3: top-1 일치율 vs n
    fig, axes = plt.subplots(1, len(BITS), figsize=(4.2 * len(BITS), 3.9), dpi=200, sharey=False)
    for ax, b in zip(axes, BITS):
        for m, c in (("mse", C_MSE), ("prod", C_PROD)):
            mean = np.array([np.mean(per_seed[f"b{b}_n{n}_{m}"]["top1"]) for n in ns])
            ci = np.array([ci95(per_seed[f"b{b}_n{n}_{m}"]["top1"]) for n in ns])
            ax.plot(ns, mean * 100, "-o", color=c, lw=2, ms=6, label=m)
            ax.fill_between(ns, ci[:, 0] * 100, ci[:, 1] * 100, color=c, alpha=0.18, lw=0)
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns])
        ax.set_xlabel("문맥 길이 n (토큰)")
        ax.set_title(f"b = {b}비트", loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel("top-1 일치율 (%)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("그림 3. 가장 많이 본 키가 원본과 같은 비율 (띠: 95% 신뢰구간)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(f"figures/{prefix}fig3_top1_vs_n.png")
    plt.close(fig)

    # 그림 2-b: 로짓 기울기 (1단계와의 연결)
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)
    n = ns[-1]
    for m, c, off in (("mse", C_MSE, -0.1), ("prod", C_PROD, 0.1)):
        mean = [np.mean(per_seed[f"b{b}_n{n}_{m}"]["logit_slope"]) for b in BITS]
        ci = np.array([ci95(per_seed[f"b{b}_n{n}_{m}"]["logit_slope"]) for b in BITS])
        ax.errorbar(np.array(BITS) + off, mean, yerr=[np.array(mean) - ci[:, 0], ci[:, 1] - np.array(mean)],
                    fmt="o", ms=7, color=c, capsize=3, label=f"{m} 실측")
    ax.plot(BITS, [1 - D[b] for b in BITS], ls=":", color=INK2, lw=1.2, label="mse 예측 1 − D")
    ax.axhline(1, color=INK, lw=1)
    ax.set_xticks(BITS); ax.set_xlabel("비트폭 b"); ax.set_ylabel("로짓 기울기 (압축 점수 / 원본 점수)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"그림 2-b. 실제 모델에서도 mse는 점수를 1 − D배로 줄이는가 (n = {n:,})", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(f"figures/{prefix}fig2b_logit_slope.png")
    plt.close(fig)


# ------------------------------------------------------------------ 메인
def main():
    a = get_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DT = torch.float64
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    prefix = "smoke_" if a.smoke else ""

    model, windows, data_note = load_everything(a, device)
    dims = capture.model_dims(model)
    d, group = dims["head_dim"], dims["n_heads"] // dims["n_kv_heads"]
    layers = pick_layers(dims["n_layers"])
    scaling = d ** -0.5
    sanity = capture.check_impl_matches_eager(model, windows[0][None, :512].to(device))
    print(f"모델 {a.model if not a.smoke else 'smoke'}  {dims}  층 {layers}  장치 {device}")
    print(f"끼어들기 검사: 기본 구현과의 출력 최대 차이 = {sanity}")
    caps = capture_all(model, windows, layers, device)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Q, K, V 추출 완료 ({time.time() - t0:.1f}초)")

    # 원본 어텐션의 성격 (시드와 무관)
    desc = {}
    for n in a.ns:
        vals = [attn_descriptives(c[li][0].to(device, DT), c[li][1].to(device, DT), n, a.queries, scaling, group)
                for c in caps for li in layers]
        desc[str(n)] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    desc_layer = {}
    for li in layers:
        vals = [attn_descriptives(c[li][0].to(device, DT), c[li][1].to(device, DT), a.ns[-1], a.queries, scaling, group)
                for c in caps]
        desc_layer[str(li)] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}

    per_seed = {f"b{b}_n{n}_{m}": {k: [] for k in METRICS} for b in BITS for n in a.ns for m in METHODS}
    per_layer = {f"L{li}_b{b}_n{n}_{m}": {k: [] for k in METRICS} for li in layers for b in BITS for n in a.ns for m in METHODS}
    for s in range(a.seeds):
        acc = {key: {k: [] for k in METRICS} for key in per_seed}
        for li in layers:
            gen = torch.Generator().manual_seed(SEED_BASE + 1000 * s + li)
            Pi = random_rotation(d, gen, DT)
            S = torch.randn(d, d, generator=gen, dtype=DT)
            Qs = {(b, "mse"): TurboQuantMSE(d, b, rotation=Pi, device=device) for b in BITS}
            Qs.update({(b, "prod"): TurboQuantProd(d, b, rotation=Pi, qjl=S, device=device) for b in BITS})
            lacc = {f"b{b}_n{n}_{m}": {k: [] for k in METRICS} for b in BITS for n in a.ns for m in METHODS}
            for c in caps:
                q, k, v = (t.to(device, DT) for t in c[li])
                for (b, m), Qz in Qs.items():
                    k_hat = Qz.dequantize(Qz.quantize(k))
                    for n in a.ns:
                        r = attn_metrics(q, k, v, k_hat, n, a.queries, scaling, group)
                        for kk_, vv_ in r.items():
                            acc[f"b{b}_n{n}_{m}"][kk_].append(vv_)
                            lacc[f"b{b}_n{n}_{m}"][kk_].append(vv_)
            for key, vals in lacc.items():
                for kk_, arr in vals.items():
                    per_layer[f"L{li}_{key}"][kk_].append(float(np.mean(arr)))
        for key, vals in acc.items():
            for kk_, arr in vals.items():
                per_seed[key][kk_].append(float(np.mean(arr)))
        if s == 0 or (s + 1) % 5 == 0:
            print(f"  시드 {s + 1}/{a.seeds} 완료 ({time.time() - t0:.0f}초)")

    n_cond = len(BITS) * len(a.ns)
    rows = []
    for b in BITS:
        for n in a.ns:
            pm, mm = per_seed[f"b{b}_n{n}_prod"], per_seed[f"b{b}_n{n}_mse"]
            rows.append({"bits": b, "n": n,
                         "kl": compare(pm["kl"], mm["kl"], True, n_cond),
                         "rel_err": compare(pm["rel_err"], mm["rel_err"], True, n_cond),
                         "top1": compare(pm["top1"], mm["top1"], False, n_cond),
                         "logit_slope": {"mse_mean": float(np.mean(mm["logit_slope"])),
                                         "prod_mean": float(np.mean(pm["logit_slope"]))}})
    # H-c: top-1 차이(mse − prod)의 log2 n에 대한 기울기
    hc = {}
    x = np.log2(np.array(a.ns, float))
    for b in BITS:
        slopes = []
        for s in range(a.seeds):
            gap = [per_seed[f"b{b}_n{n}_mse"]["top1"][s] - per_seed[f"b{b}_n{n}_prod"]["top1"][s] for n in a.ns]
            slopes.append(float(np.polyfit(x, gap, 1)[0]))
        tt = stats.ttest_1samp(slopes, 0.0)
        hc[str(b)] = {"slope_mean_per_doubling": float(np.mean(slopes)), "ci95": ci95(slopes),
                      "t": float(tt.statistic) if np.isfinite(tt.statistic) else 0.0,
                      "p": float(tt.pvalue) if np.isfinite(tt.pvalue) else 1.0}

    D = {b: lloyd_max(d, b)[2] for b in BITS}
    make_figures(per_seed, a.ns, prefix, D, a.seeds)
    out = {
        "experiment": "2단계 어텐션 실험" + (" (smoke: 무작위 소형 모델, 결과 의미 없음)" if a.smoke else ""),
        "model": a.model if not a.smoke else "random-small-qwen2", "dims": dims, "layers": layers,
        "data": data_note, "design": {"bits": BITS, "ns": a.ns, "seeds": a.seeds, "windows": a.windows,
                                      "window_len": a.window_len, "queries_per_window": a.queries,
                                      "values": "압축하지 않음", "seed_rule": f"{SEED_BASE} + 1000*s + layer",
                                      "bonferroni_alpha": ALPHA / n_cond},
        "sanity_max_diff_vs_eager": sanity, "D_theory": {str(b): D[b] for b in BITS},
        "descriptives_by_n": desc, "descriptives_by_layer_at_max_n": desc_layer,
        "rows": rows, "H_c_top1_gap_slope": hc,
        "per_seed": per_seed, "per_layer_per_seed": per_layer,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
        "runtime_sec": time.time() - t0,
    }
    with open(f"results/{prefix}exp2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"완료: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
