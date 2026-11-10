"""
3단계. 모델 실험 (perplexity) + 키 진단

질문: 키를 mse로 압축할 때와 prod로 압축할 때, 언어모델이 다음 단어를 맞히는 능력
      (perplexity, 낮을수록 좋음)은 원본(B0) 대비 얼마나 나빠지는가? (가설 H-d)
      그리고 2단계에서 본 '큰 오차'의 원인은 키의 어떤 성질 때문인가? (진단)

설계
  - 모델: Qwen2.5-0.5B. 모든 층의 어텐션에 '압축 → 복원'을 끼워 넣는다(RoPE 적용 후 키와 값).
  - 데이터: WikiText-2 test에서 겹치지 않는 1,024토큰 조각 40개
  - 설정 (b = 2, 3, 4 각각)
        B0        : 압축 없음 (기준)
        V만       : 값만 mse b비트, 키는 원본          → 값 압축이 차지하는 몫
        K-mse + V : 키 mse b비트,  값 mse b비트
        K-prod + V: 키 prod b비트, 값 mse b비트         → 위와 다른 점은 키 압축 방식 하나뿐
  - 반복: 시드 5개 (시드마다 층별 회전행렬, QJL 행렬을 새로 뽑음. 같은 시드에서 키 mse와 prod는 같은 회전 공유)
  - 대응 단위: 텍스트 조각 40개. 조각마다 시드 5개의 평균 손실(토큰당 음의 로그우도)을 구해
              같은 조각끼리 K-prod와 K-mse를 비교 (대응표본 t + 윌콕슨, 본페로니 0.05/3)
  - 진단 (원본 모델, 24개 층 전부, 조각 4개):
        키 길이, 공통 성분 비율 = ||키들의 평균||² / 평균(||키||²),
        이상치 채널 정도 = 가장 큰 채널의 크기 / 채널 크기의 중앙값,
        첫 토큰 값 크기 비 = ||첫 토큰의 값|| / 나머지 토큰 값 길이의 중앙값,
        평균을 뺀 키의 길이 비 = 평균(||키 − 평균||²) / 평균(||키||²)  (= 1 − 공통 성분 비율)

실행: python exp3_ppl.py           (Colab T4 GPU, 약 5~10분)
      python exp3_ppl.py --smoke   (인터넷 없이 무작위 소형 모델로 동작 확인)
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
import matplotlib.ticker
import numpy as np
import torch
from scipy import stats

from tq import TurboQuantMSE, TurboQuantProd, random_rotation
from tq import capture
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style

apply_style()

BITS = (2, 3, 4)
SEED_BASE = 300_000
ALPHA = 0.05
C_VONLY = "#1baf7a"   # 값만 압축 (팔레트 3번 슬롯)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.chunks, a.chunk_len = 2, 6, 256
    return a


def load_everything(a, device):
    if a.smoke:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192)
        cfg._attn_implementation = capture.IMPL
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg).to(device).eval()
        g = torch.Generator().manual_seed(1)
        return model, [torch.randint(0, 1000, (a.chunk_len,), generator=g) for _ in range(a.chunks)], "무작위 토큰 (smoke)"
    from datasets import load_dataset
    tok, model = capture.load_model(a.model, device)
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n_avail = len(ids) // a.chunk_len
    pick = np.linspace(0, n_avail - 1, a.chunks).astype(int)
    chunks = [ids[i * a.chunk_len:(i + 1) * a.chunk_len] for i in pick]
    return model, chunks, f"WikiText-2 test, 전체 {len(ids):,}토큰을 {a.chunk_len}토큰 조각 {n_avail}개로 나눠 그중 {a.chunks}개를 고르게 선택"


# ------------------------------------------------------------------ 압축기 만들기
def make_kv_transform(setting, b, seed, d, device):
    """setting: 'v_only' | 'k_mse' | 'k_prod'. 층별 압축기를 처음 쓸 때 만든다."""
    cache = {}
    DT = torch.float32

    def get(li):
        if li not in cache:
            g = torch.Generator().manual_seed(SEED_BASE + 1000 * seed + li)
            Pi_k = random_rotation(d, g, torch.float64)
            S = torch.randn(d, d, generator=g, dtype=torch.float64)
            Pi_v = random_rotation(d, g, torch.float64)
            kq = None
            if setting == "k_mse":
                kq = TurboQuantMSE(d, b, rotation=Pi_k, device=device, dtype=DT)
            elif setting == "k_prod":
                kq = TurboQuantProd(d, b, rotation=Pi_k, qjl=S, device=device, dtype=DT)
            vq = TurboQuantMSE(d, b, rotation=Pi_v, device=device, dtype=DT)
            cache[li] = (kq, vq)
        return cache[li]

    def fn(li, key, value):
        kq, vq = get(li)
        if kq is not None:
            key = kq.dequantize(kq.quantize(key)).to(key.dtype)
        value = vq.dequantize(vq.quantize(value)).to(value.dtype)
        return key, value

    return fn


@torch.no_grad()
def chunk_losses(model, chunks, device):
    out = []
    for c in chunks:
        ids = c[None].to(device)
        loss = model(ids, labels=ids).loss       # 토큰당 평균 음의 로그우도 (다음 토큰 예측)
        out.append(float(loss))
    return np.array(out)


# ------------------------------------------------------------------ 키 진단
@torch.no_grad()
def diagnose(model, chunks, device, n_layers):
    capture.CAPTURE["layers"] = set(range(n_layers))
    acc = {li: {"key_norm": [], "common_share": [], "outlier_ratio": [], "sink_value_ratio": [],
                "value_norm": []} for li in range(n_layers)}
    for c in chunks:
        capture.CAPTURE["store"] = {}
        model(c[None].to(device))
        for li in range(n_layers):
            _, k, v = (t[0].double() for t in capture.CAPTURE["store"][li])     # (Hkv, T, d)
            for h in range(k.shape[0]):
                kh, vh = k[h], v[h]
                sq = (kh * kh).sum(-1)
                mean = kh.mean(0)
                ch = kh.pow(2).mean(0).sqrt()                                     # 채널별 크기
                vn = vh.norm(dim=-1)
                acc[li]["key_norm"].append(float(sq.sqrt().mean()))
                acc[li]["common_share"].append(float((mean @ mean) / sq.mean()))
                acc[li]["outlier_ratio"].append(float(ch.max() / ch.median()))
                acc[li]["sink_value_ratio"].append(float(vn[0] / vn[1:].median()))
                acc[li]["value_norm"].append(float(vn.mean()))
    capture.CAPTURE["layers"] = set()
    return {str(li): {k: float(np.mean(v)) for k, v in d.items()} for li, d in acc.items()}


# ------------------------------------------------------------------ 통계
def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def paired(a, b, n_cond):
    """a − b (조각 단위 대응). a, b: 조각별 손실."""
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    t = stats.ttest_rel(a, b)
    try:
        w = stats.wilcoxon(a, b).pvalue
    except ValueError:
        w = 1.0
    sd = diff.std(ddof=1)
    return {"diff_mean_nll": float(diff.mean()), "diff_ci95": ci95(diff),
            "t": float(t.statistic) if np.isfinite(t.statistic) else 0.0,
            "p_t": float(t.pvalue) if np.isfinite(t.pvalue) else 1.0, "p_wilcoxon": float(w),
            "d_z": float(diff.mean() / sd) if sd > 0 else 0.0,
            "significant": bool(t.pvalue < ALPHA / n_cond and w < ALPHA / n_cond)}


# ------------------------------------------------------------------ 그림
def make_figures(res, prefix, stage2_layers):
    ppl0 = res["B0"]["ppl"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    for key, lab, c, off in (("v_only", "값만 압축", C_VONLY, -0.18), ("k_mse", "키 mse + 값", C_MSE, 0.0),
                             ("k_prod", "키 prod + 값", C_PROD, 0.18)):
        xs, ys, lo, hi = [], [], [], []
        for b in BITS:
            r = res[f"{key}_b{b}"]
            xs.append(b + off); ys.append(r["ppl"]); lo.append(r["ppl_seed_ci95"][0]); hi.append(r["ppl_seed_ci95"][1])
        ys, lo, hi = np.array(ys), np.array(lo), np.array(hi)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt="o", ms=8, color=c, capsize=3, lw=1.6, label=lab)
    ax.axhline(ppl0, color=INK, lw=1.2, ls="--")
    ax.text(4.32, ppl0, f"원본 B0\n{ppl0:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.4g}")
    ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_xticks(BITS); ax.set_xlim(1.6, 4.6); ax.set_xlabel("비트폭 b")
    ax.set_ylabel("perplexity (낮을수록 좋음, 로그 눈금)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("그림 4. 언어모델 성능: 키 압축 방식에 따른 perplexity (오차 막대: 시드 간 95% CI)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"figures/{prefix}fig4_perplexity.png"); plt.close(fig)

    diag = res["diagnostics"]
    layers = sorted(int(k) for k in diag)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=200)
    panels = (("common_share", "키 공통 성분 비율", "||평균 키||² / 평균 ||키||²"),
              ("outlier_ratio", "키 이상치 채널 정도", "최대 채널 크기 / 중앙값"),
              ("sink_value_ratio", "첫 토큰의 값 크기", "||첫 토큰 값|| / 다른 토큰 중앙값"))
    for ax, (k, title, ylab) in zip(axes, panels):
        vals = [diag[str(li)][k] for li in layers]
        cols = [C_PROD if li in stage2_layers else C_MSE for li in layers]
        ax.bar(layers, vals, color=cols, width=0.8)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("층 번호"); ax.set_ylabel(ylab, fontsize=9)
        if k == "sink_value_ratio":
            ax.axhline(1, color=INK, lw=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_PROD), plt.Rectangle((0, 0), 1, 1, color=C_MSE)]
    axes[0].legend(handles, ["2단계에서 쓴 층", "그 밖의 층"], frameon=False, fontsize=8.5)
    fig.suptitle("그림 5. 키와 값의 성질 진단 (원본 모델, 키/값 헤드 평균)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(f"figures/{prefix}fig5_kv_diagnostics.png"); plt.close(fig)


# ------------------------------------------------------------------ 메인
def main():
    a = get_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    prefix = "smoke_" if a.smoke else ""

    model, chunks, data_note = load_everything(a, device)
    dims = capture.model_dims(model)
    d = dims["head_dim"]
    print(f"모델 {'smoke' if a.smoke else a.model}  {dims}  장치 {device}  조각 {len(chunks)}개 × {a.chunk_len}토큰")
    sanity = capture.check_impl_matches_eager(model, chunks[0][None, :512].to(device))
    print(f"끼어들기 검사: 기본 구현과의 출력 최대 차이 = {sanity}")

    capture.KV_TRANSFORM["fn"] = None
    diag = diagnose(model, chunks[:4], device, dims["n_layers"])
    print(f"키 진단 완료 ({time.time() - t0:.0f}초)")

    loss = {"B0": chunk_losses(model, chunks, device)}
    print(f"B0 perplexity {math.exp(loss['B0'].mean()):.3f} ({time.time() - t0:.0f}초)")
    per_seed_ppl = {}
    for setting in ("v_only", "k_mse", "k_prod"):
        for b in BITS:
            key = f"{setting}_b{b}"
            L = []
            for s in range(a.seeds):
                capture.KV_TRANSFORM["fn"] = make_kv_transform(setting, b, s, d, device)
                L.append(chunk_losses(model, chunks, device))
            capture.KV_TRANSFORM["fn"] = None
            L = np.stack(L)                                      # (시드, 조각)
            loss[key] = L.mean(0)                                # 조각별 시드 평균
            per_seed_ppl[key] = np.exp(L.mean(1)).tolist()       # 시드별 perplexity
            print(f"  {key:10s} perplexity {math.exp(L.mean()):9.3f}  ({time.time() - t0:.0f}초)")

    res = {"B0": {"ppl": float(math.exp(loss["B0"].mean())), "mean_nll": float(loss["B0"].mean())}}
    for key in per_seed_ppl:
        ppl = float(math.exp(loss[key].mean()))
        res[key] = {"ppl": ppl, "mean_nll": float(loss[key].mean()),
                    "ppl_increase_pct": (ppl / res["B0"]["ppl"] - 1) * 100,
                    "ppl_per_seed": per_seed_ppl[key], "ppl_seed_ci95": ci95(per_seed_ppl[key])}
    tests = {}
    for b in BITS:
        tests[str(b)] = {
            "prod_vs_mse": paired(loss[f"k_prod_b{b}"], loss[f"k_mse_b{b}"], len(BITS)),
            "mse_vs_B0": paired(loss[f"k_mse_b{b}"], loss["B0"], len(BITS)),
            "mse_vs_vonly": paired(loss[f"k_mse_b{b}"], loss[f"v_only_b{b}"], len(BITS)),
        }
    res["tests"] = tests
    res["diagnostics"] = diag
    stage2_layers = [4, 12, 22] if not a.smoke else [1, 2]
    make_figures(res, prefix, stage2_layers)

    out = {"experiment": "3단계 perplexity + 키 진단" + (" (smoke: 무작위 소형 모델, 결과 의미 없음)" if a.smoke else ""),
           "model": a.model if not a.smoke else "random-small-qwen2", "dims": dims, "data": data_note,
           "design": {"bits": BITS, "seeds": a.seeds, "chunks": a.chunks, "chunk_len": a.chunk_len,
                      "keys": "RoPE 적용 후, 모든 층", "values": "mse b비트 (V만, K-mse, K-prod 공통)",
                      "seed_rule": f"{SEED_BASE} + 1000*s + layer", "bonferroni_alpha": ALPHA / len(BITS),
                      "stage2_layers": stage2_layers},
           "sanity_max_diff_vs_eager": sanity, "results": res,
           "per_chunk_nll": {k: v.tolist() for k, v in loss.items()},
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
           "runtime_sec": time.time() - t0}
    with open(f"results/{prefix}exp3_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"완료: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
