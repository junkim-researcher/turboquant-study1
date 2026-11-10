"""
4단계 (확장). 키 평균 빼기: 팀이 진단을 근거로 설계한 개선안

배경(3단계 진단): Qwen2.5-0.5B의 키는 평균 58%(앞쪽 층은 96%)가 모든 키가 공유하는 공통 성분이다.
  어텐션은 키 사이의 차이만 쓰지만, TurboQuant의 오차는 벡터 전체 길이에 비례한다.

아이디어: 모든 키에서 같은 벡터 c를 빼고 압축한 뒤, 복원할 때 c를 다시 더한다.
      k̂ = c + Q(k − c)
  - c는 층과 키/값 헤드마다 하나뿐이므로 저장 부담이 거의 없다 (헤드당 64개 숫자).
  - 각 쿼리 q의 점수는 q·k̂ = q·c + q·Q(k − c). q·c는 그 쿼리의 모든 키에 똑같이 더해지는 상수라
    softmax에서 상쇄된다. 즉 c는 오차를 만들지 않고, 압축 오차는 짧아진 벡터 (k − c)에서만 생긴다.
  - c는 실험에 쓰는 텍스트가 아니라 별도 텍스트(WikiText-2 train)에서 미리 구한다 (실제로 쓸 수 있는 방식).

설계
  - 3단계와 같은 모델, 같은 조각 40개, 같은 시드 규칙(같은 회전행렬과 QJL 행렬, 같은 값 압축)
    → '평균을 뺐는가' 하나만 다르다 (대응 설계).
  - 3단계 결과(평균 안 뺌)는 results/exp3_results.json의 조각별 손실을 그대로 쓰고,
    B0와 키 mse b=4를 다시 돌려 3단계 값과 일치하는지 확인한다(재현성 검사).
  - 새로 돌리는 설정: 키 mse(평균 뺌) + 값, 키 prod(평균 뺌) + 값  × b = 2, 3, 4
  - 비교 (조각 단위 대응 t + 윌콕슨)
        ① 평균 뺌 vs 안 뺌 (mse, prod 각각, 6개 비교, 본페로니 0.05/6)
        ② 평균 뺀 뒤에도 prod가 mse보다 나쁜가 (3개 비교, 0.05/3)
  - 층별로 평균을 뺀 뒤 압축할 벡터의 길이²가 얼마나 줄었는지(시험 텍스트 기준)도 잰다.

실행: python exp4_centered.py           (Colab T4, 약 8~12분, 3단계 결과 파일 필요)
      python exp4_centered.py --smoke   (무작위 소형 모델로 동작 확인)
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

from tq import TurboQuantMSE, TurboQuantProd, random_rotation
from tq import capture
from tq.plotstyle import C_MSE, C_PROD, INK, INK2, apply_style
from exp3_ppl import SEED_BASE, chunk_losses, ci95, paired, make_kv_transform, C_VONLY
import exp3_ppl

apply_style()
BITS = (2, 3, 4)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--chunk-len", type=int, default=1024)
    ap.add_argument("--calib-chunks", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.chunks, a.chunk_len, a.calib_chunks = 2, 6, 256, 2
    return a


def load_calibration(a, model_tok):
    """평균 c를 구할 별도 텍스트: WikiText-2 train (시험 텍스트와 겹치지 않음)."""
    if a.smoke:
        g = torch.Generator().manual_seed(99)
        return [torch.randint(0, 1000, (a.chunk_len,), generator=g) for _ in range(a.calib_chunks)], "무작위 토큰 (smoke)"
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = model_tok("\n\n".join(ds["text"][:6000]), return_tensors="pt").input_ids[0]
    n_avail = len(ids) // a.chunk_len
    pick = np.linspace(0, n_avail - 1, a.calib_chunks).astype(int)
    return ([ids[i * a.chunk_len:(i + 1) * a.chunk_len] for i in pick],
            f"WikiText-2 train 앞부분, {a.chunk_len}토큰 조각 {a.calib_chunks}개")


@torch.no_grad()
def key_means(model, chunks, device, n_layers):
    """층, 키/값 헤드별 키 평균 (RoPE 적용 후, 모든 위치)."""
    capture.CAPTURE["layers"] = set(range(n_layers))
    sums, count = {}, 0
    for c in chunks:
        capture.CAPTURE["store"] = {}
        model(c[None].to(device))
        for li in range(n_layers):
            k = capture.CAPTURE["store"][li][1][0].double()          # (Hkv, T, d)
            sums[li] = sums.get(li, 0) + k.sum(1)
        count += chunks[0].shape[0]
    capture.CAPTURE["layers"] = set()
    return {li: (s / count) for li, s in sums.items()}            # (Hkv, d) float64, CPU


@torch.no_grad()
def norm_reduction(model, chunks, device, centers):
    """시험 텍스트에서 평균(||k − c||²) / 평균(||k||²): 압축할 벡터의 길이²가 몇 %로 줄었나."""
    n_layers = len(centers)
    capture.CAPTURE["layers"] = set(range(n_layers))
    acc = {li: [0.0, 0.0] for li in range(n_layers)}
    for c in chunks:
        capture.CAPTURE["store"] = {}
        model(c[None].to(device))
        for li in range(n_layers):
            k = capture.CAPTURE["store"][li][1][0].double()
            kc = k - centers[li][:, None, :]
            acc[li][0] += float((kc * kc).sum()); acc[li][1] += float((k * k).sum())
    capture.CAPTURE["layers"] = set()
    return {str(li): v[0] / v[1] for li, v in acc.items()}


def make_centered_transform(setting, b, seed, d, device, centers):
    """3단계 make_kv_transform과 같은 시드, 같은 순서로 Π, S, 값 회전을 뽑는다 → 평균 빼기만 다름."""
    cache = {}
    DT = torch.float32

    def get(li):
        if li not in cache:
            g = torch.Generator().manual_seed(SEED_BASE + 1000 * seed + li)
            Pi_k = random_rotation(d, g, torch.float64)
            S = torch.randn(d, d, generator=g, dtype=torch.float64)
            Pi_v = random_rotation(d, g, torch.float64)
            if setting == "k_mse_c":
                kq = TurboQuantMSE(d, b, rotation=Pi_k, device=device, dtype=DT)
            else:
                kq = TurboQuantProd(d, b, rotation=Pi_k, qjl=S, device=device, dtype=DT)
            vq = TurboQuantMSE(d, b, rotation=Pi_v, device=device, dtype=DT)
            c = centers[li].to(device=device, dtype=DT)[None, :, None, :]   # (1, Hkv, 1, d)
            cache[li] = (kq, vq, c)
        return cache[li]

    def fn(li, key, value):
        kq, vq, c = get(li)
        kc = key.to(DT) - c
        key = (kq.dequantize(kq.quantize(kc)) + c).to(key.dtype)
        value = vq.dequantize(vq.quantize(value)).to(value.dtype)
        return key, value

    return fn


def make_figures(res, red, prefix):
    ppl0 = res["B0"]["ppl"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    series = (("v_only", "값만 압축 (키 완벽, 하한선)", C_VONLY, "o", True, -0.24),
              ("k_mse", "키 mse (평균 안 뺌)", C_MSE, "o", False, -0.12),
              ("k_mse_c", "키 mse (평균 뺌)", C_MSE, "o", True, 0.0),
              ("k_prod", "키 prod (평균 안 뺌)", C_PROD, "s", False, 0.12),
              ("k_prod_c", "키 prod (평균 뺌)", C_PROD, "s", True, 0.24))
    for key, lab, col, mk, filled, off in series:
        if f"{key}_b2" not in res:
            continue
        xs = [b + off for b in BITS]
        ys = np.array([res[f"{key}_b{b}"]["ppl"] for b in BITS])
        ci = np.array([res[f"{key}_b{b}"].get("ppl_seed_ci95", [res[f"{key}_b{b}"]["ppl"]] * 2) for b in BITS])
        ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], fmt=mk, ms=8, color=col, capsize=3, lw=1.5,
                    mfc=col if filled else "none", mew=1.6, label=lab)
    ax.axhline(ppl0, color=INK, lw=1.2, ls="--")
    ax.text(4.36, ppl0, f"원본\n{ppl0:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 100 else f"{v:.4g}")
    ax.yaxis.set_major_formatter(fmt)
    lo_, hi_ = ax.get_ylim()
    if hi_ / lo_ < 10:            # 범위가 좁으면 작은 눈금에도 숫자 표시
        ax.yaxis.set_minor_formatter(fmt); ax.tick_params(axis="y", which="minor", labelsize=8)
    else:
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(BITS); ax.set_xlim(1.6, 4.65); ax.set_xlabel("비트폭 b")
    ax.set_ylabel("perplexity (낮을수록 좋음, 로그 눈금)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("그림 6. 키 평균 빼기의 효과 (채운 표시 = 평균 뺌, 빈 표시 = 3단계)", loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"figures/{prefix}fig6_centered_perplexity.png"); plt.close(fig)

    layers = sorted(int(k) for k in red)
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
    vals = [red[str(li)] * 100 for li in layers]
    ax.bar(layers, vals, color=C_MSE, width=0.8)
    ax.axhline(100, color=INK, lw=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("층 번호"); ax.set_ylabel("평균 뺀 뒤 길이² (%)")
    ax.set_title(f"그림 7. 평균을 빼면 압축할 키의 길이²가 몇 %로 줄어드나 (시험 텍스트, 평균 {np.mean(vals):.0f}%)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(f"figures/{prefix}fig7_centered_norm_reduction.png"); plt.close(fig)


def main():
    a = get_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    prefix = "smoke_" if a.smoke else ""

    # 3단계와 같은 모델과 시험 조각
    ea = argparse.Namespace(model=a.model, seeds=a.seeds, chunks=a.chunks, chunk_len=a.chunk_len, smoke=a.smoke)
    model, chunks, data_note = exp3_ppl.load_everything(ea, device)
    dims = capture.model_dims(model)
    d, n_layers = dims["head_dim"], dims["n_layers"]
    tok = None
    if not a.smoke:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
    calib, calib_note = load_calibration(a, tok)
    print(f"모델 {'smoke' if a.smoke else a.model}  {dims}  장치 {device}")

    capture.KV_TRANSFORM["fn"] = None
    centers = key_means(model, calib, device, n_layers)
    red = norm_reduction(model, chunks[:4], device, centers)
    print(f"키 평균 계산 완료, 압축할 키 길이²: 평균 빼기 전의 {np.mean(list(red.values())) * 100:.1f}% ({time.time() - t0:.0f}초)")

    # 3단계 결과 불러오기 (smoke는 직접 계산)
    loss, res = {}, {}
    stage3 = None if a.smoke else json.load(open("results/exp3_results.json", encoding="utf-8"))
    loss["B0"] = chunk_losses(model, chunks, device)
    res["B0"] = {"ppl": float(math.exp(loss["B0"].mean()))}
    check = {}
    if stage3 is not None:
        for key in ("v_only", "k_mse", "k_prod"):
            for b in BITS:
                loss[f"{key}_b{b}"] = np.array(stage3["per_chunk_nll"][f"{key}_b{b}"])
                res[f"{key}_b{b}"] = stage3["results"][f"{key}_b{b}"]
        # 재현성 검사: B0, 키 mse b=4를 다시 돌려 3단계와 비교
        check["B0_ppl_now_vs_stage3"] = [res["B0"]["ppl"], stage3["results"]["B0"]["ppl"]]
        L = []
        for s in range(a.seeds):
            capture.KV_TRANSFORM["fn"] = make_kv_transform("k_mse", 4, s, d, device)
            L.append(chunk_losses(model, chunks, device))
        capture.KV_TRANSFORM["fn"] = None
        check["k_mse_b4_ppl_now_vs_stage3"] = [float(math.exp(np.stack(L).mean())), stage3["results"]["k_mse_b4"]["ppl"]]
        print(f"재현성 검사: B0 {check['B0_ppl_now_vs_stage3']}, 키 mse b=4 {check['k_mse_b4_ppl_now_vs_stage3']}")
    else:
        for key in ("v_only", "k_mse", "k_prod"):
            for b in BITS:
                L = []
                for s in range(a.seeds):
                    capture.KV_TRANSFORM["fn"] = make_kv_transform(key, b, s, d, device)
                    L.append(chunk_losses(model, chunks, device))
                capture.KV_TRANSFORM["fn"] = None
                L = np.stack(L)
                loss[f"{key}_b{b}"] = L.mean(0)
                res[f"{key}_b{b}"] = {"ppl": float(math.exp(L.mean())), "ppl_seed_ci95": ci95(np.exp(L.mean(1)))}

    for key in ("k_mse_c", "k_prod_c"):
        for b in BITS:
            L = []
            for s in range(a.seeds):
                capture.KV_TRANSFORM["fn"] = make_centered_transform(key, b, s, d, device, centers)
                L.append(chunk_losses(model, chunks, device))
            capture.KV_TRANSFORM["fn"] = None
            L = np.stack(L)
            loss[f"{key}_b{b}"] = L.mean(0)
            per_seed = np.exp(L.mean(1))
            ppl = float(math.exp(L.mean()))
            res[f"{key}_b{b}"] = {"ppl": ppl, "ppl_increase_pct": (ppl / res["B0"]["ppl"] - 1) * 100,
                                  "ppl_per_seed": per_seed.tolist(), "ppl_seed_ci95": ci95(per_seed)}
            print(f"  {key}_b{b:<2} perplexity {ppl:9.3f}  ({time.time() - t0:.0f}초)")

    tests = {}
    for b in BITS:
        tests[str(b)] = {
            "mse_centered_vs_not": paired(loss[f"k_mse_c_b{b}"], loss[f"k_mse_b{b}"], 6),
            "prod_centered_vs_not": paired(loss[f"k_prod_c_b{b}"], loss[f"k_prod_b{b}"], 6),
            "prod_c_vs_mse_c": paired(loss[f"k_prod_c_b{b}"], loss[f"k_mse_c_b{b}"], 3),
            "mse_c_vs_B0": paired(loss[f"k_mse_c_b{b}"], loss["B0"], 3),
            "mse_c_vs_vonly": paired(loss[f"k_mse_c_b{b}"], loss[f"v_only_b{b}"], 3),
        }
    make_figures(res, red, prefix)
    out = {"experiment": "4단계 키 평균 빼기" + (" (smoke: 무작위 소형 모델, 결과 의미 없음)" if a.smoke else ""),
           "model": a.model if not a.smoke else "random-small-qwen2", "dims": dims,
           "data": data_note, "calibration": calib_note,
           "design": {"bits": BITS, "seeds": a.seeds, "chunks": a.chunks, "chunk_len": a.chunk_len,
                      "method": "k̂ = c + Q(k − c), c = 층, 키/값 헤드별 키 평균 (보정 텍스트)",
                      "seed_rule": f"{SEED_BASE} + 1000*s + layer (3단계와 동일)",
                      "bonferroni": {"centered_vs_not": 0.05 / 6, "others": 0.05 / 3}},
           "reproducibility_check": check, "norm_sq_ratio_after_centering": red,
           "results": res, "tests": tests,
           "per_chunk_nll": {k: np.asarray(v).tolist() for k, v in loss.items() if k.endswith("_c_b2") or "_c_b" in k or k == "B0"},
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"},
           "runtime_sec": time.time() - t0}
    with open(f"results/{prefix}exp4_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"완료: {time.time() - t0:.1f}초")


if __name__ == "__main__":
    main()
