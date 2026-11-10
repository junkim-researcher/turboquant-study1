"""
2단계 판정 요약: 조건별 승자, 가설 H-b, H-c 판정, 원본 어텐션의 성격.
사용: python summarize_exp2.py            (results/exp2_results.json)
      python summarize_exp2.py --smoke    (results/smoke_exp2_results.json)
"""
import json
import sys

path = "results/smoke_exp2_results.json" if "--smoke" in sys.argv else "results/exp2_results.json"
res = json.load(open(path, encoding="utf-8"))
rows = res["rows"]


def fmt(x):
    return f"{x:.2e}" if abs(x) < 1e-2 else f"{x:.4f}"


print("=" * 86)
print(f" 2단계 어텐션 실험 판정 요약   {res['experiment']}")
print(f" 모델 {res['model']}  헤드 차원 {res['dims']['head_dim']}  층 {res['layers']}  "
      f"시드 {res['design']['seeds']}개  장치 {res['env']['device']}")
print(f" 끼어들기 검사(기본 구현과의 출력 차이): {res['sanity_max_diff_vs_eager']}")
print("=" * 86)
print(" [0] 원본 어텐션의 성격 (압축 전)")
for n, dsc in res["descriptives_by_n"].items():
    print(f"   n={int(n):>5,}  엔트로피 {dsc['entropy']:.2f}  1등 키 가중치 {dsc['top1_weight']*100:5.1f}%  "
          f"첫 토큰(어텐션 싱크)이 1등인 비율 {dsc['sink_top1_rate']*100:5.1f}%")
print("-" * 86)
print(f" [1] 조건별 비교 (prod − mse, 대응표본 t + 윌콕슨, 본페로니 {res['design']['bonferroni_alpha']:.4f})")
print(f" {'b':>2} {'n':>6} | {'출력오차 mse':>11} {'prod':>9} {'승자':>5} | {'KL mse':>9} {'prod':>9} {'승자':>5} | "
      f"{'top-1 mse':>9} {'prod':>6} {'승자':>5}")
for r in rows:
    e, k, t = r["rel_err"], r["kl"], r["top1"]
    print(f" {r['bits']:>2} {r['n']:>6,} | {fmt(e['mse_mean']):>11} {fmt(e['prod_mean']):>9} {e['winner']:>5} | "
          f"{fmt(k['mse_mean']):>9} {fmt(k['prod_mean']):>9} {k['winner']:>5} | "
          f"{t['mse_mean']*100:8.1f}% {t['prod_mean']*100:5.1f}% {t['winner']:>5}")
print("-" * 86)
print(" [2] 로짓 기울기 (압축 점수 / 원본 점수, 가장 긴 n)  예측: mse = 1 − D, prod = 1")
nmax = max(r["n"] for r in rows)
for r in rows:
    if r["n"] == nmax:
        D = res["D_theory"][str(r["bits"])]
        print(f"   b={r['bits']}  mse {r['logit_slope']['mse_mean']:.4f} (예측 {1 - D:.4f})   "
              f"prod {r['logit_slope']['prod_mean']:.4f} (예측 1.0000)")
print("-" * 86)
print(" [3] 가설 판정")
for b in (2, 3, 4):
    sel = [r for r in rows if r["bits"] == b]
    wins = {m: sum(r["rel_err"]["winner"] == m for r in sel) for m in ("mse", "prod", "차이 없음")}
    print(f"   H-b  b={b}: 출력 오차 기준 mse 승 {wins['mse']} / prod 승 {wins['prod']} / 차이 없음 {wins['차이 없음']}  (n 조건 {len(sel)}개)")
for b, h in res["H_c_top1_gap_slope"].items():
    verdict = "지지" if h["p"] < 0.05 and h["slope_mean_per_doubling"] > 0 else ("반대 방향" if h["p"] < 0.05 else "지지 안 됨")
    print(f"   H-c  b={b}: n이 2배 될 때 top-1 격차(mse − prod) 변화 {h['slope_mean_per_doubling']*100:+.2f}%p "
          f"[{h['ci95'][0]*100:+.2f}, {h['ci95'][1]*100:+.2f}]  p={h['p']:.3f}  → {verdict}")
print("-" * 86)
print(f" 실행 시간 {res['runtime_sec'] / 60:.1f}분   torch {res['env']['torch']}")
print("=" * 86)
