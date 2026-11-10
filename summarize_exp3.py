"""
3단계 판정 요약: perplexity, 가설 H-d, 키 진단.
사용: python summarize_exp3.py  [--smoke]
"""
import json
import sys

path = "results/smoke_exp3_results.json" if "--smoke" in sys.argv else "results/exp3_results.json"
out = json.load(open(path, encoding="utf-8"))
res, tests, diag = out["results"], out["results"]["tests"], out["results"]["diagnostics"]

print("=" * 84)
print(f" 3단계 판정 요약   {out['experiment']}")
print(f" 모델 {out['model']}  시드 {out['design']['seeds']}개  조각 {out['design']['chunks']}개 × {out['design']['chunk_len']}토큰  "
      f"장치 {out['env']['device']}")
print(f" 끼어들기 검사(기본 구현과의 출력 차이): {out['sanity_max_diff_vs_eager']}")
print("=" * 84)
print(f" [1] perplexity (낮을수록 좋음)   원본 B0 = {res['B0']['ppl']:.3f}")
print(f" {'b':>2} | {'값만 압축':>16} | {'키 mse + 값':>18} | {'키 prod + 값':>18}")
for b in (2, 3, 4):
    v, m, p = res[f"v_only_b{b}"], res[f"k_mse_b{b}"], res[f"k_prod_b{b}"]
    print(f" {b:>2} | {v['ppl']:8.3f} ({v['ppl_increase_pct']:+6.1f}%) | {m['ppl']:9.3f} ({m['ppl_increase_pct']:+6.1f}%) | "
          f"{p['ppl']:9.3f} ({p['ppl_increase_pct']:+6.1f}%)")
print("-" * 84)
print(f" [2] 조각 단위 대응 비교 (토큰당 손실 차이, 본페로니 {out['design']['bonferroni_alpha']:.4f})")
for b in (2, 3, 4):
    t = tests[str(b)]
    pm, mb = t["prod_vs_mse"], t["mse_vs_B0"]
    print(f"   b={b}  prod − mse = {pm['diff_mean_nll']:+.4f} [{pm['diff_ci95'][0]:+.4f}, {pm['diff_ci95'][1]:+.4f}]  "
          f"p_t={pm['p_t']:.1e} p_윌콕슨={pm['p_wilcoxon']:.1e}  {'유의' if pm['significant'] else '유의하지 않음'}   "
          f"| mse − B0 = {mb['diff_mean_nll']:+.4f} {'(유의)' if mb['significant'] else '(유의하지 않음)'}")
print("-" * 84)
print(" [3] 가설 H-d: perplexity 증가폭 순서가 2단계(H-b)와 같은 방향인가 (prod가 더 나쁜가)")
for b in (2, 3, 4):
    pm = tests[str(b)]["prod_vs_mse"]
    ok = pm["significant"] and pm["diff_mean_nll"] > 0
    print(f"   b={b}: {'지지 (prod가 유의하게 더 나쁨)' if ok else ('반대 (mse가 더 나쁨)' if pm['significant'] else '판정 불가 (유의하지 않음)')}")
print("-" * 84)
print(" [4] 키 진단 (원본 모델, 키/값 헤드 평균)")
print(f" {'층':>3} | {'키 길이':>7} | {'공통 성분':>8} | {'이상치 채널':>9} | {'첫 토큰 값 크기 비':>14}")
s2 = set(out["design"]["stage2_layers"])
for li in sorted(diag, key=int):
    g = diag[li]
    mark = " ◀ 2단계" if int(li) in s2 else ""
    print(f" {int(li):>3} | {g['key_norm']:7.2f} | {g['common_share']*100:7.1f}% | {g['outlier_ratio']:9.1f} | "
          f"{g['sink_value_ratio']:14.3f}{mark}")
n = len(diag)
avg = {k: sum(diag[li][k] for li in diag) / n for k in ("common_share", "outlier_ratio", "sink_value_ratio")}
print(f" 평균 | 공통 성분 {avg['common_share']*100:.1f}%  이상치 채널 {avg['outlier_ratio']:.1f}배  첫 토큰 값 크기 비 {avg['sink_value_ratio']:.3f}")
print(f"   → 공통 성분 비율이 x%이면, 키에서 평균을 뺄 때 압축할 벡터의 길이² 가 (100 − x)%로 줄어든다 (어텐션 결과는 그대로).")
print("-" * 84)
print(f" 실행 시간 {out['runtime_sec'] / 60:.1f}분   torch {out['env']['torch']}")
print("=" * 84)
