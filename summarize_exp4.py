"""
4단계 판정 요약: 키 평균 빼기의 효과.
사용: python summarize_exp4.py  [--smoke]
"""
import json
import sys

path = "results/smoke_exp4_results.json" if "--smoke" in sys.argv else "results/exp4_results.json"
out = json.load(open(path, encoding="utf-8"))
res, tests, red = out["results"], out["tests"], out["norm_sq_ratio_after_centering"]
B0 = res["B0"]["ppl"]


def x(v):
    return f"{v / B0:.2f}배"


print("=" * 88)
print(f" 4단계 판정 요약   {out['experiment']}")
print(f" 평균 c를 구한 텍스트: {out['calibration']}   장치 {out['env']['device']}")
if out["reproducibility_check"]:
    c = out["reproducibility_check"]
    print(f" 재현성 검사(지금 / 3단계): B0 {c['B0_ppl_now_vs_stage3'][0]:.3f} / {c['B0_ppl_now_vs_stage3'][1]:.3f},  "
          f"키 mse b=4 {c['k_mse_b4_ppl_now_vs_stage3'][0]:.3f} / {c['k_mse_b4_ppl_now_vs_stage3'][1]:.3f}")
print("=" * 88)
print(f" [1] perplexity (원본 B0 = {B0:.3f}, 괄호는 원본 대비 배수)")
print(f" {'b':>2} | {'값만(하한선)':>15} | {'mse 안 뺌':>16} {'mse 평균 뺌':>17} | {'prod 안 뺌':>16} {'prod 평균 뺌':>17}")
for b in (2, 3, 4):
    g = lambda k: res[f"{k}_b{b}"]["ppl"]
    print(f" {b:>2} | {g('v_only'):7.2f} ({x(g('v_only'))}) | {g('k_mse'):8.2f} ({x(g('k_mse'))}) "
          f"{g('k_mse_c'):8.2f} ({x(g('k_mse_c'))}) | {g('k_prod'):8.2f} ({x(g('k_prod'))}) "
          f"{g('k_prod_c'):8.2f} ({x(g('k_prod_c'))})")
print("-" * 88)
print(" [2] 조각 단위 대응 비교 (토큰당 손실 차이, 음수 = 앞의 것이 더 좋음)")
names = (("mse_centered_vs_not", "mse: 평균 뺌 − 안 뺌", out["design"]["bonferroni"]["centered_vs_not"]),
         ("prod_centered_vs_not", "prod: 평균 뺌 − 안 뺌", out["design"]["bonferroni"]["centered_vs_not"]),
         ("prod_c_vs_mse_c", "평균 뺀 뒤 prod − mse", out["design"]["bonferroni"]["others"]),
         ("mse_c_vs_vonly", "평균 뺀 mse − 값만(하한선)", out["design"]["bonferroni"]["others"]))
for b in (2, 3, 4):
    for key, lab, alpha in names:
        t = tests[str(b)][key]
        print(f"   b={b}  {lab:<22} {t['diff_mean_nll']:+.4f} [{t['diff_ci95'][0]:+.4f}, {t['diff_ci95'][1]:+.4f}]  "
              f"p_t={t['p_t']:.1e}  {'유의' if t['significant'] else '유의하지 않음'} (기준 {alpha:.4f})")
print("-" * 88)
vals = [red[k] for k in sorted(red, key=int)]
print(f" [3] 평균을 뺀 뒤 압축할 키의 길이² (시험 텍스트): 층 평균 {sum(vals) / len(vals) * 100:.1f}%  "
      f"(가장 많이 줄어든 층 {min(vals) * 100:.1f}%, 가장 적게 {max(vals) * 100:.1f}%)")
print(f"     층별: " + ", ".join(f"{int(k)}:{red[k] * 100:.0f}%" for k in sorted(red, key=int)))
print("-" * 88)
print(f" 실행 시간 {out['runtime_sec'] / 60:.1f}분")
print("=" * 88)
