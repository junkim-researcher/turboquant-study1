"""
1단계 판정 요약: 예측표와의 비교 + mse vs prod 통계 검정 결과를 한눈에.
reference/ 에 Claude 실행 결과가 있으면 숫자가 같은지도 대조한다.
"""
import json
import os

res = json.load(open("results/exp1_results.json", encoding="utf-8"))
rows = res["rows"]


def pct(x):
    return f"{x*100:+.1f}%"


def ok(x):
    return "O" if abs(x) <= 0.10 else "X"


print("=" * 80)
print(" 1단계 내적 실험 판정 요약  (시드 30개, 95% 신뢰구간은 t분포, 예측 일치 기준 ±10%)")
print("=" * 80)
print(" [1] 편향: 실제 내적 0.5일 때 상대 편향  (예측: mse = −D, prod = 0)")
print(f" {'d':>4} {'b':>2} | {'mse 실측':>8} {'예측':>8} {'일치':>4} | {'prod 실측':>9} {'95% 신뢰구간':>18} {'0 포함':>5}")
for r in rows:
    m, p = r["mse_bias_rel"], r["prod_bias_rel"]
    ci = f"[{pct(p['ci95'][0])}, {pct(p['ci95'][1])}]"
    print(f" {r['d']:>4} {r['bits']:>2} | {pct(m['mean']):>8} {pct(r['pred']['mse_bias_rel']):>8} "
          f"{ok(r['pred_check']['mse_bias_rel']):>4} | {pct(p['mean']):>9} {ci:>18} "
          f"{'예' if r['prod_bias_test']['zero_in_ci'] else '아니오':>5}")

print("-" * 80)
print(" [2-a] 편향을 뺀 흩어짐 × d  (고정 수직 쌍: 회전만 바뀔 때)  ← 예측표의 분산과 같은 정의")
print(f" {'d':>4} {'b':>2} | {'mse 실측':>8} {'예측 D(1−D)':>11} {'일치':>4} | {'prod 실측':>9} {'예측':>7} {'일치':>4} | {'배율':>5}")
for r in rows:
    ms, ps = r["mse_spread_x_d"]["mean"], r["prod_spread_x_d"]["mean"]
    print(f" {r['d']:>4} {r['bits']:>2} | {ms:8.4f} {r['pred']['mse_var_x_d']:11.4f} "
          f"{ok(r['pred_check']['mse_spread_vs_D(1-D)']):>4} | {ps:9.4f} {r['pred']['prod_var_x_d']:7.4f} "
          f"{ok(r['pred_check']['prod_spread_vs_pi/2*D(b-1)']):>4} | {ps/ms:5.2f}")

print("-" * 80)
print(" [2-b] 무작위 쌍 전체 오차 분산 × d  (편향이 쌍마다 달라 분산에 섞임 → mse 예측은 D)")
print(f" {'d':>4} {'b':>2} | {'mse 실측':>8} {'예측 D':>8} {'일치':>4} {'(처음 비교: D(1−D) 대비)':>22} | {'prod 실측':>9} | {'배율':>5}")
for r in rows:
    print(f" {r['d']:>4} {r['bits']:>2} | {r['mse_var_x_d']['mean']:8.4f} {r['pred']['mse_total_mse_x_d']:8.4f} "
          f"{ok(r['pred_check']['mse_random_pair_mse_vs_D']):>4} {pct(r['initial_comparison_mse_random_var_vs_D(1-D)']):>22} | "
          f"{r['prod_var_x_d']['mean']:9.4f} | {r['var_compare']['ratio_prod_over_mse']:5.2f}")

print("-" * 80)
print(f" [3] 방식 비교: 무작위 쌍 오차 분산 (prod − mse), 대응표본, 본페로니 유의수준 {res['design']['bonferroni_alpha']:.4f}")
for r in rows:
    v = r["var_compare"]
    lo, hi = v["diff_prod_minus_mse"]["ci95"]
    print(f" d={r['d']:>3} b={r['bits']}  차이 {v['diff_prod_minus_mse']['mean']:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"t={v['t']:6.1f}  p_t={v['p_t']:.1e}  p_윌콕슨={v['p_wilcoxon']:.1e}  d_z={v['d_z']:5.1f}  "
          f"{'유의' if v['significant_bonferroni'] else '유의하지 않음'}")

print("-" * 80)
checks = ["mse_bias_rel", "mse_spread_vs_D(1-D)", "prod_spread_vs_pi/2*D(b-1)", "mse_random_pair_mse_vs_D"]
n_ok = sum(abs(r["pred_check"][c]) <= 0.10 for r in rows for c in checks)
n_zero = sum(r["prod_bias_test"]["zero_in_ci"] for r in rows)
n_sig = sum(r["var_compare"]["significant_bonferroni"] for r in rows)
print(f" 예측 일치(±10%): {n_ok}/{len(rows)*len(checks)}   prod 편향 신뢰구간에 0 포함: {n_zero}/{len(rows)}   "
      f"분산 차이 유의: {n_sig}/{len(rows)}")

ref_path = "reference/exp1_results_claude.json"
if os.path.exists(ref_path):
    ref = json.load(open(ref_path, encoding="utf-8"))
    diffs = [abs(a - b) for key, vals in res["per_seed"].items() for metric, arr in vals.items()
             for a, b in zip(arr, ref["per_seed"][key][metric])]
    mx = max(diffs)
    print(f" Claude 실행 결과와의 최대 차이: {mx:.2e}  →  {'같은 결과 (재현 성공)' if mx < 1e-6 else '차이 있음, 확인 필요'}")
print(f" 실행 환경: torch {res['env']['torch']}, 실행 시간 {res['runtime_sec']:.1f}초")
print("=" * 80)
