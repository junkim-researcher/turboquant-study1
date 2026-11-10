"""
0단계 판정 요약.
results/ 의 실행 결과를 한눈에 보이게 요약하고,
reference/ 에 있는 Claude 실행 결과와 숫자가 같은지 대조한다.
(같은 시드를 쓰므로 실행 환경이 달라도 거의 같은 값이 나와야 한다.)
"""
import json
import math

res = json.load(open("results/exp0_results.json", encoding="utf-8"))
ref = json.load(open("reference/exp0_results_claude.json", encoding="utf-8"))

A, B, C, D, E = (res[k] for k in ("A_coord_distribution", "B_qr_sign_fix", "C_mse_reproduction",
                                  "D_unbiasedness", "E_storage"))
prod = [r for r in D if r["method"] == "prod"]
mse1 = [r for r in D if r["method"] == "mse" and r["bits"] == 1]
c_pass = sum(r["pass_5pct"] for r in C)
d_pass = sum(r["zero_in_2se"] for r in prod)
e_ok = all(r["mse_bits_per_vec_formula"] == r["mse_bits_per_vec_packed"] and
           r["prod_bits_per_vec_formula"] == r["prod_bits_per_vec_packed"] for r in E)

print("=" * 60)
print(" 0단계(E0) 판정 요약")
print("=" * 60)
print(f"[A] 좌표 분포   KS p(베타)={A['ks_vs_beta']['p']:.3f}  {'통과' if A['ks_vs_beta']['p'] > 0.05 else '확인 필요'}")
print(f"[B] 부호 보정   대각 양수 비율 보정O {B['with_fix']['diag_positive_ratio']:.3f} / 보정X {B['without_fix']['diag_positive_ratio']:.3f}")
print(f"[C] MSE 재현    {c_pass}/{len(C)} 조건 ±5% 이내, 최대 차이 {max(abs(r['rel_diff_vs_theory']) for r in C)*100:.2f}%")
print(f"[D] prod 불편   {d_pass}/{len(prod)} 조건에서 0 포함 (±2SE, 약 1개는 우연히 벗어날 수 있음)")
print(f"[D] mse b=1     추정/실제 = {min(r['ratio_est_over_true'] for r in mse1):.3f} ~ {max(r['ratio_est_over_true'] for r in mse1):.3f}  (2/π = {2/math.pi:.3f})")
print(f"[E] 저장 크기   계산식과 실측 {'일치' if e_ok else '불일치'}")

# Claude 실행 결과와 대조
pairs = [(r["mse_measured"], s["mse_measured"]) for r, s in zip(C, ref["C_mse_reproduction"])]
pairs += [(r["mean_err"], s["mean_err"]) for r, s in zip(D, ref["D_unbiasedness"])]
max_diff = max(abs(a - b) for a, b in pairs)
print("-" * 60)
print(f"Claude 실행 결과와의 최대 차이: {max_diff:.2e}  →  {'같은 결과 (재현 성공)' if max_diff < 1e-6 else '차이 있음, 확인 필요'}")
print(f"실행 환경: torch {res['env']['torch']}, numpy {res['env']['numpy']}, 실행 시간 {res['runtime_sec']:.1f}초")
print("=" * 60)
