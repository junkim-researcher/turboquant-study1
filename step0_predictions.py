"""
STEP 0. 이론 예측표 계산 (실험 결과를 보기 전에 고정)

길이 1인 키 x와 쿼리 q의 내적 <q, x>를 압축된 키로 추정할 때의 오차를 예측한다.
D_b = b비트 Lloyd-Max 코드북의 기대 복원 제곱오차 (tq/codebook.py에서 직접 계산)

  [mse, b비트]
    회전에 대해 평균 내면 복원 벡터는 원래 벡터의 (1 - D_b)배
      → 내적 편향(상대)      = -D_b              (참값보다 D_b 비율만큼 작게 나옴)
      → 오차 분산 × d        ≈ D_b (1 - D_b)     (편향 성분을 뺀 흩어짐)
      → 무작위 q에 대한 평균제곱오차 × d = D_b   (편향과 분산을 합친 전체)
  [prod, b비트]  = (b-1)비트 mse + 1비트 QJL
      → 내적 편향            = 0                 (불편)
      → 오차 분산 × d        ≈ (π/2) D_(b-1)     (QJL 분산, 잔차 노름² ≈ D_(b-1))

근사 조건: 고차원, 쿼리와 키가 서로 거의 수직(무작위)일 때. 실험(1단계)으로 확인할 대상.
"""
import datetime
import json
import math
import os

from tq.codebook import lloyd_max

PAPER_D = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}           # 논문 표기값 (정규분포 근사, 반올림)
PAPER_PROD = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}          # 논문의 prod 내적 왜곡 × d


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for d in (64, 128):
        for b in (2, 3, 4):
            D_b = lloyd_max(d, b)[2]
            D_b1 = lloyd_max(d, b - 1)[2]
            mse_var = D_b * (1 - D_b)
            prod_var = math.pi / 2 * D_b1
            rows.append({
                "d": d, "bits": b,
                "D_b_exact": D_b, "D_b_paper": PAPER_D[b],
                "mse_bias_rel": -D_b,
                "mse_var_x_d": mse_var,
                "mse_total_mse_x_d": D_b,
                "prod_bias_rel": 0.0,
                "prod_var_x_d": prod_var,
                "prod_var_x_d_paper": PAPER_PROD[b],
                "var_ratio_prod_over_mse": prod_var / mse_var,
                "bits_per_vector_mse": d * b + 16,
                "bits_per_vector_prod": d * b + 32,
            })

    out = {
        "fixed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        "note": "실험 결과를 보기 전에 고정한 이론 예측값. 길이 1 벡터, 고차원 근사.",
        "predictions": rows,
    }
    with open("results/step0_predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'d':>4} {'b':>2} | {'D_b(직접)':>9} {'D_b(논문)':>9} | {'mse편향':>8} {'mse분산×d':>9} | {'prod분산×d':>10} {'(논문)':>7} | {'분산비':>6}")
    for r in rows:
        print(f"{r['d']:>4} {r['bits']:>2} | {r['D_b_exact']:9.4f} {r['D_b_paper']:9.3f} | "
              f"{r['mse_bias_rel']*100:7.2f}% {r['mse_var_x_d']:9.4f} | "
              f"{r['prod_var_x_d']:10.4f} {r['prod_var_x_d_paper']:7.3f} | {r['var_ratio_prod_over_mse']:6.2f}")


if __name__ == "__main__":
    main()
