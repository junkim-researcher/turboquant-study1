"""
0단계 [D] 후속 확인: 시드 200개에서 '0 미포함'이 나온 prod 2개 조건을
겹치지 않는 새 시드 2,000개로 다시 측정해 우연인지 확인한다.
(24개 조건을 ±2SE(약 95%) 기준으로 판정하면 우연만으로도 평균 1개 남짓은 벗어난다.)
"""
import json, math
import numpy as np, torch
from tq import TurboQuantProd, random_rotation
from exp0_verify import make_pair

DT = torch.float64
cases = [(64, 1, 0.9), (128, 4, 0.1)]
K = 2000
out = []
for d, b, rho in cases:
    x, q = make_pair(d, rho, 777 + int(rho * 100))
    errs = []
    for k in range(K):
        g = torch.Generator().manual_seed(500_000 + k)
        Pi = random_rotation(d, g, DT); S = torch.randn(d, d, generator=g, dtype=DT)
        qp = TurboQuantProd(d, b, rotation=Pi, qjl=S)
        errs.append(float(q @ qp.dequantize(qp.quantize(x))) - rho)
    e = np.array(errs); m, se = e.mean(), e.std(ddof=1) / math.sqrt(K)
    r = {"d": d, "bits": b, "true_ip": rho, "K": K, "seeds": "500000+k", "mean_err": m, "se": se, "z": m / se, "zero_in_2se": bool(abs(m) <= 2 * se)}
    out.append(r); print(r)
json.dump(out, open("results/exp0_D_followup.json", "w"), indent=2)
