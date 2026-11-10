"""
베타 분포용 Lloyd-Max 코드북 계산

길이 1인 d차원 벡터를 무작위로 회전하면, 각 좌표 x는 [-1, 1]에서
    f_d(x) = C_d * (1 - x^2)^((d-3)/2),   C_d = Γ(d/2) / (√π Γ((d-1)/2))
분포를 따른다 (TurboQuant 논문 Lemma 1). 이 분포에 대해 평균제곱오차가
가장 작은 2^b개의 눈금(centroid)을 Lloyd-Max 반복으로 구한다.

이 분포는 적분이 닫힌 형태로 계산되므로(수치 적분 오차 없음) 정확하다.
  - 구간 [a, b]의 확률:  P = F(b) - F(a),  F는 Beta((d-1)/2, (d-1)/2)의 CDF를 (x+1)/2에 적용
  - 구간 [a, b]의 1차 모멘트: ∫ x f dx = C_d [(1-a²)^(α+1) - (1-b²)^(α+1)] / (2(α+1)),  α=(d-3)/2
"""
from functools import lru_cache

import numpy as np
from scipy.special import betainc, gammaln
from scipy.stats import norm


def _const(d: int) -> float:
    return float(np.exp(gammaln(d / 2) - 0.5 * np.log(np.pi) - gammaln((d - 1) / 2)))


def coord_pdf(x, d: int):
    """회전 후 좌표 하나의 확률밀도 f_d(x)."""
    x = np.asarray(x, dtype=np.float64)
    a = (d - 3) / 2
    return _const(d) * np.clip(1 - x * x, 0, None) ** a


def _cdf(x, d: int):
    s = (d - 1) / 2
    return betainc(s, s, (np.clip(x, -1, 1) + 1) / 2)


def _first_moment(lo, hi, d: int):
    a = (d - 3) / 2
    g = lambda t: np.clip(1 - t * t, 0, None) ** (a + 1)
    return _const(d) * (g(lo) - g(hi)) / (2 * (a + 1))


@lru_cache(maxsize=None)
def lloyd_max(d: int, bits: int, tol: float = 1e-14, max_iter: int = 100000):
    """
    d차원, bits비트 Lloyd-Max 코드북을 반환한다.

    반환값: (centroids, boundaries, D)
      centroids : 길이 2^bits, 오름차순 눈금값
      boundaries: 길이 2^bits - 1, 눈금 사이 경계(이웃 눈금의 중점)
      D         : 길이 1인 벡터 전체의 기대 복원 제곱오차 E||x - x̃||²
                  (좌표별 기대 제곱오차 × d)
    bits = 0 이면 눈금 0 하나만 있는 코드북(모든 좌표를 0으로 복원)을 반환한다.
    """
    if bits == 0:
        return np.array([0.0]), np.array([]), 1.0

    k = 2 ** bits
    # 초깃값: 고차원 근사인 N(0, 1/d)의 등확률 분위수
    p = (np.arange(k) + 0.5) / k
    c = norm.ppf(p) / np.sqrt(d)

    for _ in range(max_iter):
        t = (c[:-1] + c[1:]) / 2                    # 경계 = 이웃 눈금의 중점
        edges = np.concatenate([[-1.0], t, [1.0]])
        mass = _cdf(edges[1:], d) - _cdf(edges[:-1], d)
        mom = _first_moment(edges[:-1], edges[1:], d)
        c_new = mom / mass                          # 눈금 = 구간의 조건부 평균
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new

    t = (c[:-1] + c[1:]) / 2
    edges = np.concatenate([[-1.0], t, [1.0]])
    mass = _cdf(edges[1:], d) - _cdf(edges[:-1], d)
    # 눈금이 조건부 평균이면 E[(x - Q(x))²] = E[x²] - Σ c_i² P_i,  E[x²] = 1/d
    D = 1.0 - d * float(np.sum(c * c * mass))
    return c, t, D


if __name__ == "__main__":
    for d in (64, 128):
        for b in (1, 2, 3, 4):
            c, t, D = lloyd_max(d, b)
            print(f"d={d:4d} b={b}  D={D:.5f}  눈금×√d (양수 쪽) = {np.round(c[c > 0] * np.sqrt(d), 4)}")
