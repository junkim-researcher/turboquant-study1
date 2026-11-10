"""
TurboQuant 양자화기 (PyTorch 구현)

논문: Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate", arXiv:2504.19874 (v1), Algorithm 1(TurboQuant_mse), Algorithm 2(TurboQuant_prod).

벡터는 행(row) 단위로 다룬다. x의 모양은 (..., d).
  - 회전:   y = Π u        →  행 벡터로는  y = u @ Π.T
  - 역회전: ũ = Π^T ỹ      →  행 벡터로는  ũ = ỹ @ Π

노름(벡터 길이)은 16비트 실수(FP16)로 저장한다고 보고, 저장 후 값으로 반올림해 쓴다.
같은 코드를 2~3단계(Colab, 실제 모델)에서도 그대로 쓸 수 있도록 장치(device)와
자료형(dtype)을 인자로 받는다.
"""
import math

import torch

from .codebook import lloyd_max


def random_rotation(d: int, gen: torch.Generator, dtype=torch.float64, sign_fix: bool = True):
    """
    정규분포 난수 행렬을 QR 분해해 d×d 직교행렬 Π를 만든다.
    sign_fix=True면 R의 대각 성분 부호로 Q의 열 부호를 보정한다
    (보정해야 Π가 '모든 회전 중에서 고르게' 뽑힌다. Mezzadri, 2007).
    """
    G = torch.randn(d, d, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(G)
    if sign_fix:
        s = torch.sign(torch.diagonal(R))
        s[s == 0] = 1.0
        Q = Q * s.unsqueeze(0)
    return Q.to(dtype)


def _fp16_round(t: torch.Tensor) -> torch.Tensor:
    """FP16으로 저장했다가 다시 읽은 값 (저장 정밀도 반영)."""
    return t.to(torch.float16).to(t.dtype)


class TurboQuantMSE:
    """TurboQuant_mse: 회전 → 좌표별 Lloyd-Max 눈금 번호 저장. 복원 오차(MSE)를 최소화."""

    def __init__(self, d: int, bits: int, seed: int = 0, device="cpu", dtype=torch.float64,
                 rotation: torch.Tensor = None, sign_fix: bool = True):
        self.d, self.bits, self.dtype, self.device = d, bits, dtype, device
        if rotation is None:
            gen = torch.Generator().manual_seed(seed)
            rotation = random_rotation(d, gen, dtype, sign_fix=sign_fix)
        self.Pi = rotation.to(device=device, dtype=dtype)
        c, t, D = lloyd_max(d, bits)
        self.centroids = torch.tensor(c, device=device, dtype=dtype)
        self.boundaries = torch.tensor(t, device=device, dtype=dtype)
        self.D_theory = D  # 길이 1 벡터의 기대 복원 제곱오차 (직접 계산한 이론값)

    # ---- 단위벡터용 (노름 처리 없이) ----
    def quantize_unit(self, u: torch.Tensor) -> torch.Tensor:
        y = u @ self.Pi.T
        if self.bits == 0:
            return torch.zeros_like(y, dtype=torch.long)
        return torch.bucketize(y, self.boundaries)

    def dequantize_unit(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids[idx] @ self.Pi

    # ---- 일반 벡터용 ----
    def quantize(self, x: torch.Tensor) -> dict:
        x = x.to(device=self.device, dtype=self.dtype)
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        u = x / n.clamp_min(1e-30)
        return {"idx": self.quantize_unit(u), "norm": _fp16_round(n)}

    def dequantize(self, code: dict) -> torch.Tensor:
        return self.dequantize_unit(code["idx"]) * code["norm"]

    def bits_per_vector(self) -> int:
        return self.d * self.bits + 16  # 눈금 번호 + 노름(FP16)


class TurboQuantProd:
    """
    TurboQuant_prod: (b-1)비트 TurboQuant_mse + 잔차에 1비트 QJL.
    내적 추정이 평균적으로 정확(불편)하도록 설계.

    복원식(논문 Algorithm 2):
        ũ = ũ_mse + (√(π/2) / d) × γ × Sᵀ z,   z = sign(S r),  r = u - ũ_mse,  γ = ||r||
        x̃ = ||x|| × ũ
    """

    def __init__(self, d: int, bits: int, seed: int = 0, device="cpu", dtype=torch.float64,
                 rotation: torch.Tensor = None, qjl: torch.Tensor = None, sign_fix: bool = True):
        assert bits >= 1
        self.d, self.bits, self.dtype, self.device = d, bits, dtype, device
        gen = torch.Generator().manual_seed(seed)
        if rotation is None:
            rotation = random_rotation(d, gen, dtype, sign_fix=sign_fix)
        if qjl is None:
            # 회전행렬과 겹치지 않도록 다른 스트림(seed + 큰 수)에서 뽑는다
            gen2 = torch.Generator().manual_seed(seed + 1_000_003)
            qjl = torch.randn(d, d, generator=gen2, dtype=torch.float64)
        self.mse = TurboQuantMSE(d, bits - 1, device=device, dtype=dtype, rotation=rotation)
        self.S = qjl.to(device=device, dtype=dtype)
        self.coef = math.sqrt(math.pi / 2) / d

    def quantize(self, x: torch.Tensor) -> dict:
        x = x.to(device=self.device, dtype=self.dtype)
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        u = x / n.clamp_min(1e-30)
        idx = self.mse.quantize_unit(u)
        r = u - self.mse.dequantize_unit(idx)
        gamma = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        z = torch.sign(r @ self.S.T)
        z[z == 0] = 1.0
        return {"idx": idx, "z": z.to(torch.int8), "gamma": _fp16_round(gamma), "norm": _fp16_round(n)}

    def dequantize(self, code: dict) -> torch.Tensor:
        u_mse = self.mse.dequantize_unit(code["idx"])
        u_qjl = self.coef * code["gamma"] * (code["z"].to(self.dtype) @ self.S)
        return (u_mse + u_qjl) * code["norm"]

    def bits_per_vector(self) -> int:
        return self.d * self.bits + 32  # 눈금 번호 (b-1)d + 부호 d + 노름 FP16 + 잔차 노름 FP16
