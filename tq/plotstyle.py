"""
그래프 공통 스타일.
설치된 한글 글꼴을 자동으로 찾아 쓴다. (Colab에서 fonts-noto-cjk를 설치하면
'Noto Sans CJK JP'라는 이름으로 잡히는데, 이 글꼴에도 한글이 들어 있다.)
"""
import matplotlib
from matplotlib import font_manager

KOREAN_FONT_CANDIDATES = [
    "Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans KR", "NanumGothic",
    "Malgun Gothic", "AppleGothic", "Noto Sans CJK SC",
]

C_MSE, C_PROD = "#2a78d6", "#eb6834"          # mse = 파랑, prod = 주황 (모든 그림에서 고정)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pick_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in KOREAN_FONT_CANDIDATES:
        if cand in names:
            return cand
    return None


def apply_style():
    font = pick_korean_font()
    if font is None:
        print("[안내] 한글 글꼴을 찾지 못해 그래프의 한글이 깨질 수 있습니다. "
              "Colab이라면 `!apt-get -qq install -y fonts-noto-cjk` 후 런타임을 다시 시작하세요.")
    matplotlib.rcParams.update({
        "font.family": font or "DejaVu Sans", "axes.unicode_minus": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    })
    return font
