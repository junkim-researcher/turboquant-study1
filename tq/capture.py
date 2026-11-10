"""
실제 언어모델의 어텐션 함수에 끼어들기 (2단계, 3단계 공용)

transformers는 각 층의 어텐션 계산을 '어텐션 함수'에 맡기는데, 이때
위치 정보(RoPE)가 이미 들어간 쿼리, 키, 값이 그대로 전달된다.
= 실제 KV 캐시에 저장되는 것과 똑같은 키와 값.

여기에 우리 함수 "tq"를 등록해서
  - 2단계: 지정한 층의 쿼리, 키, 값을 복사해 둔다 (CAPTURE)
  - 3단계: 키와 값을 '압축 → 복원'한 것으로 바꿔 넣는다 (KV_TRANSFORM)
계산 자체는 원래의 sdpa 어텐션에 그대로 넘긴다.
"""
import torch

try:
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError as e:  # 아주 오래된 transformers
    raise ImportError("transformers 4.48 이상이 필요합니다: pip install -U transformers") from e

try:  # 버전에 따라 마스크 함수도 같은 이름으로 등록해야 한다
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
except ImportError:
    ALL_MASK_ATTENTION_FUNCTIONS = AttentionMaskInterface = None

IMPL = "tq"
CAPTURE = {"layers": set(), "store": {}}     # layer_idx -> (q, k, v)  (CPU, float32)
KV_TRANSFORM = {"fn": None}                  # fn(layer_idx, key, value) -> (key, value)


def _tq_attention(module, query, key, value, attention_mask, **kwargs):
    li = getattr(module, "layer_idx", None)
    if li in CAPTURE["layers"]:
        CAPTURE["store"][li] = (query.detach().float().cpu(), key.detach().float().cpu(),
                                value.detach().float().cpu())
    if KV_TRANSFORM["fn"] is not None:
        key, value = KV_TRANSFORM["fn"](li, key, value)
    return ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, attention_mask, **kwargs)


AttentionInterface.register(IMPL, _tq_attention)
if AttentionMaskInterface is not None and "sdpa" in ALL_MASK_ATTENTION_FUNCTIONS:
    AttentionMaskInterface.register(IMPL, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


def load_model(name: str, device: str, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation=IMPL)
    except TypeError:  # 예전 버전은 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation=IMPL)
    return tok, model.to(device).eval()


def model_dims(model):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    return {"n_layers": cfg.num_hidden_layers, "n_heads": n_heads, "n_kv_heads": n_kv, "head_dim": head_dim}


@torch.no_grad()
def check_impl_matches_eager(model, input_ids):
    """
    끼어들기가 결과를 바꾸지 않는지 확인: 우리 함수("tq")로 돌린 출력과
    transformers 기본 구현("eager")으로 돌린 출력의 최대 차이를 돌려준다.
    (float32 기준 10^-4 이하면 같은 계산으로 본다)
    """
    if not hasattr(model, "set_attn_implementation"):
        return None
    out_tq = model(input_ids).logits
    model.set_attn_implementation("eager")
    try:
        out_ref = model(input_ids).logits
    finally:
        model.set_attn_implementation(IMPL)
    return float((out_tq - out_ref).abs().max())
