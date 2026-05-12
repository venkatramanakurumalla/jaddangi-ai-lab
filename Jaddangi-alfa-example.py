# ============================================================
# JADDANGI-ALFA · v1.6.8 (PERFECT MATH) + QWEN2-0.5B
# Fixes: Removed Phantom QK-Norm, Fixed Causal Time-Flow
# ============================================================

import os, gc, math, random, warnings, numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================
VOCAB_SIZE = 151936
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
INTERMEDIATE_SIZE = 4864
MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 1000000.0
ATTN_DROPOUT = 0.0
RMSNORM_EPS = 1e-6

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ============================================================
# CORE MODULES
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1: return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class Qwen2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.num_kv_heads = NUM_KV_HEADS
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS

        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * self.head_dim, bias=True)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * self.head_dim, bias=True)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * self.head_dim, bias=True)
        self.o_proj = nn.Linear(NUM_HEADS * self.head_dim, HIDDEN_SIZE, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, MAX_POSITION_EMBEDDINGS, ROPE_THETA)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        # Projections
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE directly (No QK-Norm!)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Perfect Causal Masking Logic
        is_causal = (attention_mask is None and q_len > 1)
        sdpa_mask = None

        if attention_mask is not None:
            sdpa_mask = attention_mask[:, None, None, :].bool()
            if q_len > 1:
                # tril_ ensures we only look at past tokens!
                causal_mask = torch.ones((q_len, key_states.size(2)), device=hidden_states.device, dtype=torch.bool).tril_(key_states.size(2) - q_len)
                sdpa_mask = sdpa_mask & causal_mask[None, None, :, :]
            is_causal = False

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=sdpa_mask,
            dropout_p=ATTN_DROPOUT if self.training else 0.0,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), past_key_value

class Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Qwen2Attention()
        self.mlp = Qwen2MLP()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_kv

# ============================================================
# FULL MODEL
# ============================================================
@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

class JaddangiForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList([Qwen2DecoderLayer() for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, labels=None):
        bsz, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            past_len = past_key_values[0][0].size(2) if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            hidden_states, kv = layer(hidden_states, attention_mask, position_ids, past, use_cache)
            if new_cache is not None: new_cache.append(kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1), ignore_index=-100)

        return ModelOutput(loss=loss, logits=logits, past_key_values=tuple(new_cache) if new_cache else None)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, eos_token_id=None, **kwargs):
        device = next(self.parameters()).device
        past_key_values = None

        for _ in range(max_new_tokens):
            current_input = input_ids[:, -1:] if past_key_values is not None else input_ids
            outputs = self.forward(current_input, past_key_values=past_key_values, use_cache=True)

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

# ============================================================
# WEIGHT LOADING
# ============================================================
def load_qwen2_weights():
    print("\n📥 Loading Qwen2-0.5B...")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.float32, device_map="cpu")
    src = hf_model.state_dict()
    model = JaddangiForCausalLM()
    dest = model.state_dict()

    dest['embed_tokens.weight'].copy_(src['model.embed_tokens.weight'])
    dest['norm.weight'].copy_(src['model.norm.weight'])
    dest['lm_head.weight'].copy_(src['lm_head.weight'])

    for i in range(NUM_LAYERS):
        p_src = f'model.layers.{i}.'
        p_dst = f'layers.{i}.'

        dest[f'{p_dst}input_layernorm.weight'].copy_(src[f'{p_src}input_layernorm.weight'])
        dest[f'{p_dst}post_attention_layernorm.weight'].copy_(src[f'{p_src}post_attention_layernorm.weight'])

        dest[f'{p_dst}self_attn.q_proj.weight'].copy_(src[f'{p_src}self_attn.q_proj.weight'])
        dest[f'{p_dst}self_attn.q_proj.bias'].copy_(src[f'{p_src}self_attn.q_proj.bias'])
        dest[f'{p_dst}self_attn.k_proj.weight'].copy_(src[f'{p_src}self_attn.k_proj.weight'])
        dest[f'{p_dst}self_attn.k_proj.bias'].copy_(src[f'{p_src}self_attn.k_proj.bias'])
        dest[f'{p_dst}self_attn.v_proj.weight'].copy_(src[f'{p_src}self_attn.v_proj.weight'])
        dest[f'{p_dst}self_attn.v_proj.bias'].copy_(src[f'{p_src}self_attn.v_proj.bias'])
        dest[f'{p_dst}self_attn.o_proj.weight'].copy_(src[f'{p_src}self_attn.o_proj.weight'])

        dest[f'{p_dst}mlp.gate_proj.weight'].copy_(src[f'{p_src}mlp.gate_proj.weight'])
        dest[f'{p_dst}mlp.up_proj.weight'].copy_(src[f'{p_src}mlp.up_proj.weight'])
        dest[f'{p_dst}mlp.down_proj.weight'].copy_(src[f'{p_src}mlp.down_proj.weight'])

    model.load_state_dict(dest)
    del hf_model, src, dest
    gc.collect()
    torch.cuda.empty_cache()

    print("✅ Weights loaded!")
    return model

# ============================================================
# TESTS
# ============================================================
@torch.no_grad()
def test_generation(model, tokenizer):
    print("\n🧪 Generation test...")
    model.eval()
    device = next(model.parameters()).device
    prompts = ["The capital of France is", "Python is a programming", "Once upon a time"]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(inputs['input_ids'], max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  {prompt} → {text}\n")
    model.train()

@torch.no_grad()
def test_logit_match(model, tokenizer):
    print("\n🔍 Logit comparison with HuggingFace Qwen2...")
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.float32).to(next(model.parameters()).device)
    model.eval(); hf_model.eval()

    inputs = tokenizer("The capital of France is Paris", return_tensors="pt").to(next(model.parameters()).device)
    out1 = model(inputs['input_ids'])
    out2 = hf_model(**inputs)

    diff = (out1.logits - out2.logits).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    if diff.max().item() < 1e-3: print("  ✅ Perfect match!")
    else: print("  ❌ Differences detected")

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=" * 60)
    print("🔮 JADDANGI-ALFA v1.6.8 (PERFECT MATH) + QWEN2-0.5B")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = load_qwen2_weights().to("cuda" if torch.cuda.is_available() else "cpu")

    test_logit_match(model, tokenizer)
    test_generation(model, tokenizer)

    print("\n✨ Ready!")
