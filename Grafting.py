
# ============================================================
# JADDANGI-ALFA · v1.6.9 (FRONTIER) + QWEN2-0.5B GRAFTING
# Unified Chassis: Perfect Math, 32K RoPE, & VRAM Profiling
# ============================================================

import os, gc, math, random, warnings, time, numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import ModelOutput

# --- Environment & CUDA Setup ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# --- Configuration ---
VOCAB_SIZE = 151936
HIDDEN_SIZE = 896
NUM_LAYERS = 24
NUM_HEADS = 14
NUM_KV_HEADS = 2
INTERMEDIATE_SIZE = 4864
MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 1000000.0
RMSNORM_EPS = 1e-6
TOKENIZER_NAME = "Qwen/Qwen2-0.5B"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ============================================================
# CORE ARCHITECTURE MODULES
# ============================================================

class JaddangiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_norm).to(input_dtype)

class JaddangiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(2) # [B, S, 1, D]
        sin = self.sin_cached[position_ids].unsqueeze(2)
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1: return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim).reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class JaddangiAttention(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.num_heads, self.num_kv_heads = NUM_HEADS, NUM_KV_HEADS
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS

        self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * self.head_dim, bias=True)
        self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * self.head_dim, bias=True)
        self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * self.head_dim, bias=True)
        self.o_proj = nn.Linear(NUM_HEADS * self.head_dim, HIDDEN_SIZE, bias=False)
        self.rotary_emb = JaddangiRotaryEmbedding(self.head_dim, base=ROPE_THETA)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k, v = torch.cat([past_key_value[0], k], dim=1), torch.cat([past_key_value[1], v], dim=1)

        present_kv = (k, v) if use_cache else None
        q, k, v = q.transpose(1, 2), repeat_kv(k.transpose(1, 2), self.num_key_value_groups), repeat_kv(v.transpose(1, 2), self.num_key_value_groups)

        # Causal mask logic for SDPA
        is_causal = (attention_mask is None and q_len > 1)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].bool()
            if q_len > 1:
                causal = torch.ones((q_len, k.size(2)), device=q.device, dtype=torch.bool).tril_(k.size(2)-q_len)
                mask = mask & causal[None, None, :, :]
            attention_mask, is_causal = mask, False

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal, scale=1.0/math.sqrt(self.head_dim))
        return self.o_proj(attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)), present_kv

class JaddangiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class JaddangiDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = JaddangiAttention(None), JaddangiMLP()
        self.input_layernorm, self.post_attention_layernorm = JaddangiRMSNorm(HIDDEN_SIZE), JaddangiRMSNorm(HIDDEN_SIZE)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states, kv = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)), kv

# ============================================================
# FULL ENGINE & GENERATION
# ============================================================

@dataclass
class JaddangiOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

class JaddangiAlfaEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = nn.ModuleList([JaddangiDecoderLayer() for _ in range(NUM_LAYERS)])
        self.norm = JaddangiRMSNorm(HIDDEN_SIZE)
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            past_len = past_key_values[0][0].size(1) if past_key_values else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        hidden_states, new_cache = self.embed_tokens(input_ids), ([] if use_cache else None)
        for i, layer in enumerate(self.layers):
            hidden_states, kv = layer(hidden_states, attention_mask, position_ids, past_key_values[i] if past_key_values else None, use_cache)
            if new_cache is not None: new_cache.append(kv)

        return JaddangiOutput(logits=self.lm_head(self.norm(hidden_states)), past_key_values=tuple(new_cache) if use_cache else None)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, eos_token_id=None):
        past_key_values = None
        for _ in range(max_new_tokens):
            out = self.forward(input_ids[:, -1:] if past_key_values else input_ids, past_key_values=past_key_values, use_cache=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids, past_key_values = torch.cat([input_ids, next_token], dim=-1), out.past_key_values
            if eos_token_id and (next_token == eos_token_id).all(): break
        return input_ids

# ============================================================
# CERTIFICATION SUITE
# ============================================================

class JaddangiCertifier:
    def __init__(self, engine, tokenizer, device):
        self.engine = engine
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def certify(self, context_limit=16384):
        print("\n" + "="*50 + "\n🚀 JADDANGI-ALFA v1.6.9 CERTIFICATION\n" + "="*50)
        self.test_logits()
        self.test_vram()
        self.test_niah(context_limit)

    def test_logits(self):
        print("\n🔍 [1/3] Logit Bit-Accuracy...")
        ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.float32).to(self.device)
        ids = self.tokenizer("The laws of physics state", return_tensors="pt").input_ids.to(self.device)
        diff = (self.engine(ids).logits - ref(ids).logits).abs().max().item()
        print(f"   Result: {'✅ PERFECT' if diff < 1e-4 else '❌ DRIFT'} (Max Δ: {diff:.8f})")
        del ref; gc.collect(); torch.cuda.empty_cache()

    def test_vram(self):
        print("\n💾 [2/3] VRAM Scaling...")
        for length in [4096, 16384]:
            torch.cuda.reset_max_memory_allocated(self.device)
            self.engine(torch.randint(0, 1000, (1, length)).to(self.device), use_cache=True)
            print(f"   - {length:5d} tokens: {torch.cuda.max_memory_allocated(self.device)/1024**2:7.2f} MB")

    def test_niah(self, length):
        print(f"\n🧵 [3/3] NIAH Signal ({length} tokens)...")
        needle, question = "PASSCODE: 'JADDANGI-GOLD-99'", "What is the passcode?"
        filler_ids = self.tokenizer.encode("The engine maintains signal. ")
        full_tokens = (filler_ids * ((length - 100) // len(filler_ids)))
        idx = int(len(full_tokens) * 0.85)
        full_tokens = full_tokens[:idx] + self.tokenizer.encode(needle) + full_tokens[idx:] + self.tokenizer.encode(f"\n\nQ: {question} A:")
        ids = torch.tensor([full_tokens], device=self.device)
        print(f"   🧠 Prefilling..."), (start := time.time())
        res = self.tokenizer.decode(self.engine.generate(ids, max_new_tokens=25)[0][ids.shape[1]:])
        print(f"   ⏱️  {time.time()-start:.1f}s | Response: '{res.strip()}'")
        print(f"   Result: {'✅ PASS' if 'JADDANGI-GOLD-99' in res else '❌ FAIL'}")

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    # Dynamically check for GPU to prevent AssertionError
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device.type.upper()}")

    tk = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = JaddangiAlfaEngine().to(device)

    # Grafting weights from HuggingFace
    print("📥 Grafting weights...")
    src = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=torch.float32).state_dict()
    model.load_state_dict({k.replace('model.', ''): v for k, v in src.items() if 'q_norm' not in k and 'k_norm' not in k}, strict=False)

    # Run certification only if on GPU (CPU would take hours for 16K)
    if device.type == "cuda":
        cert = JaddangiCertifier(model, tk, device)
        cert.certify(context_limit=16384)
    else:
        print("\n⚠️ Running on CPU. Skipping intensive 16K Stress Test to prevent freezing.")
        print("To run the full suite, connect to a GPU runtime in Colab.")

        # Simple generation check for CPU
        print("\n🧪 Quick CPU Test:")
        ids = tk.encode("The capital of France is", return_tensors="pt").to(device)
        out = model.generate(ids, max_new_tokens=10)
        print("   " + tk.decode(out[0], skip_special_tokens=True))
