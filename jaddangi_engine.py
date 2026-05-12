"""
JADDANGI ALFA ENGINE v1.6.9 (Glass Box Architecture)
Designed by: Kurumalla Venkataramana | Jaddangi IT & AI Consultancy
Description: A pure, zero-dependency PyTorch implementation of the 0.5B Small Language Model.
Capabilities: 32K Long-Context, Native SDPA, CPU/Edge Optimized, RoPE Injection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================
# 1. ARCHITECTURE CONFIGURATION
# ====================================================================
class JaddangiConfig:
    def __init__(self):
        # Qwen2-0.5B exact physical dimensions
        self.vocab_size = 151936
        self.hidden_size = 896
        self.intermediate_size = 4864
        self.num_hidden_layers = 24
        self.num_attention_heads = 14
        self.num_key_value_heads = 14
        self.hidden_act = "silu"
        self.max_position_embeddings = 32768  # 32K Long Context
        self.rms_norm_eps = 1e-6
        self.rope_theta = 1000000.0           # High theta for long-context precision
        self.attention_dropout = 0.0

# ====================================================================
# 2. CORE PHYSICS (Norms & Math)
# ====================================================================
class JaddangiRMSNorm(nn.Module):
    """Root Mean Square Normalization: Stabilizes the neural network variance."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Injects mathematical 3D coordinates into the text so the AI understands word order."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # Rotate half the dimensions
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Splits the tensor and rotates the values for complex space mapping."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# ====================================================================
# 3. THE NEURONS (Attention & MLP)
# ====================================================================
class JaddangiAttention(nn.Module):
    """The 'Reading' mechanism: Calculates how every word relates to every other word."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Qwen2 uses bias in the QKV projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_ids, cos, sin, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Inject RoPE (Rotary Positional Embeddings)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # PyTorch SDPA: Hardware-accelerated memory-efficient attention (Flash Attention)
        attn_output = F.scaled_dot_product_attention(
            query_states, 
            key_states, 
            value_states, 
            attn_mask=attention_mask,
            is_causal=(attention_mask is None and q_len > 1)
        )

        # Reshape and project out
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

class JaddangiMLP(nn.Module):
    """The 'Thinking' mechanism: SwiGLU forward-feed network where factual logic is stored."""
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU Activation: SiLU(Gate) * Up
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# ====================================================================
# 4. THE CHASSIS (Layers & Engine)
# ====================================================================
class JaddangiDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = JaddangiAttention(config)
        self.mlp = JaddangiMLP(config)
        self.input_layernorm = JaddangiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = JaddangiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_ids, cos, sin, attention_mask=None):
        # 1. Norm + Attention + Residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        # 2. Norm + MLP + Residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class JaddangiAlfaEngine(nn.Module):
    """The Master Engine. 0.5 Billion Parameters. Pure PyTorch."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else JaddangiConfig()
        
        # Vocab Embeddings
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        
        # 24 Layers of Intelligence
        self.layers = nn.ModuleList([JaddangiDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        self.norm = JaddangiRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Language Modeling Head (Converts hidden thoughts back to words)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Pre-compute RoPE frequencies for up to 32K context
        self._setup_rope()

    def _setup_rope(self):
        inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, self.config.hidden_size // self.config.num_attention_heads, 2).float() / (self.config.hidden_size // self.config.num_attention_heads)))
        t = torch.arange(self.config.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        # Slice the RoPE cache to the current sequence length
        cos = self.cos_cached[0, 0, :seq_len, :].to(device)
        sin = self.sin_cached[0, 0, :seq_len, :].to(device)

        # Pass through all 24 layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, cos, sin)

        # Final Normalization and Prediction
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        class EngineOutput:
            def __init__(self, logits):
                self.logits = logits
        return EngineOutput(logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7):
        """Autonomous Text Generation Loop"""
        self.eval()
        for _ in range(max_new_tokens):
            # Get predictions
            outputs = self(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply Temperature
            if temperature > 0.0:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            # Append token to context
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if we hit EOS (Assuming typical EOS token ID ~ 151643)
            if next_token.item() in [151643, 151645]:
                break
                
        return input_ids

# --- For Local Testing ---
if __name__ == "__main__":
    print("🔬 Booting Jaddangi-Alfa Glass Box Architecture...")
    config = JaddangiConfig()
    engine = JaddangiAlfaEngine(config)
    
    total_params = sum(p.numel() for p in engine.parameters())
    print(f"✅ Engine initialized successfully.")
    print(f"🧮 Total Parameter Count: {total_params:,}")
    print(f"🧠 Context Window: {config.max_position_embeddings:,} Tokens")
