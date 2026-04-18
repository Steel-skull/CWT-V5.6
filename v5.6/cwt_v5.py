"""
Cognitive Workspace Transformer (CWT) v5.5
Hub-and-Spoke Memory Topology with Adaptive Pondering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, NamedTuple
from einops import rearrange
from torch.utils.checkpoint import checkpoint as ckpt_fn
from torch.profiler import record_function


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 1280
    n_layers: int = 16
    n_heads: int = 10
    ffn_dim: int = 3584

    # Hub-and-Spoke memory
    d_spoke: int = 64
    d_hub_private: int = 16
    d_hub_shared: int = 256
    d_tag: int = 16

    # System 1 / System 2
    n_system2_layers: int = 2
    enable_pondering: bool = True
    max_ponder_steps: int = 5       # Hard cap on additional S2 passes (was 3 in v5.2)
    ponder_lambda: float = 0.3      # Geometric prior (expect ~3 steps)
    ponder_loss_weight: float = 0.01
    ponder_halt_threshold: float = 0.95  # Inference early stop

    # Decay
    decay_window: int = 3
    decay_bias_init: float = 3.0
    decay_gradient_floor: float = 0.5
    decay_gradient_floor_system2: float = 0.85

    # Context
    max_seq_len: int = 4096
    rope_theta: float = 10000.0

    # Adaptive depth
    exit_kl_threshold: float = 0.01
    min_layers: int = 3
    uncertainty_threshold: float = 0.7

    # Probe & Feedback
    epistemic_dim: int = 3
    probe_bottleneck: int = 256
    probe_layers: int = 5           # Last N layers always probed (stratified)
    probe_loss_weight: float = 0.1
    s1_exit_weight: float = 0.1     # NEW: gradient pressure on S1 standalone quality

    # Convergence Head (distilled probe replacement)
    d_convergence: int = 64
    conv_distill_weight: float = 0.001   # Calibrated for log-space MSE loss scale

    # CWT-MLA (Hub-Derived Latent Attention)
    d_kv_latent: int = 128        # Compressed KV latent dimension
    n_kv_groups: int = 0          # 0 = MLA mode (default). >0 = fall back to GQA with this many KV groups

    # Epistemic thresholds (calibrated at runtime via calibrate_epistemic_thresholds)
    epistemic_convergent_max: float = 0.1    # Default; overwritten by calibration
    epistemic_resolvable_max: float = 0.3    # Default; overwritten by calibration

    # Training
    dropout: float = 0.1
    init_write_scale: float = 0.5

    # Computed
    d_s_spokes: int = 0
    d_s_hub_priv: int = 0
    d_s_hub: int = 0
    d_s_tags: int = 0
    d_s: int = 0
    d_write: int = 0
    d_collapse_input: int = 0
    d_hub_total: int = 0
    d_readable: int = 0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_system2_layers < self.n_layers
        assert self.n_system2_layers >= 1

        self.d_s_spokes = self.n_layers * self.d_spoke
        self.d_s_hub_priv = self.n_layers * self.d_hub_private
        self.d_s_hub = self.d_s_hub_priv + self.d_hub_shared
        self.d_s_tags = self.n_layers * self.d_tag
        self.d_s = self.d_s_spokes + self.d_s_hub + self.d_s_tags

        self.d_write = self.d_spoke + self.d_hub_private + self.d_hub_shared
        self.d_hub_total = self.d_s_hub_priv + self.d_hub_shared
        self.d_collapse_input = self.d_s_spokes + self.d_hub_total
        self.d_readable = self.d_spoke + self.d_s_hub + self.d_s_tags

        print(f"  CWT v5.5 Configuration:")
        print(f"    Pondering: {'Adaptive (max {})'.format(self.max_ponder_steps) if self.enable_pondering else 'Disabled'}")
        print(f"    Probe: Shared Embedding, Stratified (last {self.probe_layers} + 1 random)")
        print(f"    Hub: {self.d_hub_total} | Total d_s: {self.d_s}")
        print(f"    Conv distill weight: {self.conv_distill_weight}")
        print(f"    CWT-MLA: d_kv_latent={self.d_kv_latent}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_head, max_seq_len=8192, theta=10000.0, scaling_factor=1.0, scaling_mode="none", original_max_seq_len=4096, beta_fast=32, beta_slow=1):
        super().__init__()
        
        if scaling_mode == "yarn" and scaling_factor > 1.0:
            # YaRN frequency modification
            freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
            
            low_freq_wavelen = original_max_seq_len / beta_slow
            high_freq_wavelen = original_max_seq_len / beta_fast
            
            new_freqs = []
            for freq in freqs:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    # High frequency: keep as-is (local patterns)
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    # Low frequency: interpolate (global positions)
                    new_freqs.append(freq / scaling_factor)
                else:
                    # Medium: smooth blend
                    smooth = (wavelen - high_freq_wavelen) / (
                        low_freq_wavelen - high_freq_wavelen
                    )
                    new_freqs.append(
                        (1 - smooth) * freq + smooth * (freq / scaling_factor)
                    )
            freqs = torch.tensor(new_freqs)
        else:
            freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
            # Apply Linear RoPE Scaling for Context Extension
            if scaling_mode == "linear":
                freqs = freqs / scaling_factor

        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        self.register_buffer("cos_cached", angles.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", angles.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x, offset=0):
        T = x.shape[2]
        cos = self.cos_cached[:, :, offset:offset+T, :]
        sin = self.sin_cached[:, :, offset:offset+T, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).flatten(-2)


class CWTLatentCache:
    """Ponder-aware KV cache storing compressed hub latents.
    
    Stores d_latent-dimensional latents instead of full K/V (2 × d_model).
    K/V are reconstructed from latents via the attention module's up-projections.
    
    During pondering: only the current token's latent is overwritten per step.
    All previous positions' latents are frozen.
    
    Memory per layer per position: d_latent (default 128)
    vs full K/V cache: 2 × d_model (default 1024)
    → 8× compression at default settings
    """
    
    def __init__(self, n_layers, d_latent, max_seq_len, batch_size, device,
                 dtype=torch.bfloat16, track_dynamics=False):
        self.n_layers = n_layers
        self.d_latent = d_latent
        self.max_seq_len = max_seq_len
        self.track_dynamics = track_dynamics
        
        # Main cache: (n_layers, B, max_seq_len, d_latent)
        self.cache = torch.zeros(
            n_layers, batch_size, max_seq_len, d_latent,
            device=device, dtype=dtype
        )
        self._seq_lens = torch.zeros(n_layers, dtype=torch.long, device=device)
        
        # ── Dynamics Tracking ──
        if track_dynamics:
            # Per-ponder-step: track latent before overwrite so we can measure delta
            # Stored per-layer for the current generation position only
            self._prev_latent = {}  # layer_idx → (B, 1, d_latent)
            
            # Accumulated telemetry for current token being generated
            self._ponder_deltas = []        # list of (n_layers,) tensors per ponder step
            self._ponder_magnitudes = []    # list of (n_layers,) tensors per ponder step
            
            # Accumulated telemetry across all generated tokens
            self._token_history = []  # list of dicts, one per generated token
    
    def update(self, layer_idx, latent, start_position):
        """Store latent(s) at specified position(s). Track dynamics if enabled."""
        T = latent.shape[1]
        end_position = start_position + T
        
        if self.track_dynamics and T == 1:
            # Single-token generation: track the overwrite
            old_latent = self.cache[layer_idx, :, start_position:end_position, :].clone()
            
            if layer_idx in self._prev_latent:
                # We have a previous value at this position for this layer
                # This means we're in a ponder step (overwriting same position)
                delta = (latent - old_latent).norm(dim=-1).mean().item()
                magnitude = latent.norm(dim=-1).mean().item()
                
                if not self._ponder_deltas or len(self._ponder_deltas[-1]) > layer_idx:
                    # New ponder step
                    self._ponder_deltas.append({})
                    self._ponder_magnitudes.append({})
                
                self._ponder_deltas[-1][layer_idx] = delta
                self._ponder_magnitudes[-1][layer_idx] = magnitude
            
            self._prev_latent[layer_idx] = latent.clone()
        
        # Standard cache update
        self.cache[layer_idx, :, start_position:end_position, :] = latent
        self._seq_lens[layer_idx] = max(
            self._seq_lens[layer_idx].item(), end_position
        )
    
    def get(self, layer_idx, up_to):
        """Retrieve cached latents for positions 0..up_to-1.
        
        Args:
            layer_idx: int — which layer
            up_to: int — number of positions to retrieve
            
        Returns:
            (B, up_to, d_latent)
        """
        # Return as float32 to match model weights during generation
        return self.cache[layer_idx, :, :up_to, :].to(torch.float32)
    
    def seq_len(self, layer_idx):
        """How many positions are cached for this layer."""
        return self._seq_lens[layer_idx].item()
    
    def current_position(self):
        """The position index for the next token to be generated.
        Uses layer 0's seq_len as the canonical position tracker.
        """
        return self._seq_lens[0].item()
    
    # ── Dynamics API ──
    
    def begin_token(self):
        """Call before processing a new token. Resets per-token ponder tracking."""
        if not self.track_dynamics:
            return
        self._prev_latent.clear()
        self._ponder_deltas = []
        self._ponder_magnitudes = []
    
    def end_token(self):
        """Call after fully processing a token. Commits per-token stats to history."""
        if not self.track_dynamics:
            return
        
        pos = self.current_position() - 1  # position just finished
        
        stats = {
            'position': pos,
            'n_ponder_overwrites': len(self._ponder_deltas),
        }
        
        if self._ponder_deltas:
            # Aggregate ponder deltas across steps
            # Each entry in _ponder_deltas is a dict {layer_idx: delta_float}
            n_steps = len(self._ponder_deltas)
            
            # Per-step mean delta across layers
            step_deltas = []
            for step_dict in self._ponder_deltas:
                if step_dict:
                    step_deltas.append(
                        sum(step_dict.values()) / len(step_dict)
                    )
            
            stats['ponder_deltas'] = step_deltas  # delta per ponder step
            stats['ponder_delta_total'] = sum(step_deltas)
            stats['ponder_delta_trend'] = (
                step_deltas[-1] - step_deltas[0] if len(step_deltas) > 1 else 0.0
            )
            # Negative trend = converging (good). Positive = diverging (bad).
            
            # Per-step mean magnitude
            step_mags = []
            for mag_dict in self._ponder_magnitudes:
                if mag_dict:
                    step_mags.append(
                        sum(mag_dict.values()) / len(mag_dict)
                    )
            stats['ponder_magnitudes'] = step_mags
        
        # Cross-position diversity at current cache state
        # Measure how different the cached latents are across positions
        if pos > 1:
            # Sample a layer (last S1 layer) for diversity measurement
            sample_layer = self.n_layers // 2
            cached = self.cache[sample_layer, 0, :pos, :]  # (pos, d_latent)
            
            # Mean pairwise cosine distance (sample if too many positions)
            n_sample = min(pos, 100)
            if pos > n_sample:
                idx = torch.randperm(pos)[:n_sample]
                cached_sample = cached[idx]
            else:
                cached_sample = cached
            
            # Cosine similarity matrix
            norms = cached_sample.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized = cached_sample / norms
            sim_matrix = normalized @ normalized.T
            
            # Mean off-diagonal similarity
            n = sim_matrix.shape[0]
            mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
            mean_similarity = sim_matrix[mask].mean().item()
            
            stats['cross_position_similarity'] = mean_similarity
            # Low (0.0-0.3) = diverse (good). High (0.8-1.0) = redundant (bad).
            
            # Effective dimensionality via explained variance
            if pos >= self.d_latent:
                # SVD on cached latents
                centered = cached - cached.mean(dim=0, keepdim=True)
                try:
                    _, S_vals, _ = torch.svd_lowrank(centered.float(), q=min(32, self.d_latent))
                    variance_explained = S_vals ** 2
                    total_var = variance_explained.sum()
                    if total_var > 0:
                        proportions = variance_explained / total_var
                        # Effective dimensionality: exp(entropy of variance proportions)
                        entropy = -(proportions * proportions.clamp(min=1e-10).log()).sum()
                        stats['effective_dimensionality'] = entropy.exp().item()
                    else:
                        stats['effective_dimensionality'] = 1.0
                except Exception:
                    stats['effective_dimensionality'] = -1.0  # SVD failed
        
        self._token_history.append(stats)
        self._prev_latent.clear()
        self._ponder_deltas = []
        self._ponder_magnitudes = []
    
    def get_dynamics_summary(self):
        """Aggregate dynamics across all generated tokens.
        
        Returns dict of summary statistics for display/logging.
        """
        if not self.track_dynamics or not self._token_history:
            return {}
        
        history = self._token_history
        n_tokens = len(history)
        
        summary = {
            'n_tokens_tracked': n_tokens,
        }
        
        # Ponder convergence statistics
        all_deltas = [h['ponder_deltas'] for h in history if 'ponder_deltas' in h]
        all_trends = [h['ponder_delta_trend'] for h in history if 'ponder_delta_trend' in h]
        all_totals = [h['ponder_delta_total'] for h in history if 'ponder_delta_total' in h]
        
        if all_deltas:
            # Average delta per ponder step across all tokens
            max_steps = max(len(d) for d in all_deltas)
            avg_per_step = []
            for step in range(max_steps):
                step_vals = [d[step] for d in all_deltas if step < len(d)]
                if step_vals:
                    avg_per_step.append(sum(step_vals) / len(step_vals))
            summary['avg_ponder_delta_per_step'] = avg_per_step
            
            # Convergence rate: what fraction of tokens show decreasing deltas?
            n_converging = sum(1 for t in all_trends if t < -1e-6)
            n_diverging = sum(1 for t in all_trends if t > 1e-6)
            n_flat = len(all_trends) - n_converging - n_diverging
            summary['ponder_converging_frac'] = n_converging / max(len(all_trends), 1)
            summary['ponder_diverging_frac'] = n_diverging / max(len(all_trends), 1)
            summary['ponder_flat_frac'] = n_flat / max(len(all_trends), 1)
            
            summary['mean_total_ponder_delta'] = (
                sum(all_totals) / len(all_totals) if all_totals else 0.0
            )
        
        # Cross-position diversity
        all_sim = [h['cross_position_similarity'] for h in history
                   if 'cross_position_similarity' in h]
        if all_sim:
            summary['mean_cross_position_similarity'] = sum(all_sim) / len(all_sim)
            summary['cross_position_similarity_trend'] = (
                all_sim[-1] - all_sim[0] if len(all_sim) > 1 else 0.0
            )
            # Rising similarity over generation = positions becoming more alike (bad)
            # Stable = healthy diversity maintained
        
        # Effective dimensionality
        all_edim = [h['effective_dimensionality'] for h in history
                    if 'effective_dimensionality' in h and h['effective_dimensionality'] > 0]
        if all_edim:
            summary['mean_effective_dimensionality'] = sum(all_edim) / len(all_edim)
            # Target: 30-80% of d_latent.
            # Too low = not using capacity. Too high = no structure.
        
        return summary
    
    def get_per_token_dynamics(self):
        """Return the raw per-token history for detailed analysis."""
        return self._token_history if self.track_dynamics else []
    
    def format_dynamics_display(self, max_width=100):
        """Format dynamics as ASCII visualization for generation traces.
        
        Returns list of (label, bar_string) tuples.
        """
        if not self.track_dynamics or not self._token_history:
            return []
        
        history = self._token_history
        n = len(history)
        
        displays = []
        
        # Ponder total delta per token (how much cache changed during pondering)
        totals = [h.get('ponder_delta_total', 0.0) for h in history]
        if any(t > 0 for t in totals):
            max_total = max(totals) if max(totals) > 0 else 1.0
            bar = ""
            blocks = " ▂▃▄▅▆▇█"
            for t in totals:
                level = int((t / max_total) * (len(blocks) - 1))
                level = max(0, min(level, len(blocks) - 1))
                bar += blocks[level]
            displays.append(("Cache Ponder Churn", bar[:max_width]))
        
        # Ponder convergence direction per token
        trends = [h.get('ponder_delta_trend', 0.0) for h in history]
        if any(abs(t) > 1e-6 for t in trends):
            bar = ""
            for t in trends:
                if t < -1e-4:
                    bar += "↓"   # Converging
                elif t > 1e-4:
                    bar += "↑"   # Diverging
                else:
                    bar += "─"   # Flat
            displays.append(("Ponder Convergence", bar[:max_width]))
        
        # Cross-position similarity over generation
        sims = [h.get('cross_position_similarity', -1.0) for h in history]
        sims_valid = [(i, s) for i, s in enumerate(sims) if s >= 0]
        if sims_valid:
            bar_chars = list(" " * n)
            blocks = " ▂▃▄▅▆▇█"
            for i, s in sims_valid:
                level = int(s * (len(blocks) - 1))
                level = max(0, min(level, len(blocks) - 1))
                bar_chars[i] = blocks[level]
            displays.append(("Cache Diversity (inv)", "".join(bar_chars)[:max_width]))
        
        return displays
    
    def memory_bytes(self):
        return self.cache.nelement() * self.cache.element_size()
    
    def __repr__(self):
        filled = self._seq_lens[0].item()
        total_mb = self.memory_bytes() / 1024 / 1024
        dyn = " +dynamics" if self.track_dynamics else ""
        return (f"CWTLatentCache(layers={self.n_layers}, latent={self.d_latent}, "
                f"filled={filled}/{self.max_seq_len}, mem={total_mb:.1f}MB{dyn})")


class CWTLatentAttention(nn.Module):
    """Multi-head Latent Attention derived from workspace hub.
    
    Q: computed from full workspace content (private + public state)
       → Private state shapes what each position searches for
    K/V: derived from hub content only (public state)
       → Public state is what other positions can see
    Cache: stores compressed hub latent (d_kv_latent) per position
       → Reconstruct K/V on the fly via up-projections
    
    During pondering, only the current token's latent is updated.
    All other positions' cached latents are frozen.
    """
    
    def __init__(self, config, shared_rope=None):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_latent = config.d_kv_latent
        
        # Q path: from full workspace projection (d_model) → Q
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # KV path: hub → compressed latent → K and V
        self.norm_kv = RMSNorm(config.d_hub_total)
        self.kv_down = nn.Linear(config.d_hub_total, self.d_latent, bias=False)
        self.kv_up = nn.Linear(self.d_latent, 2 * config.d_model, bias=False)
        
        # Output projection (unchanged)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        if shared_rope is not None:
            self.rope = shared_rope
        else:
            scale = getattr(config, 'rope_scaling_factor', 1.0)
            self.rope = RotaryPositionEmbedding(
                self.d_head, config.max_seq_len, config.rope_theta, scaling_factor=scale
            )
        self.dropout_p = config.dropout
    
    def forward(self, H_q, hub_content, kv_cache=None, layer_idx=None, cache_start_pos=None):
        """
        Args:
            H_q: (B, T, d_model) — projected from full workspace readable.
                 Used for Q computation. This carries spoke+hub+tag info.
            hub_content: (B, T, d_hub_total) — raw hub slice of workspace.
                 Used for K/V computation. Public state only.
            kv_cache: optional CWTLatentCache for generation
            layer_idx: which layer (for cache indexing)
            cache_start_pos: starting position in the sequence for the current
                 tokens (int). Used for RoPE offset and cache position.
                 None during training (positions start at 0).
                 Set to current sequence position during generation.
        """
        with record_function("cwt::attention"):
            B, T, _ = H_q.shape
            
            # Q from full workspace content (includes spoke influence)
            q = rearrange(
                self.q_proj(H_q), "b t (h d) -> b h t d", h=self.n_heads
            )
            
            # KV latent from hub content only (public state)
            c_kv = self.kv_down(self.norm_kv(hub_content))  # (B, T, d_latent)
            
            if kv_cache is not None:
                # Store current tokens' latents in cache
                pos = cache_start_pos if cache_start_pos is not None else 0
                kv_cache.update(layer_idx, c_kv, pos)
                
                if T == 1 and cache_start_pos is not None:
                    # Single-token generation: reconstruct K/V from full cache
                    n_cached = kv_cache.seq_len(layer_idx)
                    all_latents = kv_cache.get(layer_idx, n_cached)  # (B, n_cached, d_latent)
                    kv_full = self.kv_up(all_latents)
                    k_full, v_full = kv_full.chunk(2, dim=-1)
                    k = rearrange(k_full, "b t (h d) -> b h t d", h=self.n_heads)
                    v = rearrange(v_full, "b t (h d) -> b h t d", h=self.n_heads)
                    # RoPE: Q at current position, K at positions 0..n_cached-1
                    q = self.rope(q, offset=cache_start_pos)
                    k = self.rope(k, offset=0)
                    
                    # No causal mask needed — all cached positions are in the past
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                else:
                    # Prefill: multiple tokens, populate cache, use standard causal attention
                    kv = self.kv_up(c_kv)
                    k, v = kv.chunk(2, dim=-1)
                    k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
                    v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
                    offset = cache_start_pos if cache_start_pos is not None else 0
                    q = self.rope(q, offset=offset)
                    k = self.rope(k, offset=offset)
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True,
                        dropout_p=self.dropout_p if self.training else 0.0
                    )
            else:
                # Training: no cache, standard causal attention
                kv = self.kv_up(c_kv)
                k, v = kv.chunk(2, dim=-1)
                k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)
                v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)
                q = self.rope(q)
                k = self.rope(k)
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    dropout_p=self.dropout_p if self.training else 0.0
                )
            
            return self.o_proj(rearrange(out, "b h t d -> b t (h d)"))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        with record_function("cwt::ffn"):
            return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LayerDesignator(nn.Module):
    def __init__(self, n_layers, d_tag, d_model):
        super().__init__()
        self.d_identity = (d_tag // 2) & ~1
        self.d_signature = d_tag - self.d_identity
        identity = torch.zeros(n_layers, self.d_identity)
        if self.d_identity > 0:
            pos = torch.arange(n_layers).float().unsqueeze(1)
            dim_r = torch.arange(0, self.d_identity, 2).float()
            freqs = torch.exp(dim_r * -(math.log(10000.0) / max(self.d_identity, 1)))
            identity[:, 0::2] = torch.sin(pos * freqs)
            identity[:, 1::2] = torch.cos(pos * freqs)
        self.register_buffer("identity", identity)
        self.signature_proj = nn.Linear(d_model, self.d_signature, bias=False)

    def forward(self, H, layer_idx):
        B, T, _ = H.shape
        ident = self.identity[layer_idx].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        return torch.cat([ident, self.signature_proj(H)], dim=-1)


class FlooredMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S, gate, floor):
        ctx.save_for_backward(S, gate)
        ctx.floor = floor
        return S * gate

    @staticmethod
    def backward(ctx, grad_output):
        S, gate = ctx.saved_tensors
        clamped = torch.max(gate, torch.tensor(ctx.floor, device=gate.device, dtype=gate.dtype))
        return grad_output * clamped, grad_output * S, None

def floored_multiply(S, gate, floor):
    if floor <= 0.0: return S * gate
    return FlooredMultiply.apply(S, gate, floor)


def fused_workspace_write(
    S: torch.Tensor,
    w_spoke: torch.Tensor,
    w_hub_priv: torch.Tensor,
    w_hub_shared: torch.Tensor,
    tag: torch.Tensor,
    ss: int, se: int,
    hps: int, hpe: int,
    hss: int, hse: int,
    ts: int, te: int,
) -> torch.Tensor:
    S[:, :, ss:se] = S[:, :, ss:se] + w_spoke
    S[:, :, hps:hpe] = S[:, :, hps:hpe] + w_hub_priv
    S[:, :, hss:hse] = S[:, :, hss:hse] + w_hub_shared
    S[:, :, ts:te] = S[:, :, ts:te] + tag
    return S


def fused_workspace_read(
    S: torch.Tensor,
    weight: torch.Tensor,
    ss: int, se: int,
    hs: int, he: int,
    ts: int, te: int,
) -> torch.Tensor:
    readable = torch.cat([
        S[:, :, ss:se],
        S[:, :, hs:he],
        S[:, :, ts:te],
    ], dim=-1)
    return F.linear(readable, weight)


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY TOPOLOGY (HUB & SPOKE)
# ══════════════════════════════════════════════════════════════════════════════

class HubDecayGate(nn.Module):
    def __init__(self, layer_idx, config, is_system2=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.decay_window = config.decay_window
        self.gradient_floor = config.decay_gradient_floor_system2 if is_system2 else config.decay_gradient_floor
        self.n_targets = layer_idx
        if self.n_targets == 0: return
        self.query_proj = nn.Linear(config.d_model, config.d_tag, bias=False)
        self.gate_proj = nn.Sequential(
            nn.Linear(config.d_tag * 2, config.d_hub_shared, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, S, H, workspace, current_layer_idx):
        with record_function("cwt::decay_gate"):
            if self.decay_window == 0 or self.n_targets == 0: return S, None
            start = max(0, current_layer_idx - self.decay_window) if self.decay_window > 0 else 0
            if current_layer_idx - start <= 0: return S, None
            n_tags = current_layer_idx - start
            tag_region_start = workspace.tag_offset + start * workspace.d_tag
            tag_region_end = workspace.tag_offset + current_layer_idx * workspace.d_tag
            mean_tag = S[:, :, tag_region_start:tag_region_end].reshape(
                S.shape[0], S.shape[1], n_tags, workspace.d_tag
            ).mean(dim=2)
            query = self.query_proj(H)
            gate = self.gate_proj(torch.cat([query, mean_tag], dim=-1))
            hs_start, hs_end = workspace.get_hub_shared_slice()
            # Gate only hub_shared slice — saves (B, T, d_s-d_hub_shared) * 2 in FlooredMultiply's saved tensors
            hub_shared = S[:, :, hs_start:hs_end]
            gated_hub = floored_multiply(hub_shared, gate, self.gradient_floor)
            S_new = torch.cat([S[:, :, :hs_start], gated_hub, S[:, :, hs_end:]], dim=-1)
            return S_new, gate


class CognitiveWorkspace(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_spoke = config.d_spoke
        self.d_hub_private = config.d_hub_private
        self.d_hub_shared = config.d_hub_shared
        self.d_tag = config.d_tag
        self.d_s = config.d_s
        self.d_s_spokes = config.d_s_spokes
        self.d_s_hub_priv = config.d_s_hub_priv
        self.d_s_hub = config.d_s_hub
        self.hub_priv_offset = self.d_s_spokes
        self.hub_shared_offset = self.hub_priv_offset + self.d_s_hub_priv
        self.tag_offset = self.hub_shared_offset + self.d_hub_shared
        self.designator = LayerDesignator(config.n_layers, config.d_tag, config.d_model)
        self.decay_gates = nn.ModuleList([
            HubDecayGate(i, config, is_system2=(i >= config.n_layers - config.n_system2_layers))
            for i in range(config.n_layers)
        ])

    def get_spoke_slice(self, layer_idx): return layer_idx * self.d_spoke, (layer_idx + 1) * self.d_spoke
    def get_hub_private_slice(self, layer_idx): return self.hub_priv_offset + layer_idx * self.d_hub_private, self.hub_priv_offset + (layer_idx + 1) * self.d_hub_private
    def get_hub_shared_slice(self): return self.hub_shared_offset, self.tag_offset
    def get_hub_slice(self): return self.hub_priv_offset, self.tag_offset
    def get_tag_slice(self, layer_idx): return self.tag_offset + layer_idx * self.d_tag, self.tag_offset + (layer_idx + 1) * self.d_tag
    def get_collapse_slice(self): return 0, self.d_s_spokes + self.d_s_hub

    def get_readable_parts(self, layer_idx):
        return [self.get_spoke_slice(layer_idx), self.get_hub_slice(), (self.tag_offset, self.d_s)]

    def write_to_workspace(self, S, w_spoke, w_hub_priv, w_hub_shared, tag, layer_idx, H):
        hub_gate = None
        S_orig = S
        if layer_idx > 0:
            S, hub_gate = self.decay_gates[layer_idx](S, H, self, layer_idx)
            # FlooredMultiply normally returns a new tensor, but decay gate has
            # early-return paths (decay_window=0, no targets, empty window) that
            # return S unchanged.  Guard: clone if we got the same object back.
            if S.data_ptr() == S_orig.data_ptr():
                S = S.clone()
        else:
            S = S.clone()  # Layer 0: need explicit clone for autograd safety
        ss, se = self.get_spoke_slice(layer_idx)
        hps, hpe = self.get_hub_private_slice(layer_idx)
        hss, hse = self.get_hub_shared_slice()
        ts, te = self.get_tag_slice(layer_idx)
        S = fused_workspace_write(S, w_spoke, w_hub_priv, w_hub_shared, tag, ss, se, hps, hpe, hss, hse, ts, te)
        return S, hub_gate


class WriteRecord(NamedTuple):
    magnitude: torch.Tensor                          # On graph for depth_reg_loss
    layer_idx: int
    hub_shared_gate: Optional[torch.Tensor] = None   # Detached, telemetry only
    spoke_norm: Optional[torch.Tensor] = None        # Detached scalar, diagnostics
    hub_priv_norm: Optional[torch.Tensor] = None     # Detached scalar, diagnostics
    hub_shared_norm: Optional[torch.Tensor] = None   # Detached scalar, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# EPISTEMIC MODULES
# ══════════════════════════════════════════════════════════════════════════════

class HubSelfDistillation(nn.Module):
    """Predicts final hub state from intermediate hub states.
    
    Provides deep supervision without vocab-sized projections.
    Each probed layer gets gradient pressure: "your hub content
    should be progressing toward the final state."
    
    Cost per layer: d_hub -> bottleneck -> d_hub
    Old probe cost per layer: d_hub -> bottleneck -> d_model -> vocab matmul
    Savings: eliminates the d_model x vocab_size matmul entirely
    """
    def __init__(self, d_hub_total, d_bottleneck=128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_hub_total, d_bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(d_bottleneck, d_hub_total, bias=False),
        )
    
    def forward(self, hub_intermediate, hub_final_detached):
        """
        Args:
            hub_intermediate: (B, T, d_hub) from layer N — ON the computation graph
            hub_final_detached: (B, T, d_hub) from final layer — DETACHED, fixed target
        Returns:
            loss: scalar MSE
        """
        predicted = self.predictor(hub_intermediate)
        return F.mse_loss(predicted, hub_final_detached)


class ConvergenceHeadV2(nn.Module):
    """Predicts hub delta norm from hub content.
    
    Trained via distillation: the actual hub delta norm is computed
    for free from workspace writes, and this head learns to predict
    it from hub content alone. At inference time (conv_head mode),
    provides cheap convergence estimates without recomputing deltas.
    
    Single scalar output (delta norm) replaces the old KL + entropy pair.
    """
    
    def __init__(self, d_hub, d_conv=64):
        super().__init__()
        self.d_conv = d_conv
        self.proj = nn.Linear(d_hub, d_conv)
        
        # Predicts delta norm from current + previous projection pair
        self.delta_head = nn.Sequential(
            nn.Linear(d_conv * 2, d_conv),
            nn.GELU(),
            nn.Linear(d_conv, 1),
        )
        
        # Initialize to predict moderate delta
        nn.init.constant_(self.delta_head[-1].bias, 0.5)
        
        self.prev_proj = None
    
    def reset(self):
        """Call at the start of each forward pass before the layer loop."""
        self.prev_proj = None
    
    def forward(self, hub_content):
        """
        Args:
            hub_content: (B, T, d_hub) — hub slice of workspace
            
        Returns:
            pred_delta: (B, T) — predicted hub delta norm (always positive)
        """
        curr_proj = self.proj(hub_content)
        
        if self.prev_proj is None:
            # First layer: return moderate default
            pred_delta = torch.ones(
                hub_content.shape[:2],
                device=hub_content.device,
                dtype=hub_content.dtype
            ) * 0.5
            self.prev_proj = curr_proj.detach()
            return pred_delta
        
        pair = torch.cat([self.prev_proj, curr_proj], dim=-1)
        pred_delta = F.softplus(self.delta_head(pair).squeeze(-1))
        
        self.prev_proj = curr_proj.detach()
        return pred_delta


def convergence_distillation_loss(pred_delta, actual_delta):
    """MSE between predicted and actual hub delta norms.
    
    Both values are positive scalars per token. Log-space so
    proportional errors are weighted equally.
    """
    eps = 1e-6
    return F.mse_loss(
        torch.log(pred_delta + eps),
        torch.log(actual_delta.detach().clamp(min=eps))
    )


class PonderHaltHead(nn.Module):
    """Per-token halt decision for adaptive pondering.
    
    Hub-content-only input. The hub implicitly encodes convergence
    state through the accumulated delta pattern across layers.
    Blind Halt ablation: +0.3% — KL/entropy inputs were redundant.
    """
    def __init__(self, d_hub_total):
        super().__init__()
        self.halt = nn.Sequential(
            nn.Linear(d_hub_total, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, hub_content):
        return torch.sigmoid(self.halt(hub_content))  # (B, T, 1)


@dataclass
class EpistemicState:
    token_entropy: torch.Tensor
    token_non_convergence: torch.Tensor
    mean_entropy: torch.Tensor
    mean_non_convergence: torch.Tensor
    workspace_saturation: torch.Tensor
    hub_churn: torch.Tensor
    veto_rate: torch.Tensor
    ponder_steps: int = 0
    gate_mean: float = 0.5
    gate_std: float = 0.0


def classify_epistemic_state(mean_uncertainty_value: float,
                              convergent_max: float = 0.1,
                              resolvable_max: float = 0.3) -> str:
    """Classify epistemic state using (optionally calibrated) thresholds.
    
    Args:
        mean_uncertainty_value: Mean non-convergence or uncertainty value.
        convergent_max: Below this → convergent (default 0.1, override with calibrated).
        resolvable_max: Below this → resolvable (default 0.3, override with calibrated).
    """
    if mean_uncertainty_value < convergent_max: return "convergent"
    elif mean_uncertainty_value < resolvable_max: return "divergent_resolvable"
    return "divergent_unresolvable"


class DeltaNormalizer(nn.Module):
    """Tracks running statistics of hub delta norms for normalization.
    No learned parameters — pure EMA tracking."""
    
    def __init__(self, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.tensor(1.0))
        self.register_buffer('running_std', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))
    
    @torch.no_grad()
    def update(self, deltas):
        """Update running stats from a batch of delta values."""
        batch_mean = deltas.mean()
        batch_std = deltas.std().clamp(min=1e-6)
        
        if not self.initialized:
            self.running_mean.copy_(batch_mean)
            self.running_std.copy_(batch_std)
            self.initialized.fill_(True)
        else:
            self.running_mean.mul_(self.momentum).add_(batch_mean * (1 - self.momentum))
            self.running_std.mul_(self.momentum).add_(batch_std * (1 - self.momentum))
    
    def normalize(self, delta):
        """Convert raw delta to [0, 1] uncertainty.
        0 = delta is well below average (confident)
        1 = delta is well above average (uncertain)
        """
        z_score = (delta - self.running_mean) / (self.running_std + 1e-6)
        return torch.sigmoid(z_score)


def compute_epistemic_state_from_deltas(delta_history, S_final, normalizer=None):
    """Compute epistemic state from hub delta norms.
    
    Args:
        delta_history: list of (layer_idx, delta_tensor) where delta_tensor is (B, T)
        S_final: final workspace state for shape reference
    """
    device = S_final.device
    B, T = S_final.shape[:2]
    
    n = len(delta_history)
    if n < 1:
        z = torch.zeros(B, T, 1, device=device)
        return EpistemicState(z, z, torch.tensor(0.0, device=device),
                              torch.tensor(1.0, device=device),
                              torch.tensor(0.0, device=device),
                              torch.tensor(1.0, device=device),
                              torch.tensor(0.0, device=device))
    
    def extract_delta(entry):
        if isinstance(entry, tuple):
            return entry[1]
        return entry
    
    # Uncertainty from recent deltas: high deltas = still changing = uncertain
    window = min(3, n)
    delta_stack = torch.stack([extract_delta(d) for d in delta_history[-window:]], dim=0)
    mean_delta = delta_stack.mean(dim=0)  # (B, T)
    
    # Normalize to [0, 1] range — deltas typically range 0-2
    if normalizer is not None and normalizer.initialized:
        uncertainty = normalizer.normalize(mean_delta).unsqueeze(-1)
    else:
        uncertainty = torch.full_like(mean_delta.unsqueeze(-1), 0.5)
    mean_unc = uncertainty.mean()
    
    # Non-convergence: are late deltas higher than early deltas?
    if n >= 3:
        mid = n // 2
        early_deltas = [extract_delta(d).mean() for d in delta_history[:mid]]
        late_deltas = [extract_delta(d).mean() for d in delta_history[mid:]]
        if early_deltas and late_deltas:
            mean_nc = torch.stack(late_deltas).mean() / (torch.stack(early_deltas).mean() + 1e-9)
        else:
            mean_nc = torch.tensor(1.0, device=device)
    else:
        mean_nc = torch.tensor(1.0, device=device)
    
    return EpistemicState(
        token_entropy=uncertainty,
        token_non_convergence=mean_delta.unsqueeze(-1),
        mean_entropy=mean_unc,
        mean_non_convergence=mean_nc,
        workspace_saturation=torch.tensor(0.0, device=device),
        hub_churn=torch.tensor(1.0, device=device),
        veto_rate=torch.tensor(0.0, device=device)
    )


# ══════════════════════════════════════════════════════════════════════════════
# NETWORK COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingToState(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.project_to_state = nn.Linear(config.d_model, config.d_s, bias=False)
        self.norm = RMSNorm(config.d_model)

    def forward(self, input_ids):
        with record_function("cwt::embedding"):
            return self.project_to_state(self.norm(self.embed_dropout(self.embed(input_ids))))


class ExpertBlock(nn.Module):
    def __init__(self, config, layer_idx, shared_rope=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.read_project = nn.Linear(config.d_readable, config.d_model, bias=False)
        self.norm_read = RMSNorm(config.d_model)
        self.attention = CWTLatentAttention(config, shared_rope=shared_rope)
        self.norm_attn = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.ffn_dim, config.dropout)
        self.norm_ffn = RMSNorm(config.d_model)
        self.write_project = nn.Linear(config.d_model, config.d_write, bias=False)
        self.d_spoke = config.d_spoke
        self.d_hub_private = config.d_hub_private

    def forward(self, S, workspace, kv_cache=None, cache_start_pos=None):
        with record_function(f"cwt::expert_block_{self.layer_idx}"):
            # --- Q PATH: Full workspace read (spoke + hub + tags) → d_model ---
            with record_function("cwt::workspace_read"):
                ss, se = workspace.get_spoke_slice(self.layer_idx)
                hs, he = workspace.get_hub_slice()
                ts, te = workspace.tag_offset, workspace.d_s
                H_raw = fused_workspace_read(S, self.read_project.weight, ss, se, hs, he, ts, te)
                H = self.norm_read(H_raw)

            # --- KV PATH: Hub content only (public state) ---
            # Hub content is NOT projected through read_project.
            # It goes directly to attention's kv_down for latent compression.
            hub_content = S[:, :, hs:he]  # (B, T, d_hub_total) — shared across all layers

            # --- Attention + FFN with gradient checkpointing ---
            def attn_ffn(H, hub_content):
                H = H + self.attention(
                    self.norm_attn(H),    # Q input (full workspace projected)
                    hub_content,           # KV input (hub only, raw)
                    kv_cache=kv_cache,
                    layer_idx=self.layer_idx,
                    cache_start_pos=cache_start_pos,
                )
                H = H + self.ffn(self.norm_ffn(H))
                return H

            if self.training:
                H = ckpt_fn(attn_ffn, H, hub_content, use_reentrant=False)
            else:
                H = attn_ffn(H, hub_content)

            # --- Write back to workspace (unchanged) ---
            with record_function("cwt::workspace_write"):
                w_full = self.write_project(H)
                w_spoke = w_full[:, :, :self.d_spoke]
                w_hub_priv = w_full[:, :, self.d_spoke:self.d_spoke + self.d_hub_private]
                w_hub_shared = w_full[:, :, self.d_spoke + self.d_hub_private:]
                magnitude = torch.norm(w_full, p=2, dim=-1, keepdim=True)
                tag = workspace.designator(H, self.layer_idx)
                S_new, hub_gate = workspace.write_to_workspace(S, w_spoke, w_hub_priv, w_hub_shared, tag, self.layer_idx, H)

            # Slim WriteRecord: only magnitude on graph; detached scalars for diagnostics
            return S_new, WriteRecord(
                magnitude=magnitude,
                layer_idx=self.layer_idx,
                hub_shared_gate=hub_gate.detach() if hub_gate is not None else None,
                spoke_norm=w_spoke.detach().norm(dim=-1).mean(),
                hub_priv_norm=w_hub_priv.detach().norm(dim=-1).mean(),
                hub_shared_norm=w_hub_shared.detach().norm(dim=-1).mean(),
            )


class SubspaceCollapse(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.collapse_project = nn.Sequential(
            RMSNorm(config.d_collapse_input),
            nn.Linear(config.d_collapse_input, config.d_model, bias=False),
        )

    def forward(self, S_final, workspace):
        with record_function("cwt::collapse"):
            cs, ce = workspace.get_collapse_slice()
            return self.collapse_project(S_final[:, :, cs:ce])


class OutputDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.d_model)
        self.layer1 = SwiGLUFFN(config.d_model, config.ffn_dim, config.dropout)
        self.norm1 = RMSNorm(config.d_model)
        self.layer2 = SwiGLUFFN(config.d_model, config.ffn_dim, config.dropout)
        self.norm2 = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        with record_function("cwt::output_decoder"):
            x = self.input_norm(x)

            # Checkpoint FFN stack to save activation memory
            def _ffn_stack(x):
                x = x + self.layer1(x)
                x = self.norm1(x)
                x = x + self.layer2(x)
                x = self.norm2(x)
                return x

            if self.training:
                x = ckpt_fn(_ffn_stack, x, use_reentrant=False)
            else:
                x = _ffn_stack(x)
            return self.lm_head(x)


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelOutput:
    logits: torch.Tensor
    active_layers: int
    epistemic: EpistemicState
    write_records: List[WriteRecord]
    S_final: Optional[torch.Tensor] = None
    hub_delta_history: Optional[List[tuple]] = None  # [(layer_idx, delta_tensor), ...]
    distill_loss: Optional[torch.Tensor] = None       # was probe_loss
    halt_probs: Optional[List[torch.Tensor]] = None
    conv_distill_loss: Optional[torch.Tensor] = None
    conv_predictions: Optional[List[tuple]] = None     # [(layer_idx, pred_delta), ...]
    s1_exit_loss: Optional[torch.Tensor] = None

EFFORT_TO_STEPS = {
    "low": 1,
    "medium": 2, 
    "high": None,  # Use config.max_ponder_steps
}

class CognitiveWorkspaceTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.system2_start = config.n_layers - config.n_system2_layers

        self.workspace = CognitiveWorkspace(config)
        self.embedding = EmbeddingToState(config)
        
        # Shared RoPE instance
        d_head = config.d_model // config.n_heads
        scale = getattr(config, 'rope_scaling_factor', 1.0)
        self.shared_rope = RotaryPositionEmbedding(
            d_head, config.max_seq_len, config.rope_theta, scaling_factor=scale
        )
        
        self.blocks = nn.ModuleList([ExpertBlock(config, i, shared_rope=self.shared_rope) for i in range(config.n_layers)])
        self.collapse = SubspaceCollapse(config)
        self.output_decoder = OutputDecoder(config)

        # Hub Self-Distillation
        self.hub_distill = HubSelfDistillation(config.d_hub_total, config.probe_bottleneck)

        # Convergence Head V2 (MLP predicting hub delta norm)
        self.convergence_head = ConvergenceHeadV2(config.d_hub_total, config.d_convergence)

        self.delta_normalizer = DeltaNormalizer()

        # Multiplicative Epistemic Gate
        self.epistemic_gate = nn.Sequential(
            nn.Linear(config.epistemic_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.d_model),
        )

        # Adaptive Pondering Halt Head
        self.halt_head = PonderHaltHead(config.d_hub_total)

        self.apply(self._init_weights)
        self._init_special_weights(config)

        # Workspace state recorder hook (set by generate_cached, None during training)
        self._recorder = None
        
        # Weight Tying: embed <-> lm_head
        self.output_decoder.lm_head.weight = self.embedding.embed.weight


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @torch.no_grad()
    def generate_cached(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=50,
                        pondering=True, reasoning_effort="high", use_conv_head=True,
                        track_cache_dynamics=False, recorder=None, tokenizer=None):
        """Autoregressive generation with ponder-aware latent KV cache.
        
        Returns:
            full_sequence: (B, T_prompt + N) token ids
            cache: CWTLatentCache with dynamics
            token_diagnostics: list of dicts, one per generated token, with keys:
                active_layers, ponder_steps, write_magnitude, hub_shared_write_mag,
                mean_uncertainty, mean_non_convergence, workspace_saturation,
                hub_churn, veto_rate, gate_mean, gate_std, epistemic_state,
                halt_probs
        """
        self.eval()
        B, T_prompt = prompt_ids.shape
        device = prompt_ids.device
        
        cache = CWTLatentCache(
            n_layers=self.config.n_layers,
            d_latent=self.config.d_kv_latent,
            max_seq_len=T_prompt + max_new_tokens,
            batch_size=B,
            device=device,
            dtype=torch.bfloat16,
            track_dynamics=track_cache_dynamics,
        )
        
        # --- Workspace state recording (init run, but don't hook yet — skip prefill) ---
        if recorder is not None:
            recorder.begin_run(
                prompt=tokenizer.decode(prompt_ids[0]) if tokenizer else "(prompt)",
                n_layers=self.config.n_layers,
                d_hub_shared=self.config.d_hub_shared,
                d_hub_private=self.config.d_hub_private,
                d_spoke=self.config.d_spoke,
                d_tag=self.config.d_tag,
                metadata={
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'pondering': pondering,
                    'reasoning_effort': reasoning_effort,
                }
            )
        
        # Prefill (not recorded — processes T tokens simultaneously)
        output = self.forward(
            prompt_ids,
            enable_early_exit=False,
            pondering=pondering,
            reasoning_effort=reasoning_effort,
            use_conv_head=use_conv_head,
            kv_cache=cache,
            cache_start_pos=0,
        )
        
        # Now activate the recorder hook for autoregressive generation
        if recorder is not None:
            self._recorder = recorder
        
        next_logits = output.logits[:, -1, :]
        generated_ids = []
        token_diagnostics = []
        
        for step in range(max_new_tokens):
            # Sample
            if temperature > 0:
                logits = next_logits / temperature
                if top_k > 0:
                    topk_vals, topk_idx = logits.topk(top_k, dim=-1)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, topk_idx, topk_vals)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            current_pos = T_prompt + step
            
            cache.begin_token()  # reset per-token tracking
            
            # Tell the recorder which token we're about to process
            if self._recorder is not None:
                tok_text = tokenizer.decode(next_token[0]) if tokenizer else str(next_token[0].item())
                self._recorder.set_current_token(step, tok_text)
            
            output = self.forward(
                next_token,
                enable_early_exit=False,
                pondering=pondering,
                reasoning_effort=reasoning_effort,
                use_conv_head=use_conv_head,
                kv_cache=cache,
                cache_start_pos=current_pos,
            )
            
            cache.end_token()  # commit per-token stats
            
            # ── Collect per-token diagnostics from this forward pass ──
            ep = output.epistemic
            
            if output.conv_predictions and len(output.conv_predictions) > 0:
                conv_deltas = []
                for entry in output.conv_predictions:
                    if len(entry) == 2:
                        _, delta = entry
                    else:
                        delta = entry
                    conv_deltas.append(delta.mean().item())
                
                direct_entropy = ep.mean_entropy.item()
                direct_non_convergence = sum(conv_deltas) / len(conv_deltas) if conv_deltas else 0.0
            else:
                direct_entropy = ep.mean_entropy.item()
                direct_non_convergence = ep.mean_non_convergence.item()

            td = {
                "active_layers": output.active_layers,
                "ponder_steps": ep.ponder_steps,
                "mean_uncertainty": direct_entropy,
                "mean_non_convergence": direct_non_convergence,
                "workspace_saturation": ep.workspace_saturation.item(),
                "hub_churn": ep.hub_churn.item(),
                "veto_rate": ep.veto_rate.item() if not use_conv_head else None,
                "gate_mean": ep.gate_mean,
                "gate_std": ep.gate_std,
            }
            # Write magnitudes from write_records
            if output.write_records:
                mags = [wr.magnitude.mean().item() for wr in output.write_records]
                td["write_magnitude"] = sum(mags) / len(mags)
                hub_shared_mags = [wr.hub_shared_norm.item() for wr in output.write_records
                                   if wr.hub_shared_norm is not None]
                td["hub_shared_write_mag"] = (sum(hub_shared_mags) / len(hub_shared_mags)) if hub_shared_mags else 0.0
            else:
                td["write_magnitude"] = 0.0
                td["hub_shared_write_mag"] = 0.0
            # Halt probs (list of mean values per ponder step)
            if output.halt_probs:
                td["halt_probs"] = [hp.mean().item() for hp in output.halt_probs]
            else:
                td["halt_probs"] = []
            # Epistemic state classification — always use normalized [0,1] uncertainty
            # from DeltaNormalizer (mean_entropy), NOT raw mean_non_convergence which
            # is a late/early delta ratio (~2-4) and would always classify as unresolvable.
            mu = ep.mean_entropy.item()
            convergent_max = getattr(self.config, 'epistemic_convergent_max', 0.1)
            resolvable_max = getattr(self.config, 'epistemic_resolvable_max', 0.3)
            td["epistemic_state"] = classify_epistemic_state(mu, convergent_max, resolvable_max)
            
            token_diagnostics.append(td)
            
            next_logits = output.logits[:, -1, :]
        
        generated = torch.cat(generated_ids, dim=1)
        full_sequence = torch.cat([prompt_ids, generated], dim=1)
        
        # --- Detach recorder from model (caller is responsible for end_run) ---
        if recorder is not None:
            self._recorder = None
        
        return full_sequence, cache, token_diagnostics

    def _init_special_weights(self, config):
        for i, block in enumerate(self.blocks):
            block.write_project.weight.data *= config.init_write_scale / math.sqrt(i + 1)

        # CWT-MLA: initialize kv_down and up-projections
        for block in self.blocks:
            attn = block.attention
            # kv_down: compress hub to latent — normal init, modest scale
            nn.init.xavier_normal_(attn.kv_down.weight)
            attn.kv_down.weight.data *= 0.5  # conservative initial compression
            # kv_up: reconstruct from latent — normal init
            nn.init.xavier_normal_(attn.kv_up.weight)
            # q_proj, o_proj: standard init (already handled by _init_weights)

        # Soft Ponder Lock
        for i, gate in enumerate(self.workspace.decay_gates):
            if gate.n_targets > 0:
                bias_val = (config.decay_bias_init + 2.0) if i >= self.system2_start else config.decay_bias_init
                nn.init.constant_(gate.gate_proj[0].bias, bias_val)

        # Epistemic Gate biased to transparent
        nn.init.zeros_(self.epistemic_gate[-1].weight)
        nn.init.constant_(self.epistemic_gate[-1].bias, 3.0)

        # Halt Head biased toward halting (sigmoid(1.0)≈0.73)
        nn.init.constant_(self.halt_head.halt[-1].bias, 1.0)

        # ConvergenceHeadV2: restore bias after apply(_init_weights) zeroed it
        nn.init.constant_(self.convergence_head.delta_head[-1].bias, 0.5)

    def _run_conv_epistemic(self, S, h_s, h_e, hub_delta_history, conv_predictions_list, layer_idx):
        """Lightweight epistemic computation using ConvergenceHeadV2 only.
        
        Returns: (delta_mean, probed, conv_delta_token)
        """
        pred_delta = self.convergence_head(S[:, :, h_s:h_e].detach())
        conv_predictions_list.append((layer_idx, pred_delta.detach()))
        
        delta_mean = pred_delta.mean().item()
        probed = True
        
        return delta_mean, probed, pred_delta.detach()

    def forward(self, input_ids, target_ids=None, enable_early_exit=True,
                pondering=None, reasoning_effort="high", use_conv_head=False,
                kv_cache=None, cache_start_pos=None):
        """
        Args:
            input_ids: (B, T) token ids
            target_ids: (B, T) targets for probe loss (training only)
            enable_early_exit: Allow System 1 early exit
            pondering: Override pondering behavior at inference time.
                       None = use config.enable_pondering (default)
                       True = force pondering on
                       False = force pondering off (single S2 pass)
            reasoning_effort: "low" / "medium" / "high" or int.
                       Sets max additional S2 passes.
                       Only used when pondering is active.
                       "low"=1, "medium"=2, "high"=config.max_ponder_steps
                       int=exact cap
            use_conv_head: If True, use convergence head instead of full probe
                       for all epistemic signals (KL, entropy, halt, early exit).
                       ~1000x cheaper — no vocab-sized logits at any layer.
                       Should only be used at inference time.
        """
        # Resolve pondering flag
        if pondering is None:
            do_ponder = self.config.enable_pondering
        else:
            do_ponder = pondering

        # Resolve max ponder steps
        if self.training:
            # Training: always use full config (static graph)
            max_steps = self.config.max_ponder_steps
        elif isinstance(reasoning_effort, int):
            max_steps = max(0, min(reasoning_effort, self.config.max_ponder_steps))
        else:
            cap = EFFORT_TO_STEPS.get(reasoning_effort, None)
            max_steps = cap if cap is not None else self.config.max_ponder_steps

        S = self.embedding(input_ids)
        B, T, _ = S.shape
        if S.shape[-1] < self.workspace.d_s:
            S = F.pad(S, (0, self.workspace.d_s - S.shape[-1]))

        write_records, hub_churn_masks = [], []
        hub_delta_history = []           # list of (layer_pass_idx, delta_tensor)
        distill_hub_states = {}          # layer_pass_idx → hub content for self-distillation
        active_layers, ponder_steps = 0, 0
        h_s, h_e = self.workspace.get_hub_slice()

        sys1_exited = False
        delta_mean = 0.0

        halt_probs_list = []

        conv_distill_loss_accum = torch.tensor(0.0, device=S.device)
        conv_distill_count = 0
        conv_predictions_list = []

        # --- Reset ConvergenceHeadV2 state for this forward pass ---
        self.convergence_head.reset()

        # --- Stratified Probe Schedule ---
        if self.training:
            always_start = max(0, self.config.n_layers - self.config.probe_layers)
            always_probe = set(range(always_start, self.config.n_layers))
            if always_start > 0:
                early_sample = torch.randint(0, always_start, (1,)).item()
                active_probes = always_probe | {early_sample}
            else:
                active_probes = always_probe
        else:
            active_probes = set(range(self.config.n_layers))

        # Track last convergence signals for halt head (conv_head mode only)
        conv_delta_token = None   # (B, T) predicted delta

        layer_pass_counter = 0

        # --- SYSTEM 1 ---
        for n in range(self.system2_start):
            active_layers += 1
            hub_before = S[:, :, h_s:h_e].detach().clone()
            
            S, wr = self.blocks[n](S, self.workspace, kv_cache=kv_cache, cache_start_pos=cache_start_pos)
            write_records.append(wr)
            if wr.hub_shared_gate is not None:
                hub_churn_masks.append(wr.hub_shared_gate.detach())
            
            # Hub delta: how much did this layer change the hub?
            hub_after = S[:, :, h_s:h_e].detach()
            delta = (hub_after - hub_before).norm(dim=-1)  # (B, T)
            hub_delta_history.append((layer_pass_counter, delta))
            
            # Workspace state recording hook (no-op when self._recorder is None)
            if self._recorder is not None:
                self._recorder.record_layer(
                    layer_idx=n, layer_type='S1', ponder_step=0,
                    S=S, workspace=self.workspace, write_record=wr,
                    hub_delta_norm=delta.mean().item()
                )
            
            # Collect hub state for self-distillation (at probed layers only)
            if n in active_probes:
                distill_hub_states[layer_pass_counter] = S[:, :, h_s:h_e]  # ON the graph
            
            # Run convergence head (predicts delta, used for inference-time signals)
            pred_delta = self.convergence_head(S[:, :, h_s:h_e].detach())
            conv_predictions_list.append((layer_pass_counter, pred_delta))
            
            if use_conv_head:
                conv_delta_token = pred_delta.detach()
                probed = True
            else:
                probed = n in active_probes
                
            layer_pass_counter += 1

            if (enable_early_exit and n >= self.config.min_layers
                    and not self.training and probed):
                # Early exit based on delta norm
                delta_mean = delta.mean().item()
                if delta_mean < 0.1:
                    sys1_exited = True
                    break

        s1_exit_loss = torch.tensor(0.0, device=S.device)
        if not sys1_exited:
            # S1 Exit Loss: force S1 to produce standalone predictions
            def _s1_exit_decoder(s_final):
                collapsed = self.collapse(s_final, self.workspace)
                return self.output_decoder(collapsed)
                
            if self.training and target_ids is not None:
                s1_logits = ckpt_fn(_s1_exit_decoder, S, use_reentrant=False)
                s1_exit_loss = F.cross_entropy(
                    s1_logits.view(-1, s1_logits.size(-1)),
                    target_ids.view(-1)
                )
            else:
                with torch.no_grad():
                    s1_logits = _s1_exit_decoder(S)
            s1_preds = s1_logits.detach().argmax(dim=-1)  # ADD THIS for veto rate

        if not sys1_exited:
            if do_ponder and max_steps > 0:
                # === ADAPTIVE PONDERING ===
                cumulative_halt = torch.zeros(B, T, 1, device=S.device)
                running_S = torch.zeros_like(S)

                for step in range(max_steps + 1):
                    for n in range(self.system2_start, self.config.n_layers):
                        active_layers += 1
                        hub_before = S[:, :, h_s:h_e].detach().clone()
                        
                        S, wr = self.blocks[n](S, self.workspace, kv_cache=kv_cache, cache_start_pos=cache_start_pos)
                        write_records.append(wr)
                        if wr.hub_shared_gate is not None:
                            hub_churn_masks.append(wr.hub_shared_gate.detach())

                        # Hub delta: how much did this layer change the hub?
                        hub_after = S[:, :, h_s:h_e].detach()
                        delta = (hub_after - hub_before).norm(dim=-1)  # (B, T)
                        hub_delta_history.append((layer_pass_counter, delta))
                        
                        # Workspace state recording hook
                        if self._recorder is not None:
                            self._recorder.record_layer(
                                layer_idx=n, layer_type='S2', ponder_step=step,
                                S=S, workspace=self.workspace, write_record=wr,
                                hub_delta_norm=delta.mean().item()
                            )
                        
                        # Collect hub state for self-distillation (at probed layers only)
                        if n in active_probes:
                            distill_hub_states[layer_pass_counter] = S[:, :, h_s:h_e]  # ON the graph
                        
                        # Run convergence head (predicts delta, used for inference-time signals)
                        pred_delta = self.convergence_head(S[:, :, h_s:h_e].detach())
                        conv_predictions_list.append((layer_pass_counter, pred_delta))
                        
                        if use_conv_head:
                            delta_mean = pred_delta.mean().item()
                            conv_delta_token = pred_delta.detach()
                            probed = True
                        else:
                            probed = n in active_probes
                            
                        layer_pass_counter += 1

                    if step < max_steps:
                        # Compute per-token epistemic signals for halt decision
                        with torch.no_grad():
                            halt_hub = S[:, :, h_s:h_e].detach()
                        p_halt = self.halt_head(halt_hub)
                        p_continue = 1.0 - cumulative_halt
                        p_this_step = p_halt * p_continue
                        running_S = running_S + p_this_step * S
                        cumulative_halt = cumulative_halt + p_this_step
                        halt_probs_list.append(p_this_step)

                        if not self.training and cumulative_halt.mean() > self.config.ponder_halt_threshold:
                            remainder = 1.0 - cumulative_halt
                            running_S = running_S + remainder * S
                            ponder_steps = step
                            break
                    else:
                        remainder = 1.0 - cumulative_halt
                        running_S = running_S + remainder * S
                        ponder_steps = step

                S = running_S

            else:
                # === SINGLE S2 PASS ===
                for n in range(self.system2_start, self.config.n_layers):
                    active_layers += 1
                    hub_before = S[:, :, h_s:h_e].detach().clone()
                    
                    S, wr = self.blocks[n](S, self.workspace, kv_cache=kv_cache, cache_start_pos=cache_start_pos)
                    write_records.append(wr)
                    if wr.hub_shared_gate is not None:
                        hub_churn_masks.append(wr.hub_shared_gate.detach())

                    # Hub delta: how much did this layer change the hub?
                    hub_after = S[:, :, h_s:h_e].detach()
                    delta = (hub_after - hub_before).norm(dim=-1)  # (B, T)
                    hub_delta_history.append((layer_pass_counter, delta))
                    
                    # Workspace state recording hook
                    if self._recorder is not None:
                        self._recorder.record_layer(
                            layer_idx=n, layer_type='S2', ponder_step=0,
                            S=S, workspace=self.workspace, write_record=wr,
                            hub_delta_norm=delta.mean().item()
                        )
                    
                    # Collect hub state for self-distillation (at probed layers only)
                    if n in active_probes:
                        distill_hub_states[layer_pass_counter] = S[:, :, h_s:h_e]  # ON the graph
                    
                    # Run convergence head (predicts delta, used for inference-time signals)
                    pred_delta = self.convergence_head(S[:, :, h_s:h_e].detach())
                    conv_predictions_list.append((layer_pass_counter, pred_delta))
                    
                    if use_conv_head:
                        delta_mean = pred_delta.mean().item()
                        conv_delta_token = pred_delta.detach()
                        probed = True
                    else:
                        probed = n in active_probes
                        
                    layer_pass_counter += 1

        # --- HUB SELF-DISTILLATION ---
        distill_loss = torch.tensor(0.0, device=S.device)
        distill_count = 0

        if distill_hub_states and self.training and target_ids is not None:
            hub_final = S[:, :, h_s:h_e].detach()  # Final hub state — DETACHED target
            
            for layer_idx, hub_intermediate in distill_hub_states.items():
                # hub_intermediate is ON the graph, hub_final is detached
                if self.training:
                    distill_loss = distill_loss + ckpt_fn(
                        self.hub_distill, hub_intermediate, hub_final,
                        use_reentrant=False
                    )
                else:
                    distill_loss = distill_loss + self.hub_distill(hub_intermediate, hub_final)
                distill_count += 1
            
            if distill_count > 0:
                distill_loss = distill_loss / distill_count

        # --- CONVERGENCE HEAD DISTILLATION ---
        conv_distill_loss_accum = torch.tensor(0.0, device=S.device)
        conv_distill_count = 0

        for layer_idx, pred_delta in conv_predictions_list:
            # Find matching actual delta
            for d_idx, actual_delta in hub_delta_history:
                if d_idx == layer_idx:
                    conv_distill_loss_accum = conv_distill_loss_accum + convergence_distillation_loss(
                        pred_delta, actual_delta
                    )
                    conv_distill_count += 1
                    break

        if conv_distill_count > 0:
            conv_distill_loss_accum = conv_distill_loss_accum / conv_distill_count

        # --- TELEMETRY & FEEDBACK ---
        # Update delta normalizer with current batch statistics
        if self.training and hub_delta_history:
            all_deltas = torch.stack([d for _, d in hub_delta_history])
            self.delta_normalizer.update(all_deltas)

        epistemic = compute_epistemic_state_from_deltas(hub_delta_history, S, self.delta_normalizer)
        epistemic.ponder_steps = ponder_steps

        if hub_churn_masks:
            epistemic.hub_churn = torch.stack(hub_churn_masks).mean()
        else:
            epistemic.hub_churn = torch.tensor(1.0, device=S.device)

        hs_s, hs_e = self.workspace.get_hub_shared_slice()
        epistemic.workspace_saturation = (S[:, :, hs_s:hs_e].abs() > 0.01).float().mean()

        collapsed = self.collapse(S, self.workspace)

        workload = torch.full((B, T, 1), float(ponder_steps), device=S.device)
        dashboard = torch.cat([
            epistemic.token_non_convergence.detach(),
            epistemic.token_entropy.detach(),
            workload
        ], dim=-1)

        gate = torch.sigmoid(self.epistemic_gate(dashboard))
        if not self.training:
            epistemic.gate_mean = gate.mean().item()
            epistemic.gate_std = gate.std().item()
        else:
            epistemic.gate_mean = 0.0
            epistemic.gate_std = 0.0
        logits = self.output_decoder(collapsed * gate)

        # Veto rate: S1 decoder output vs final decoder output
        if ponder_steps > 0 and 's1_preds' in locals() and s1_preds is not None:
            final_preds = logits.detach().argmax(dim=-1)
            epistemic.veto_rate = (s1_preds != final_preds).float().mean()
        else:
            epistemic.veto_rate = torch.tensor(0.0, device=S.device)

        final_conv_distill_loss = conv_distill_loss_accum if conv_distill_count > 0 else torch.tensor(0.0, device=S.device)

        return ModelOutput(
            logits, active_layers, epistemic, write_records,
            S, hub_delta_history, distill_loss, halt_probs_list,
            conv_distill_loss=final_conv_distill_loss,
            conv_predictions=conv_predictions_list,
            s1_exit_loss=s1_exit_loss if 's1_exit_loss' in locals() else None
        )


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════════

class PonderLoss(nn.Module):
    """KL divergence between halt distribution and geometric prior.
    Encourages early halting. lambda_p controls expected steps:
        0.5 → ~2 steps, 0.3 → ~3.3 steps, 0.2 → ~5 steps
    """
    def __init__(self, lambda_p=0.3, max_steps=5):
        super().__init__()
        prior = torch.zeros(max_steps + 1)
        for n in range(max_steps + 1):
            prior[n] = (1 - lambda_p) ** n * lambda_p
        prior[-1] += 1.0 - prior.sum()
        self.register_buffer('prior', prior)

    def forward(self, halt_probs):
        if not halt_probs:
            return torch.tensor(0.0, device=self.prior.device)
        # halt_probs: list of (B, T, 1), length = max_ponder_steps
        q_steps = torch.stack(halt_probs, dim=0).squeeze(-1).mean(dim=(1, 2))
        remainder = (1.0 - q_steps.sum()).clamp(min=1e-9).unsqueeze(0)
        q = torch.cat([q_steps, remainder])
        q = q + 1e-9
        p = self.prior[:len(q)].to(q.device)
        p = p / p.sum()
        q = q / q.sum()
        return (q * (q.log() - p.log())).sum()


class HybridLoss(nn.Module):
    def __init__(self, config, depth_reg_weight=0.01, total_steps=20000):
        super().__init__()
        self.config = config
        self.depth_reg_weight = depth_reg_weight
        self.probe_loss_weight = config.probe_loss_weight
        self.ponder_loss_weight = config.ponder_loss_weight
        self.conv_distill_weight = config.conv_distill_weight
        self.total_steps = total_steps
        self.current_step = 0
        self.probe_loss_scale = 1.0
        self.ponder_loss_scale = 1.0
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.ponder_loss = PonderLoss(config.ponder_lambda, config.max_ponder_steps)

    def set_step(self, step): self.current_step = step

    def set_phase(self, phase_settings):
        self.probe_loss_scale = phase_settings.get("probe_loss_scale", 1.0)
        self.ponder_loss_scale = phase_settings.get("ponder_loss_scale", 1.0)
        self.s1_exit_scale = phase_settings.get("s1_exit_scale", 1.0)

    def forward(self, output, target_ids):
        with record_function("cwt::loss_lm"):
            lm_loss = self.ce_loss(output.logits.view(-1, output.logits.size(-1)), target_ids.view(-1))

        with record_function("cwt::loss_depth"):
            mean_mags = torch.stack([wr.magnitude.mean() for wr in output.write_records])
            anneal = min(1.0, self.current_step / (0.6 * self.total_steps + 1e-9))
            depth_reg_loss = mean_mags.mean() + (1.0 - anneal) * mean_mags.var()

        with record_function("cwt::loss_distill"):
            distill_loss = output.distill_loss if output.distill_loss is not None else torch.tensor(0.0, device=output.logits.device)

        with record_function("cwt::loss_ponder"):
            p_loss = torch.tensor(0.0, device=output.logits.device)
            if output.halt_probs and len(output.halt_probs) > 0:
                p_loss = self.ponder_loss(output.halt_probs)

        with record_function("cwt::loss_conv"):
            conv_loss = output.conv_distill_loss if output.conv_distill_loss is not None else torch.tensor(0.0, device=output.logits.device)

        with record_function("cwt::loss_total"):
            s1_exit_loss = getattr(output, 's1_exit_loss', None)
            if s1_exit_loss is None:
                s1_exit_loss = torch.tensor(0.0, device=output.logits.device)
            s1_exit_weight = getattr(self.config, 's1_exit_weight', 0.1)
            s1_exit_scale = getattr(self, 's1_exit_scale', 1.0)
            
            total_loss = (lm_loss
                          + self.depth_reg_weight * depth_reg_loss
                          + self.probe_loss_weight * self.probe_loss_scale * distill_loss
                          + self.ponder_loss_weight * self.ponder_loss_scale * p_loss
                          + self.conv_distill_weight * conv_loss
                          + s1_exit_weight * s1_exit_scale * s1_exit_loss)

        # Return detached scalar tensors instead of .item() floats to avoid
        # 12 CUDA sync barriers per micro-step. TrainLogger._safe() converts
        # tensors to floats at log time (only every log_interval steps).
        return total_loss, {
            "lm_loss": lm_loss.detach(), "depth_reg_loss": depth_reg_loss.detach(),
            "distill_loss": distill_loss.detach(), "ponder_loss": p_loss.detach(),
            "conv_distill_loss": conv_loss.detach(),
            "s1_exit_loss": s1_exit_loss.detach(),
            "total_loss": total_loss.detach(), "active_layers": output.active_layers,
            # Weighted contributions to total_loss (loss budget)
            "budget_lm": lm_loss.detach(),
            "budget_depth_reg": (self.depth_reg_weight * depth_reg_loss).detach(),
            "budget_distill": (self.probe_loss_weight * self.probe_loss_scale * distill_loss).detach(),
            "budget_ponder": (self.ponder_loss_weight * self.ponder_loss_scale * p_loss).detach(),
            "budget_conv": (self.conv_distill_weight * conv_loss).detach(),
            "budget_s1_exit": (s1_exit_weight * s1_exit_scale * s1_exit_loss).detach(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    counts = {}
    total = 0
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        counts[name] = f"{n / 1e6:.1f}M"
        total += n
    counts["total"] = f"{total / 1e6:.1f}M"
    return counts

def get_training_phase(step, total_steps):
    warmup_end = 2000
    ponder_ramp_end = 3500  # Extended from 3000 to 3500 (1500-step ramp, was 1000)

    if step < warmup_end:
        # Phase 1: No pondering, probe loss ramps from 0 to full
        probe_weight_scale = min(1.0, step / 500)
        return "phase_1_warmup", {
            "enable_pondering": False,
            "probe_loss_scale": probe_weight_scale,
            "ponder_loss_scale": 0.0,
            "s1_exit_scale": 0.0, # S1 exit loss off during warmup
        }
    else:
        # Phase 2: Pondering enabled, ponder loss ramps in over 1500 steps
        ponder_progress = min(1.0, (step - warmup_end) / (ponder_ramp_end - warmup_end))
        return "phase_2_adaptive", {
            "enable_pondering": True,
            "probe_loss_scale": 1.0,
            "ponder_loss_scale": ponder_progress,
            "s1_exit_scale": 1.0, # S1 exit loss on
        }


@torch.no_grad()
def calibrate_epistemic_thresholds(model, val_loader, device, n_batches=50):
    """Compute percentile-based epistemic thresholds from validation data.
    
    Run this once at the start of training (or periodically) to set
    thresholds appropriate for the current data distribution.
    
    Returns dict with 'convergent_max' and 'resolvable_max' keys.
    """
    model.eval()
    all_nc = []
    
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        else:
            input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, enable_early_exit=False)
            # Get non-convergence values from epistemic state
            nc = outputs.epistemic.token_non_convergence.squeeze(-1)  # (B, T)
            all_nc.append(nc.cpu())
    
    if not all_nc:
        print("  [calibrate] No data available, using defaults")
        return {
            'convergent_max': 0.1,
            'resolvable_max': 0.3,
        }
    
    nc = torch.cat(all_nc, dim=0).flatten()
    
    thresholds = {
        'convergent_max': nc.quantile(0.75).item(),
        'resolvable_max': nc.quantile(0.95).item(),
    }
    
    print(f"  Calibrated epistemic thresholds:")
    print(f"    Non-convergence distribution: mean={nc.mean():.4f}, std={nc.std():.4f}")
    print(f"    Convergent:    nc < {thresholds['convergent_max']:.4f} (~75% of tokens)")
    print(f"    Resolvable:    nc < {thresholds['resolvable_max']:.4f} (~95% of tokens)")
    print(f"    Unresolvable:  nc >= {thresholds['resolvable_max']:.4f} (~5% of tokens)")
    
    return thresholds


def smoke_test():
    print("=" * 60)
    print("SMOKE TEST — CWT v5.5")
    print("=" * 60)

    cfg = ModelConfig(
        vocab_size=1000, d_model=128, n_layers=4, n_heads=4, ffn_dim=512,
        d_spoke=32, d_hub_private=8, d_hub_shared=64, d_tag=8,
        n_system2_layers=1, enable_pondering=True, max_ponder_steps=5,
        ponder_lambda=0.3, ponder_loss_weight=0.01,
        decay_window=2, decay_gradient_floor=0.5, decay_gradient_floor_system2=0.85,
        max_seq_len=128, min_layers=2,
        probe_bottleneck=64, probe_layers=3, probe_loss_weight=0.1,
        d_kv_latent=32,
    )
    model = CognitiveWorkspaceTransformer(cfg)

    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    tgt = torch.randint(0, cfg.vocab_size, (2, 32))

    # --- Forward Pass ---
    model.train()
    out = model(ids, target_ids=tgt, enable_early_exit=False)
    print(f"\n  Logits: {out.logits.shape}")
    print(f"  Active layers: {out.active_layers}")
    print(f"  Distill loss: {out.distill_loss.item():.4f}")
    print(f"  Hub delta history length: {len(out.hub_delta_history)}")
    print(f"  Halt probs: {len(out.halt_probs)} entries")
    print(f"  Ponder steps: {out.epistemic.ponder_steps}")

    # --- Hub Self-Distillation Verification ---
    print("\n--- Hub Self-Distillation ---")
    hub_test = torch.randn(1, 1, cfg.d_hub_total)
    hub_target = torch.randn(1, 1, cfg.d_hub_total)
    distill_loss = model.hub_distill(hub_test, hub_target)
    assert distill_loss.dim() == 0, "Distillation loss must be a scalar!"
    distill_params = sum(p.numel() for p in model.hub_distill.parameters())
    print(f"  Distill params: {distill_params}")
    print(f"  Distill loss: {distill_loss.item():.4f}")

    # --- ConvergenceHeadV2 Verification ---
    print("\n--- ConvergenceHeadV2 ---")
    hub_test = torch.randn(2, 32, cfg.d_hub_total)
    model.convergence_head.reset()
    # First call: should return moderate delta
    pred_delta_1 = model.convergence_head(hub_test)
    assert pred_delta_1.shape == (2, 32), f"pred_delta shape wrong: {pred_delta_1.shape}"
    assert (pred_delta_1 > 0).all(), "Delta should be positive"
    print(f"  First layer delta: mean={pred_delta_1.mean():.4f} (expected ~0.5)")
    # Second call: should use learned MLP
    hub_test2 = torch.randn(2, 32, cfg.d_hub_total)
    pred_delta_2 = model.convergence_head(hub_test2)
    assert pred_delta_2.shape == (2, 32), f"pred_delta shape wrong: {pred_delta_2.shape}"
    assert (pred_delta_2 > 0).all(), "Delta should always be positive (softplus output)"
    print(f"  Second layer delta: mean={pred_delta_2.mean():.4f}")
    conv_params = sum(p.numel() for p in model.convergence_head.parameters())
    print(f"  ConvergenceHeadV2 params: {conv_params}")

    # --- Halt Head Verification ---
    print("\n--- Halt Head ---")
    assert len(out.halt_probs) == cfg.max_ponder_steps, \
        f"Expected {cfg.max_ponder_steps} halt probs, got {len(out.halt_probs)}"
    for i, hp in enumerate(out.halt_probs):
        print(f"  Step {i} halt prob mean: {hp.mean().item():.4f}")

    # --- Conv Predictions Verification ---
    print("\n--- Conv Predictions ---")
    assert out.conv_predictions is not None, "Should have conv_predictions"
    print(f"  Conv predictions: {len(out.conv_predictions)} layers")
    for entry in out.conv_predictions:
        if len(entry) == 2:
            layer_idx, delta = entry
            print(f"  Layer {layer_idx+1}: Delta={delta.mean().item():.4f}")
        else:
            delta = entry
            print(f"  Layer ?: Delta={delta.mean().item():.4f}")

    # --- Detach Verification ---
    print("\n--- Detach Verification ---")
    print(f"  Dashboard requires_grad: {out.epistemic.token_non_convergence.requires_grad} (expected False)")
    assert not out.epistemic.token_non_convergence.requires_grad

    # --- Soft Ponder Lock ---
    print("\n--- Soft Ponder Lock ---")
    sys1_bias = model.workspace.decay_gates[1].gate_proj[0].bias.mean().item()
    sys2_bias = model.workspace.decay_gates[3].gate_proj[0].bias.mean().item()
    print(f"  System 1 Decay Bias: {sys1_bias:.2f} (expected ~3.00)")
    print(f"  System 2 Decay Bias: {sys2_bias:.2f} (expected ~5.00)")
    assert sys2_bias > sys1_bias

    # --- Loss & Backward ---
    loss_fn = HybridLoss(cfg)
    total_loss, bd = loss_fn(out, tgt)
    print(f"\n  Total loss: {bd['total_loss']:.4f}")
    print(f"  Ponder loss: {bd['ponder_loss']:.4f}")
    print(f"  Conv distill loss: {bd['conv_distill_loss']:.4f}")
    print(f"  Budget conv: {bd['budget_conv']:.4f} (target: 0.02-0.04)")
    total_loss.backward()

    distill_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.hub_distill.parameters())
    halt_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.halt_head.parameters())
    print(f"  Distill has gradient: {distill_grad} (expected True)")
    print(f"  Halt head has gradient: {halt_grad} (expected True)")
    assert distill_grad, "Distill should receive gradients!"
    assert halt_grad, "Halt head should receive gradients!"

    conv_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.convergence_head.parameters())
    print(f"  Convergence head has gradient: {conv_grad} (expected True)")
    assert conv_grad, "Convergence head should receive gradients from distillation!"
    print(f"  Conv distill loss: {out.conv_distill_loss.item():.4f}")

    # --- STE Verification ---
    print("\n--- STE Verification ---")
    S = torch.randn(2, 4, 8, requires_grad=True)
    gate = torch.full((2, 4, 8), 0.1, requires_grad=True)
    result = floored_multiply(S, gate, 0.5)
    result.sum().backward()
    assert torch.allclose(S.grad, torch.full_like(S.grad, 0.5), atol=1e-6)
    print(f"  FlooredMultiply: CORRECT")

    # --- Weight Tying Verification ---
    print("\n--- Weight Tying ---")
    assert model.output_decoder.lm_head.weight is model.embedding.embed.weight, "Weights are not tied!"
    print("  lm_head and embed weights are tied ✓")

    # --- Shared RoPE Verification ---
    print("\n--- Shared RoPE ---")
    rope_id = id(model.shared_rope)
    all_shared = all(id(block.attention.rope) == rope_id for block in model.blocks)
    assert all_shared, "Not all blocks share the same RoPE instance!"
    print("  All blocks share the same RoPE instance ✓")

    # --- S1 Exit Loss Verification ---
    print("\n--- S1 Exit Loss ---")
    print(f"  S1 Exit Loss: {out.s1_exit_loss.item():.4f}")
    assert out.s1_exit_loss.item() > 0, "S1 exit loss should be > 0 during training"

    # --- Phase Gating Verification ---
    print("\n--- Phase Gating ---")
    phase1_name, phase1_settings = get_training_phase(1000, 20000)
    phase2_name, phase2_settings = get_training_phase(5000, 20000)
    assert phase1_settings["s1_exit_scale"] == 0.0, "S1 exit loss should be off in Phase 1"
    assert phase2_settings["s1_exit_scale"] == 1.0, "S1 exit loss should be on in Phase 2"
    print(f"  Phase 1 s1_exit_scale: {phase1_settings['s1_exit_scale']} ✓")
    print(f"  Phase 2 s1_exit_scale: {phase2_settings['s1_exit_scale']} ✓")

    # --- YaRN RoPE Verification ---
    print("\n--- YaRN RoPE ---")
    yarn_rope = RotaryPositionEmbedding(
        d_head=cfg.d_model // cfg.n_heads,
        max_seq_len=256,
        theta=cfg.rope_theta,
        scaling_factor=2.0,
        scaling_mode="yarn",
        original_max_seq_len=128
    )
    yarn_out = yarn_rope(torch.randn(2, 4, 32, cfg.d_model // cfg.n_heads))
    assert yarn_out.shape == (2, 4, 32, cfg.d_model // cfg.n_heads), "YaRN RoPE output shape mismatch"
    print("  YaRN RoPE instantiated and forward pass successful ✓")

    # --- Inference (early exit + pondering) ---
    model.eval()
    with torch.no_grad():
        out_eval = model(ids, enable_early_exit=True)
    print(f"\n  Inference active layers: {out_eval.active_layers}/{cfg.n_layers}")
    print(f"  Inference ponder steps: {out_eval.epistemic.ponder_steps}")
    if out_eval.halt_probs:
        print(f"  Inference halt probs used: {len(out_eval.halt_probs)}")

    # --- Reasoning Efforts Verification ---
    print("\n--- Reasoning Efforts ---")
    with torch.no_grad():
        out_low = model(ids, reasoning_effort="low")
        out_med = model(ids, reasoning_effort="medium")
        out_high = model(ids, reasoning_effort="high")
    print(f"  Low effort ponder steps: {out_low.epistemic.ponder_steps}")
    print(f"  Medium effort ponder steps: {out_med.epistemic.ponder_steps}")
    print(f"  High effort ponder steps: {out_high.epistemic.ponder_steps}")

    # --- Use Conv Head Inference ---
    print("\n--- Use Conv Head Inference ---")
    with torch.no_grad():
        out_conv = model(ids, use_conv_head=True)
    print(f"  Conv head inference successful, active layers: {out_conv.active_layers} ✓")

    # --- Generation with Cache Dynamics ---
    print("\n--- Generation with Cache Dynamics ---")
    gen_ids, gen_cache, gen_diagnostics = model.generate_cached(
        ids[:, :4], max_new_tokens=2, track_cache_dynamics=True
    )
    assert gen_ids.shape[1] == 6, "Generated sequence length mismatch"
    assert len(gen_diagnostics) == 2, "Diagnostics length mismatch"
    print(f"  Generated sequence shape: {gen_ids.shape} ✓")
    print(f"  Diagnostics collected for {len(gen_diagnostics)} tokens ✓")
    summary = gen_cache.get_dynamics_summary()
    if summary:
        print(f"  Cache dynamics summary keys: {list(summary.keys())} ✓")

    # --- Convergence distillation loss function test ---
    print("\n--- Convergence Distillation Loss ---")
    test_pred_delta = torch.tensor([0.1, 0.5, 0.2])
    test_actual_delta = torch.tensor([0.12, 0.45, 0.22])
    cdl = convergence_distillation_loss(test_pred_delta, test_actual_delta)
    assert not torch.isnan(cdl), "Distillation loss should not be NaN"
    assert cdl.item() > 0, "Distillation loss should be positive"
    print(f"  Loss value: {cdl.item():.6f} (should be small but positive)")

    # --- Epistemic threshold classification ---
    print("\n--- Epistemic Classification ---")
    assert classify_epistemic_state(0.05) == "convergent"
    assert classify_epistemic_state(0.2) == "divergent_resolvable"
    assert classify_epistemic_state(0.5) == "divergent_unresolvable"
    # With calibrated thresholds
    assert classify_epistemic_state(0.05, convergent_max=0.03) == "divergent_resolvable"
    assert classify_epistemic_state(0.05, convergent_max=0.1) == "convergent"
    print(f"  Default thresholds: PASS")
    print(f"  Calibrated thresholds: PASS")

    # --- CWT-MLA Verification ---
    print("\n--- CWT-MLA Attention ---")
    # Verify attention module interface
    attn = model.blocks[0].attention
    H_q_test = torch.randn(2, 32, cfg.d_model)
    hub_test = torch.randn(2, 32, cfg.d_hub_total)
    attn_out = attn(H_q_test, hub_test)
    assert attn_out.shape == (2, 32, cfg.d_model), f"Wrong output shape: {attn_out.shape}"
    print(f"  Attention output shape: {attn_out.shape} ✓")
    print(f"  KV latent dim: {attn.d_latent}")
    print(f"  Cache compression: {2 * cfg.d_model / attn.d_latent:.1f}× vs standard KV")
    kv_params = sum(p.numel() for p in [attn.kv_down, attn.kv_up]
                    for p in p.parameters()) if hasattr(attn.kv_down, 'parameters') else (
        attn.kv_down.weight.numel() + attn.kv_up.weight.numel()
    )
    print(f"  KV path params: {kv_params} (vs {2 * cfg.d_model * cfg.d_model} for standard K+V)")

    # --- Cache Verification ---
    print("\n--- Latent KV Cache ---")
    cache = CWTLatentCache(
        n_layers=cfg.n_layers, d_latent=cfg.d_kv_latent,
        max_seq_len=64, batch_size=2, device='cpu'
    )
    print(f"  {cache}")

    # Simulate prefill: 32 tokens
    model.eval()
    with torch.no_grad():
        prefill_out = model(ids, kv_cache=cache, cache_start_pos=0, enable_early_exit=False)
    print(f"  After prefill: {cache}")
    assert cache.current_position() > 0, "Cache should be populated after prefill"

    # Simulate one generation step
    next_token = torch.randint(0, cfg.vocab_size, (2, 1))
    gen_pos = cache.current_position()
    with torch.no_grad():
        gen_out = model(next_token, kv_cache=cache, cache_start_pos=gen_pos, enable_early_exit=False)
    assert gen_out.logits.shape == (2, 1, cfg.vocab_size), f"Wrong gen shape: {gen_out.logits.shape}"
    print(f"  After 1 gen step: {cache}")
    print(f"  Cache memory: {cache.memory_bytes() / 1024:.1f} KB")
    print(f"  Gen output shape: {gen_out.logits.shape} ✓")

    # --- Parameter Count ---
    print(f"\n--- Parameters ---")
    for name, count in count_parameters(model).items():
        print(f"  {name}: {count}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
