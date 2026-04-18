"""
Training Script for CWT v5.5

Changes from v5.3:
  - CWT-MLA: Hub-Derived Latent Attention (Q from full workspace, K/V from hub)
  - Ponder-aware Latent KV Cache (d_kv_latent per position)
  - Removed standard CausalSelfAttention
  - Explicit d_kv_latent in train configs

Prior (v5.3):
  - ConvergenceHeadV2 telemetry (conv_kl_layer_N, conv_entropy_layer_N)
  - max_ponder_steps 3→5, extended ponder ramp (1500 steps)
  - Conv distill weight 0.05→0.5 (10x)
  - Epistemic threshold calibration (percentile-based)
  - Ponder selectivity metrics (per-token depth variance)

# Fast — System 1 + single System 2 pass, no pondering
python train_v5.py --generate "Once upon a time" --resume ckpt.pt --pondering false

# Default — model decides depth up to trained max
python train_v5.py --generate "Once upon a time" --resume ckpt.pt

# Quick thinking — allow 1 extra pass at most
python train_v5.py --generate "what is 1+1?" --resume ckpt.pt --reasoning-effort low

# Full deliberation — allow all trained passes
python train_v5.py --generate "what is 1+1?" --resume ckpt.pt --reasoning-effort high

# Multi-GPU training via torchrun
torchrun --nproc_per_node=4 train_v5.py --config prod

"""
import os

# Set thread limits BEFORE any other imports (especially torch/numpy)
# to prevent OpenMP/MKL from spawning 128+ threads per process
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import json, time, argparse, math, contextlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast

from cwt_v5 import (
    ModelConfig, CognitiveWorkspaceTransformer, HybridLoss,
    count_parameters, get_training_phase, classify_epistemic_state,
    calibrate_epistemic_thresholds,
)


@dataclass
class TrainConfig:
    model: Optional[ModelConfig] = None
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"           # HF subset/config name (empty = default)
    dataset_split: str = "train[5000:]"
    val_split: str = "train[:5000]"               # Sliced from train (FineWeb-Edu has no val split)
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 512
    total_steps: int = 20000
    batch_size: int = 8
    grad_accum_steps: int = 8
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    depth_reg_weight: float = 0.01
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    eval_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "cwt-v5"
    run_name: str = ""
    max_train_examples: int = -1


def get_test_config():
    m = ModelConfig(
        vocab_size=50257, d_model=512, n_layers=8, n_heads=8, ffn_dim=1408,
        d_spoke=48, d_hub_private=16, d_hub_shared=256, d_tag=16,
        n_system2_layers=2,
        enable_pondering=True,          # Adaptive pondering ON
        max_ponder_steps=5,             # Up to 5 extra S2 passes (was 3 in v5.2)
        ponder_lambda=0.3,              # Geometric prior ~3 steps
        ponder_loss_weight=0.01,
        ponder_halt_threshold=0.95,
        decay_window=3, decay_bias_init=3.0,
        decay_gradient_floor=0.5, decay_gradient_floor_system2=0.85,
        max_seq_len=4096, exit_kl_threshold=0.01, min_layers=3,
        uncertainty_threshold=0.7,
        probe_bottleneck=128,
        probe_layers=5,                 # Stratified: last 5 always + 1 random
        probe_loss_weight=0.1,
        conv_distill_weight=0.001,
        d_kv_latent=128,                # CWT-MLA compressed KV latent dimension
        dropout=0.1, init_write_scale=0.5,
    )
    return TrainConfig(
        model=m, dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train[5000:]", val_split="train[:5000]",
        max_seq_len=4096,
        total_steps=20000,
        batch_size=3,               # Conservative — 4096 ctx + pondering on 24GB 3090
        grad_accum_steps=8,        # Effective batch = 3 × 8 × 4 GPU = 96
        lr=3e-4,
        warmup_steps=1000,          # 5% of total steps
        bf16=True, log_interval=25, eval_interval=500, eval_steps=100,
        save_interval=1000, use_wandb=True, run_name="cwt-v55-fineweb-edu",
        max_train_examples=8_000_000,
    )


def get_prod_config():
    m = ModelConfig(
        vocab_size=50257, d_model=768, n_layers=12, n_heads=12, ffn_dim=2048,
        d_spoke=48, d_hub_private=16, d_hub_shared=512, d_tag=16,
        n_system2_layers=2,
        enable_pondering=True,
        max_ponder_steps=5,
        ponder_lambda=0.3,
        ponder_loss_weight=0.01,
        ponder_halt_threshold=0.95,
        decay_window=3, decay_bias_init=3.0,
        decay_gradient_floor=0.5, decay_gradient_floor_system2=0.85,
        max_seq_len=4096, exit_kl_threshold=0.01, min_layers=3,
        uncertainty_threshold=0.7,
        probe_bottleneck=128,
        probe_layers=8,
        probe_loss_weight=0.1,
        d_kv_latent=128,            # CWT-MLA compressed KV latent dimension
        dropout=0.1, init_write_scale=0.5,
    )
    return TrainConfig(
        model=m, dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train[5000:]", val_split="train[:5000]",
        max_seq_len=4096,
        total_steps=50000,
        batch_size=2,           # Per GPU, conservative for 24GB
        grad_accum_steps=16,    # Effective batch = 2 × 16 × 4 GPU = 128
        lr=3e-4,
        warmup_steps=2000, bf16=True, log_interval=50, eval_interval=1000,
        save_interval=5000, use_wandb=True, run_name="cwt-v54-fineweb-edu-prod",
        max_train_examples=8_000_000,
    )


class TokenizedDataset(Dataset):
    CACHE_DIR = Path("data_cache")
    def __init__(self, dataset_name, split, tokenizer, max_seq_len, max_examples=-1, dataset_config=""):
        self.max_seq_len = max_seq_len
        ck = self._cache_key(dataset_name, dataset_config, split, tokenizer, max_seq_len, max_examples)
        cp = self.CACHE_DIR / f"{ck}.pt"
        if cp.exists():
            print(f"Loading cached data from {cp}...")
            self.data = torch.load(cp, weights_only=True)
        else:
            self.data = self._tokenize(dataset_name, dataset_config, split, tokenizer, max_seq_len, max_examples)
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(self.data, cp)

    @staticmethod
    def _cache_key(dn, dc, sp, tok, msl, me):
        import hashlib
        ti = getattr(tok, "name_or_path", "unk")
        h = hashlib.sha256(f"{dn}|{dc}|{sp}|{ti}|{msl}|{me}".encode()).hexdigest()[:16]
        safe_sp = sp.replace("[", "_").replace("]", "_").replace(":", "-")
        return f"{dn.replace('/', '_')}_{safe_sp}_seq{msl}_ex{me}_{h}"

    @staticmethod
    def _tokenize(dn, dc, sp, tok, msl, me):
        import tempfile, numpy as np
        from datasets import load_dataset
        CS = 5000
        load_args = (dn, dc) if dc else (dn,)
        ds = load_dataset(*load_args, split=sp)
        if me > 0: ds = ds.select(range(min(me, len(ds))))
        tc = "text" if "text" in ds.column_names else "story"
        ne, eos = len(ds), tok.eos_token_id
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        tp, tt = tmp.name, 0
        try:
            for st in range(0, ne, CS):
                en = min(st + CS, ne)
                texts = list(ds.select(range(st, en))[tc])
                be = tok(texts, add_special_tokens=False)["input_ids"]
                for ti_ids in be:
                    a = np.array(ti_ids + [eos], dtype=np.int64)
                    tmp.write(a.tobytes()); tt += len(a)
            tmp.close()
            nc = tt // (msl + 1); u = nc * (msl + 1)
            mm = np.memmap(tp, dtype=np.int64, mode="r", shape=(u,))
            d = torch.from_numpy(mm[:u].copy()).long().view(nc, msl + 1)
            del mm
            return d
        finally:
            try: os.unlink(tp)
            except OSError: pass

    def __len__(self): return len(self.data)
    def __getitem__(self, i): c = self.data[i]; return c[:-1], c[1:]


def get_lr(step, config):
    if step < config.warmup_steps: return config.lr * step / config.warmup_steps
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps))
    return config.min_lr + (config.lr - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


class TrainLogger:
    def __init__(self, config):
        self.log_dir = Path(config.log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_dir / "train.jsonl", "a")
        self.wandb = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project=config.wandb_project, name=config.run_name, config=asdict(config.model))
                self.wandb = wandb
            except Exception as e:
                print(f"  [warn] W&B init failed ({type(e).__name__}: {e}); continuing without W&B")

    @staticmethod
    def _safe(v):
        if isinstance(v, torch.Tensor): return v.item()
        if hasattr(v, 'item'): return v.item()
        if isinstance(v, float): return v if not (math.isnan(v) or math.isinf(v)) else None
        if isinstance(v, (int, str, bool, type(None))): return v
        return str(v)

    def log(self, step, metrics):
        metrics["step"] = step; metrics["timestamp"] = time.time()
        parts = [f"step {step:>6d}"]
        for k in ["lm_loss", "total_loss", "distill_loss", "ponder_loss", "conv_distill_loss", "s1_exit_loss", "lr",
                   "grad_norm", "grad_norm_conv", "grad_clip_ratio",
                   "active_layers", "ponder_steps", "epistemic_state",
                   "mean_uncertainty", "tokens_per_sec",
                   "workspace_saturation", "hub_churn_mean", "veto_rate"]:
            if k in metrics:
                parts.append(f"{k}={metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k}={metrics[k]}")
        print(" | ".join(parts))
        jm = {k: self._safe(v) for k, v in metrics.items()}
        try: self.log_file.write(json.dumps(jm) + "\n"); self.log_file.flush()
        except: pass
        if self.wandb: self.wandb.log(jm, step=step)

    def close(self):
        self.log_file.close()
        if self.wandb: self.wandb.finish()


def compute_diagnostics(output, workspace, epistemic_thresholds=None):
    d = {}
    mags = [wr.magnitude.mean().item() for wr in output.write_records]
    d["write_mag_min"] = min(mags)
    d["write_mag_max"] = max(mags)
    d["write_mag_ratio"] = max(mags) / (min(mags) + 1e-9)

    spoke_mags = [wr.spoke_norm.item() for wr in output.write_records]
    hub_priv_mags = [wr.hub_priv_norm.item() for wr in output.write_records]
    hub_shared_mags = [wr.hub_shared_norm.item() for wr in output.write_records]
    d["spoke_write_mag_mean"] = sum(spoke_mags) / len(spoke_mags)
    d["hub_priv_write_mag_mean"] = sum(hub_priv_mags) / len(hub_priv_mags)
    d["hub_shared_write_mag_mean"] = sum(hub_shared_mags) / len(hub_shared_mags)

    mu = output.epistemic.mean_entropy.item()
    d["mean_uncertainty"] = mu
    d["mean_non_convergence"] = output.epistemic.mean_non_convergence.item()

    # Use calibrated thresholds if available
    if epistemic_thresholds:
        d["epistemic_state"] = classify_epistemic_state(
            mu,
            convergent_max=epistemic_thresholds.get('convergent_max', 0.1),
            resolvable_max=epistemic_thresholds.get('resolvable_max', 0.3),
        )
    else:
        d["epistemic_state"] = classify_epistemic_state(mu)

    d["ponder_steps"] = output.epistemic.ponder_steps
    d["workspace_saturation"] = output.epistemic.workspace_saturation.item()
    d["hub_churn_mean"] = output.epistemic.hub_churn.item()
    d["veto_rate"] = output.epistemic.veto_rate.item()
    d["gate_mean"] = output.epistemic.gate_mean
    d["gate_std"] = output.epistemic.gate_std

    tu = output.epistemic.token_entropy.detach()
    d["token_uncertainty_min"] = tu.min().item()
    d["token_uncertainty_max"] = tu.max().item()
    d["token_uncertainty_std"] = tu.std().item()

    if output.hub_delta_history:
        for entry in output.hub_delta_history:
            if isinstance(entry, tuple):
                layer_idx, delta = entry
                d[f"hub_delta_layer_{layer_idx+1}"] = delta.mean().item()

    # Halt head telemetry (supports any max_ponder_steps)
    if output.halt_probs:
        for i, hp in enumerate(output.halt_probs):
            d[f"halt_prob_step_{i}"] = hp.mean().item()
        # Expected ponder steps per token
        expected = sum(i * hp.mean().item() for i, hp in enumerate(output.halt_probs))
        remainder = 1.0 - sum(hp.mean().item() for hp in output.halt_probs)
        expected += len(output.halt_probs) * max(0, remainder)
        d["expected_ponder_steps"] = expected

        # Ponder selectivity: per-token expected steps variance
        # Higher std means model differentiates token difficulty
        hp_stack = torch.stack(output.halt_probs, dim=0).squeeze(-1)  # (steps, B, T)
        step_weights = torch.arange(len(output.halt_probs), device=hp_stack.device, dtype=hp_stack.dtype).view(-1, 1, 1)
        remainder_t = (1.0 - hp_stack.sum(dim=0)).clamp(min=0)  # (B, T)
        expected_per_token = (hp_stack * step_weights).sum(dim=0) + len(output.halt_probs) * remainder_t  # (B, T)
        d["ponder_steps_mean"] = expected_per_token.mean().item()
        d["ponder_steps_std"] = expected_per_token.std().item()
        d["ponder_steps_min"] = expected_per_token.min().item()
        d["ponder_steps_max"] = expected_per_token.max().item()

        # Fraction of tokens at each depth
        max_ponder = len(output.halt_probs)
        for s in range(max_ponder + 1):
            if s < max_ponder:
                frac = ((expected_per_token >= s) & (expected_per_token < s + 1)).float().mean().item()
            else:
                frac = (expected_per_token >= s).float().mean().item()
            d[f"ponder_depth_frac_{s}"] = frac

    # ConvergenceHeadV2 telemetry (conv_delta_layer_N)
    if output.conv_predictions:
        conv_by_layer = {}
        for entry in output.conv_predictions:
            if len(entry) == 2:
                layer_idx, delta = entry
                d[f"conv_delta_layer_{layer_idx+1}"] = delta.mean().item()
                conv_by_layer[layer_idx] = delta.mean().item()

        # Probe-convergence correlation tracking
        if output.hub_delta_history and conv_by_layer:
            actual_by_layer = {}
            for entry in output.hub_delta_history:
                if isinstance(entry, tuple):
                    layer_idx, delta = entry
                    actual_by_layer[layer_idx] = delta.mean().item()
            
            matched_layers = set(actual_by_layer.keys()) & set(conv_by_layer.keys())
            if len(matched_layers) > 2:
                sorted_layers = sorted(matched_layers)
                actual_vals = [actual_by_layer[l] for l in sorted_layers]
                conv_vals = [conv_by_layer[l] for l in sorted_layers]
                
                actual_t = torch.tensor(actual_vals)
                conv_t = torch.tensor(conv_vals)
                
                if actual_t.std() > 1e-8 and conv_t.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([conv_t, actual_t]))[0, 1].item()
                    d["conv_delta_correlation"] = corr

    return d


def make_sparkline(data, fixed_min=None, fixed_max=None):
    if not data: return ""
    bars = " ▂▃▄▅▆▇█"
    mn = fixed_min if fixed_min is not None else min(data)
    mx = fixed_max if fixed_max is not None else max(data)
    if mx <= mn: return bars[0] * len(data)
    return "".join(bars[max(0, min(7, int(7 * (x - mn) / (mx - mn))))] for x in data)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device, dtype, use_amp, eval_steps, pondering=None):
    model.eval()
    tl = tlm = tlay = pond = sat = vr = n = 0
    for i, (inp, tgt) in enumerate(val_loader):
        if i >= eval_steps: break
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            out = model(inp, target_ids=tgt, enable_early_exit=True, pondering=pondering)
            _, bd = loss_fn(out, tgt)
        tl += bd["total_loss"]; tlm += bd["lm_loss"]; tlay += out.active_layers
        pond += out.epistemic.ponder_steps
        sat += out.epistemic.workspace_saturation.item()
        vr += out.epistemic.veto_rate.item()
        n += 1
    n = max(n, 1)
    ppl = math.exp(min(tlm / n, 20))
    metrics = {"loss": tl/n, "lm_loss": tlm/n, "perplexity": ppl,
               "avg_active_layers": tlay/n, "avg_ponder_steps": pond/n,
               "avg_saturation": sat/n, "avg_veto_rate": vr/n}
    print(f"  [eval] loss={metrics['loss']:.4f} ppl={ppl:.1f} layers={metrics['avg_active_layers']:.1f} ponder={metrics['avg_ponder_steps']:.2f} veto={metrics['avg_veto_rate']:.1%}")
    model.train()
    torch.cuda.empty_cache()
    return metrics


def save_ckpt(model, optimizer, scaler, step, config, ckpt_dir):
    p = ckpt_dir / f"step_{step}.pt"
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "config": asdict(config.model)}, p)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda x: int(x.stem.split("_")[1]))
    while len(ckpts) > 10: ckpts[0].unlink(); ckpts.pop(0)


def train(config, resume_path=None):
    torch.set_num_threads(min(os.cpu_count() or 4, 8))

    seed = int(os.environ.get("CWT_SEED", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # Optimize cuDNN for fixed input shapes
    import random
    random.seed(seed)

    # --- DDP Initialization ---
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    is_rank0 = (not is_ddp) or (local_rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_rank0:
        print(f"Device: {device}" + (f"  (DDP world_size={dist.get_world_size()})" if is_ddp else ""))
    dtype = torch.bfloat16 if (device.type == "cuda" and config.bf16 and torch.cuda.is_bf16_supported()) else torch.float32
    use_amp = dtype != torch.float32

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config.model.vocab_size = len(tokenizer)

    model = CognitiveWorkspaceTransformer(config.model).to(device)

    # --- DDP wrapping (replaces DataParallel) ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if is_ddp else model

    if is_rank0:
        for n, c in count_parameters(raw_model).items(): print(f"  {n}: {c}")

    nw = min(4, os.cpu_count() or 2)
    train_ds = TokenizedDataset(config.dataset_name, config.dataset_split, tokenizer, config.max_seq_len,
                                max_examples=config.max_train_examples, dataset_config=config.dataset_config)

    # --- DistributedSampler for train ---
    train_sampler = DistributedSampler(train_ds) if is_ddp else None
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=nw > 0)

    val_loader = None
    val_sampler = None
    try:
        val_ds = TokenizedDataset(config.dataset_name, config.val_split, tokenizer, config.max_seq_len,
                                  max_examples=5000, dataset_config=config.dataset_config)
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                                sampler=val_sampler,
                                num_workers=nw, pin_memory=True, drop_last=True,
                                persistent_workers=nw > 0)
    except Exception: pass

    dp, ndp = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (ndp if any(nd in name for nd in ["norm", "bias", "embed", "identity", "designator"]) else dp).append(p)
    optimizer = torch.optim.AdamW([
        {"params": dp, "weight_decay": config.weight_decay},
        {"params": ndp, "weight_decay": 0.0},
    ], lr=config.lr, betas=(0.9, 0.95), fused=device.type == "cuda")

    loss_fn = HybridLoss(config.model, depth_reg_weight=config.depth_reg_weight, total_steps=config.total_steps)
    scaler = GradScaler(device=device.type, enabled=(use_amp and dtype == torch.float16))

    start_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except: pass
        start_step = ckpt.get("step", 0)

    logger = TrainLogger(config) if is_rank0 else None
    ckpt_dir = Path(config.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    train_iter = iter(train_loader)
    optim_step, micro_step = start_step, 0
    accum_metrics, tokens_seen, t0 = {}, 0, time.time()
    current_epoch = 0
    epistemic_thresholds = None  # Will be calibrated after Phase 1

    # Cache parameter lists for separate grad clipping (convergence head vs main model)
    _conv_params = list(raw_model.convergence_head.parameters())
    _conv_param_ids = {id(p) for p in _conv_params}
    _main_params = [p for p in model.parameters() if id(p) not in _conv_param_ids]

    # --- Run Info Banner ---
    if is_rank0:
        world = dist.get_world_size() if is_ddp else 1
        eff_batch = config.batch_size * config.grad_accum_steps * world
        m = config.model
        print(f"\n{'='*60}")
        print(f"  CWT v5.4 — {config.run_name or 'unnamed run'}")
        print(f"{'='*60}")
        print(f"  Dataset        : {config.dataset_name} ({config.dataset_split})")
        print(f"  Train examples : {len(train_ds):,}  |  Val examples : {len(val_ds) if val_loader else 'N/A':,}")
        print(f"  Seq length     : {config.max_seq_len}")
        print(f"{'─'*60}")
        print(f"  Model          : d={m.d_model}  layers={m.n_layers}  heads={m.n_heads}  ffn={m.ffn_dim}")
        print(f"  CWT-MLA        : d_kv_latent={m.d_kv_latent}  compression={(2 * m.d_model / m.d_kv_latent):.0f}×")
        print(f"  Workspace      : spoke={m.d_spoke}  hub_priv={m.d_hub_private}  hub_shared={m.d_hub_shared}  tag={m.d_tag}")
        print(f"  System 2       : layers={m.n_system2_layers}  pondering={'ON' if m.enable_pondering else 'OFF'}"
               + (f"  max_steps={m.max_ponder_steps}  λ={m.ponder_lambda}" if m.enable_pondering else ""))
        print(f"  Distill        : layers={m.probe_layers}  bottleneck={m.probe_bottleneck}  weight={m.probe_loss_weight}")
        print(f"  Conv distill   : weight={m.conv_distill_weight}")
        print(f"  S1 Exit Loss   : weight={getattr(m, 's1_exit_weight', 0.1)}")
        print(f"  Early exit     : kl_thresh={m.exit_kl_threshold}  min_layers={m.min_layers}")
        print(f"{'─'*60}")
        print(f"  Steps          : {start_step} → {config.total_steps}  ({config.total_steps - start_step} remaining)")
        print(f"  Batch          : {config.batch_size}/GPU × {config.grad_accum_steps} accum × {world} GPU = {eff_batch} effective")
        print(f"  LR             : {config.lr} → {config.min_lr}  (warmup {config.warmup_steps} steps, cosine)")
        print(f"  AMP            : {dtype}  |  GradScaler: {'ON' if scaler.is_enabled() else 'OFF'}")
        print(f"  DDP            : {'ON (no_sync enabled)' if is_ddp else 'OFF (single GPU)'}")
        print(f"{'─'*60}")
        wandb_status = 'ON' if (logger and logger.wandb) else ('FAILED' if config.use_wandb else 'OFF')
        print(f"  Logging        : every {config.log_interval} steps  |  W&B: {wandb_status}")
        print(f"  Eval           : every {config.eval_interval} steps  ({config.eval_steps} batches)")
        print(f"  Checkpoints    : every {config.save_interval} steps  → {ckpt_dir}/  (keep last 10)")
        if resume_path:
            print(f"  Resumed from   : {resume_path} (step {start_step})")
        print(f"{'='*60}\n")

    while optim_step < config.total_steps:
        try: inp, tgt = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            inp, tgt = next(train_iter)
        inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

        phase_name, phase_settings = get_training_phase(optim_step, config.total_steps)
        loss_fn.set_step(optim_step)
        loss_fn.set_phase(phase_settings)

        # Only sync gradients on the final accumulation step
        is_accumulating = ((micro_step + 1) % config.grad_accum_steps != 0)
        sync_context = model.no_sync() if (is_ddp and is_accumulating) else contextlib.nullcontext()

        with sync_context:
            with autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                output = model(inp, target_ids=tgt, enable_early_exit=False,
                               pondering=phase_settings.get("enable_pondering", True))
                total_loss, bd = loss_fn(output, tgt)
                loss = total_loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            del total_loss, loss  # Free backward graph refs immediately

        tokens_seen += inp.numel()
        for k, v in bd.items(): accum_metrics[k] = v
        micro_step += 1

        is_final_accum = (micro_step % config.grad_accum_steps == 0)
        if not is_final_accum:
            del output, bd  # Free immediately — not needed for diagnostics

        if is_final_accum:
            optim_step += 1
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(_main_params, config.max_grad_norm)
            gn_conv = torch.nn.utils.clip_grad_norm_(_conv_params, 1.0)
            lr = get_lr(optim_step, config)
            for pg in optimizer.param_groups: pg["lr"] = lr
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

            if is_rank0 and optim_step % config.log_interval == 0:
                elapsed = time.time() - t0
                metrics = {**accum_metrics, "lr": lr, "grad_norm": gn, "grad_norm_conv": gn_conv,
                           "grad_clip_ratio": gn / config.max_grad_norm,
                           "tokens_per_sec": tokens_seen / (elapsed + 1e-9),
                           "phase": phase_name, "total_steps": config.total_steps}
                with torch.no_grad(): metrics.update(compute_diagnostics(output, raw_model.workspace, epistemic_thresholds))
                logger.log(optim_step, metrics)
                accum_metrics, tokens_seen, t0 = {}, 0, time.time()

            # Free forward pass output after all uses (diagnostics, etc.)
            del output, bd

            if is_rank0 and val_loader and optim_step % config.eval_interval == 0:
                current_pondering = phase_settings.get("enable_pondering", True)
                em = evaluate(raw_model, val_loader, loss_fn, device, dtype, use_amp, config.eval_steps, pondering=current_pondering)
                logger.log(optim_step, {f"val_{k}": v for k, v in em.items()})

                # Dual eval: also measure without pondering once pondering is active
                if current_pondering:
                    em_no_ponder = evaluate(raw_model, val_loader, loss_fn, device, dtype, use_amp, config.eval_steps, pondering=False)
                    logger.log(optim_step, {f"val_no_ponder_{k}": v for k, v in em_no_ponder.items()})

                # One diagnostic pass for convergence head correlation + ponder selectivity
                try:
                    with torch.no_grad():
                        sample_inp, sample_tgt = next(iter(val_loader))
                        sample_inp = sample_inp.to(device)
                        sample_tgt = sample_tgt.to(device)
                        with autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                            diag_out = raw_model(sample_inp, target_ids=sample_tgt,
                                                 enable_early_exit=False, pondering=current_pondering)
                        diag_metrics = compute_diagnostics(diag_out, raw_model.workspace, epistemic_thresholds)
                        val_diag = {}
                        if "conv_delta_correlation" in diag_metrics:
                            val_diag["val_conv_delta_correlation"] = diag_metrics["conv_delta_correlation"]
                        if "ponder_steps_std" in diag_metrics:
                            val_diag["val_ponder_steps_std"] = diag_metrics["ponder_steps_std"]
                            val_diag["val_ponder_steps_mean"] = diag_metrics["ponder_steps_mean"]
                        if val_diag:
                            logger.log(optim_step, val_diag)
                        del diag_out, diag_metrics
                except Exception:
                    pass  # Don't crash training over a diagnostic

                # Epistemic threshold calibration:
                # Run after Phase 1 ends (step 2000) and every 5000 steps thereafter
                if optim_step >= 2000 and (epistemic_thresholds is None or optim_step % 5000 == 0):
                    try:
                        epistemic_thresholds = calibrate_epistemic_thresholds(
                            raw_model, val_loader, device, n_batches=30
                        )
                        # Store thresholds in model config for downstream use
                        raw_model.config.epistemic_convergent_max = epistemic_thresholds['convergent_max']
                        raw_model.config.epistemic_resolvable_max = epistemic_thresholds['resolvable_max']
                        logger.log(optim_step, {
                            "calibrated_convergent_max": epistemic_thresholds['convergent_max'],
                            "calibrated_resolvable_max": epistemic_thresholds['resolvable_max'],
                        })
                    except Exception as e:
                        print(f"  [warn] Threshold calibration failed: {e}")

                model.train()

            if is_rank0 and optim_step % config.save_interval == 0:
                save_ckpt(raw_model, optimizer, scaler, optim_step, config, ckpt_dir)

    if is_rank0:
        save_ckpt(raw_model, optimizer, scaler, optim_step, config, ckpt_dir)
        logger.close()

    if is_ddp:
        dist.destroy_process_group()

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=50, device="cuda", use_bf16=True, pondering=None, reasoning_effort="high"):
    import time as time_mod
    model.eval()
    inp = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen = inp.clone()
    amp = use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp)

    diag = {"tokens": [], "active_layers": [], "mean_uncertainties": [],
            "prediction_changes": [], "ponder_steps": [], "write_magnitudes": [],
            "hub_shared_write_mags": [], "halt_probs": [],
            "workspace_saturation": [], "hub_churn": [], "veto_rate": [],
            "gate_mean": [], "gate_std": [], "mean_non_convergence": [],
            "epistemic_state": []}
    start_time = time_mod.time()

    # Use cached generation (returns per-token diagnostics)
    with ctx:
        full_seq, cache, token_diagnostics = model.generate_cached(
            inp, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            pondering=pondering, reasoning_effort=reasoning_effort,
            track_cache_dynamics=True
        )
    
    # Unpack per-token diagnostics from each forward pass
    generated_ids = full_seq[0, inp.shape[1]:]
    for i, token_id in enumerate(generated_ids):
        tok_str = tokenizer.decode(token_id)
        diag["tokens"].append(tok_str)
        if i < len(token_diagnostics):
            td = token_diagnostics[i]
            diag["active_layers"].append(td["active_layers"])
            diag["ponder_steps"].append(td["ponder_steps"])
            diag["write_magnitudes"].append(td["write_magnitude"])
            diag["hub_shared_write_mags"].append(td["hub_shared_write_mag"])
            diag["mean_uncertainties"].append(td["mean_uncertainty"])
            diag["mean_non_convergence"].append(td["mean_non_convergence"])
            diag["workspace_saturation"].append(td["workspace_saturation"])
            diag["hub_churn"].append(td["hub_churn"])
            diag["veto_rate"].append(td["veto_rate"])
            diag["gate_mean"].append(td["gate_mean"])
            diag["gate_std"].append(td["gate_std"])
            diag["halt_probs"].append(td["halt_probs"])
            diag["epistemic_state"].append(td["epistemic_state"])
        if token_id.item() == tokenizer.eos_token_id: break

    diag["total_time"] = time_mod.time() - start_time
    diag["tps"] = len(diag["tokens"]) / max(diag["total_time"], 1e-9)
    diag["cache"] = cache
    return tokenizer.decode(full_seq[0], skip_special_tokens=True), diag


def print_generation_dashboard(diag, n_layers):
    nt = len(diag["tokens"])
    if nt == 0: return

    print(f"\n--- Early Exit Trace ---")
    early_exits = 0
    for token, layers in zip(diag["tokens"], diag["active_layers"]):
        if layers < n_layers:
            early_exits += 1
            print(f"  [Exited at Layer {layers}] -> {token!r}")
    if early_exits == 0: print("  No tokens exited early.")
    else: print(f"  Total early exits: {early_exits}/{nt} ({early_exits/nt*100:.1f}% compute saved)")

    max_layers = max(diag["active_layers"]) if diag["active_layers"] else n_layers
    print(f"\n{'='*60}\n LATENT COGNITIVE DASHBOARD\n{'='*60}")
    print(f"  Semantic Workload (Depth) : [{make_sparkline(diag['active_layers'], 1, max_layers + 2)}]")
    if diag["ponder_steps"] and sum(diag["ponder_steps"]) > 0:
        max_p = max(diag["ponder_steps"])
        print(f"  Ponder Loops Executed     : [{make_sparkline(diag['ponder_steps'], 0, max_p + 1)}]")
    if diag["write_magnitudes"]:
        print(f"  Memory Edits (Magnitude)  : [{make_sparkline(diag['write_magnitudes'])}]")
    if diag["hub_shared_write_mags"]:
        print(f"  Hub Activity (Shared)     : [{make_sparkline(diag['hub_shared_write_mags'])}]")

    # ── Epistemic Channels ──
    print(f"\n  {'─'*28} Epistemic Channels {'─'*28}")
    if diag["mean_uncertainties"]:
        print(f"  Entropy (Doubt)           : [{make_sparkline(diag['mean_uncertainties'])}]")
    if diag["mean_non_convergence"]:
        print(f"  Non-Convergence           : [{make_sparkline(diag['mean_non_convergence'])}]")
    if diag["workspace_saturation"]:
        print(f"  Workspace Saturation      : [{make_sparkline(diag['workspace_saturation'], 0.0, 1.0)}]")
    if diag["hub_churn"]:
        print(f"  Hub Churn                 : [{make_sparkline(diag['hub_churn'])}]")
    if diag["veto_rate"]:
        valid_veto = [v for v in diag["veto_rate"] if v is not None]
        if valid_veto:
            print(f"  S1/S2 Veto Rate           : [{make_sparkline(valid_veto, 0.0, 1.0)}]")
        else:
            print(f"  S1/S2 Veto Rate           : N/A (conv head mode)")
    if diag["gate_mean"]:
        print(f"  Epistemic Gate (mean)     : [{make_sparkline(diag['gate_mean'], 0.0, 1.0)}]")
    if diag["gate_std"]:
        print(f"  Epistemic Gate (std)      : [{make_sparkline(diag['gate_std'])}]")

    # Per-token epistemic state annotation
    if diag.get("epistemic_state"):
        state_chars = {"convergent": "●", "divergent_resolvable": "◐", "divergent_unresolvable": "○"}
        state_line = "".join(state_chars.get(s, "?") for s in diag["epistemic_state"])
        print(f"  Epistemic State           : [{state_line}]")
        print(f"    ● convergent  ◐ resolvable  ○ unresolvable")

    print(f"\n{'─'*60}")
    print(f"  Avg Active Layers    : {sum(diag['active_layers'])/nt:.1f} / {n_layers}")
    if diag["ponder_steps"]:
        print(f"  Avg Ponder Loops     : {sum(diag['ponder_steps'])/nt:.2f}")
    print(f"  Avg Memory Write Mag : {sum(diag['write_magnitudes'])/nt:.2f}")
    if diag["mean_uncertainties"]:
        print(f"  Avg Entropy          : {sum(diag['mean_uncertainties'])/nt:.4f}")
    if diag["mean_non_convergence"]:
        print(f"  Avg Non-Convergence  : {sum(diag['mean_non_convergence'])/nt:.4f}")
    if diag["workspace_saturation"]:
        print(f"  Avg Workspace Sat.   : {sum(diag['workspace_saturation'])/nt:.4f}")
    if diag["hub_churn"]:
        print(f"  Avg Hub Churn        : {sum(diag['hub_churn'])/nt:.4f}")
    if diag["veto_rate"]:
        valid_veto = [v for v in diag["veto_rate"] if v is not None]
        if valid_veto:
            print(f"  Avg Veto Rate        : {sum(valid_veto)/len(valid_veto):.4f}")
        else:
            print(f"  Avg Veto Rate        : N/A (conv head mode)")
    if diag["gate_mean"]:
        print(f"  Avg Gate Value       : {sum(diag['gate_mean'])/nt:.4f}")
    print(f"  Generation Speed     : {diag['tps']:.2f} tokens/sec")
    
    if "cache" in diag and diag["cache"].track_dynamics:
        cache = diag["cache"]
        print("\n  ──────────── Cache Dynamics ────────────")
        
        for label, bar in cache.format_dynamics_display():
            print(f"  {label:<25s}: [{bar}]")
        
        summary = cache.get_dynamics_summary()
        
        if 'avg_ponder_delta_per_step' in summary:
            deltas = summary['avg_ponder_delta_per_step']
            delta_str = " → ".join(f"{d:.4f}" for d in deltas)
            print(f"  Avg ponder delta/step  : {delta_str}")
            
        if 'ponder_converging_frac' in summary:
            print(f"  Ponder converging      : {summary['ponder_converging_frac']:.1%}")
            print(f"  Ponder diverging       : {summary['ponder_diverging_frac']:.1%}")
            print(f"  Ponder flat            : {summary['ponder_flat_frac']:.1%}")
        
        if 'mean_cross_position_similarity' in summary:
            print(f"  Cross-position sim     : {summary['mean_cross_position_similarity']:.4f}")
            trend = summary.get('cross_position_similarity_trend', 0)
            trend_str = "rising ⚠" if trend > 0.05 else "falling ✓" if trend < -0.05 else "stable"
            print(f"  Similarity trend       : {trend_str} ({trend:+.4f})")
        
        if 'mean_effective_dimensionality' in summary:
            edim = summary['mean_effective_dimensionality']
            ratio = edim / cache.d_latent
            print(f"  Effective dimensionality: {edim:.1f} / {cache.d_latent} ({ratio:.1%})")
            
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train CWT v5.4")
    parser.add_argument("--config", default="test", choices=["test", "prod"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--generate", default=None)
    parser.add_argument("--pondering", type=str, default=None,
                        choices=["true", "false"],
                        help="Override pondering at inference")
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        choices=["low", "medium", "high"],
                        help="Max pondering depth")
    args = parser.parse_args()
    config = get_test_config() if args.config == "test" else get_prod_config()

    if args.generate:
        if not args.resume: print("Error: --generate requires --resume"); return
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        config.model.vocab_size = len(tok)
        model = CognitiveWorkspaceTransformer(config.model).to(dev)
        model.load_state_dict(torch.load(args.resume, map_location=dev, weights_only=False)["model"], strict=False)
        ponder_flag = None
        if args.pondering == "true": ponder_flag = True
        elif args.pondering == "false": ponder_flag = False
        text, diag = generate(model, tok, args.generate, device=dev, pondering=ponder_flag, reasoning_effort=args.reasoning_effort)

        print(f"\nPrompt: {args.generate}\nGenerated: {text}")
        print_generation_dashboard(diag, config.model.n_layers)
        return

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
