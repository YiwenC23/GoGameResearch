# train.py
from __future__ import annotations

import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.resNet import ResNet
from configs.train_config import TrainConfig, LoaderConfig
from algorithms.resNet import ResNetConfig
from train.dataset import make_dataloaders


def policy_loss_from_visits(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    # logits: [B, A], pi: [B, A], sum(pi,dim=1)==1
    logp = F.log_softmax(logits, dim=1)
    return -(pi * logp).sum(dim=1).mean()

def value_loss_tanh(value_pred: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # value_pred: [B,1] in [-1,1] after tanh, z: [B] in [-1,1]
    return F.mse_loss(value_pred.squeeze(1), z)

def policy_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    ent = -(p * logp).sum(dim=-1)
    return ent.mean()

def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}

def update_ema(ema, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if not torch.is_floating_point(v):
                ema[k] = v.detach().clone()
                continue
            if k in ema:
                ema[k].mul_(decay).add_(v, alpha=1.0 - decay)
            else:
                ema[k] = v.detach().clone()

class UseParams:
    """Context manager to temporarily swap model params (e.g., load EMA for validation)."""
    def __init__(self, model: nn.Module, new_state: Dict[str, torch.Tensor]):
        self.model = model
        self.new_state = new_state
        self.orig: Optional[Dict[str, torch.Tensor]] = None
    def __enter__(self):
        self.orig = clone_params(self.model)
        self.model.load_state_dict(self.new_state, strict=False)
        return self.model
    def __exit__(self, exc_type, exc, tb):
        assert self.orig is not None
        self.model.load_state_dict(self.orig, strict=False)
        self.orig = None

def build_cosine_warmup_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, cfg: TrainConfig):
    if not cfg.use_scheduler:
        return None
    
    warmup = cfg.warmup_steps
    if warmup <= 0 and cfg.warmup_ratio > 0.0:
        warmup = max(1, int(total_steps * cfg.warmup_ratio))
    
    min_factor = cfg.lr_min_factor
    def lr_lambda(step: int):
        if step < warmup:
            return (step + 1) / float(max(1, warmup))
        # cosine from warmup -> total_steps
        progress = min(1.0, (step - warmup) / float(max(1, total_steps - warmup)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def evaluate(model: ResNet, dl: DataLoader, device: torch.device):
    model.eval()
    tot_p, tot_v, tot = 0.0, 0.0, 0
    for batch in dl:
        planes = batch["planes"].to(device, non_blocking=True)
        pi     = batch["pi"].to(device, non_blocking=True)
        z      = batch["z"].to(device, non_blocking=True)
        logits, value = model(planes)
        p_loss = policy_loss_from_visits(logits, pi)
        v_loss = value_loss_tanh(value, z)
        B = planes.size(0)
        tot_p += float(p_loss.item()) * B
        tot_v += float(v_loss.item()) * B
        tot   += B
    return {"val_policy_loss": tot_p / max(1, tot), "val_value_loss": tot_v / max(1, tot)}

def rotate(path: Path, pattern: str, keep_last: int) -> None:
    paths = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[keep_last:]:
        try: p.unlink()
        except FileNotFoundError: pass

def fit(
    model: ResNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> str:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = build_cosine_warmup_scheduler(opt, total_steps, cfg)
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    if cfg.ema_decay is not None:
        ema_state = clone_params(model)
    
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    history_path = ckpt_dir / "loss_history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,train_policy,train_value,val_policy,val_value,val_total,seconds\n")
    
    best_val = float("inf")
    epochs_no_improve = 0
    
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        sum_p, sum_v, batches = 0.0, 0.0, 0
        
        for batch in train_loader:
            planes = batch["planes"].to(device, non_blocking=True)
            pi     = batch["pi"].to(device, non_blocking=True)
            z      = batch["z"].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits, value = model(planes)
                p_loss = policy_loss_from_visits(logits, pi)
                v_loss = value_loss_tanh(value, z)
                entropy = policy_entropy(logits)
                loss = (
                    cfg.policy_loss_weight * p_loss
                    + cfg.value_loss_weight * v_loss
                    - cfg.entropy_coeff * entropy
                )
            
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            
            # EMA update
            if ema_state is not None and cfg.ema_decay is not None:
                update_ema(ema_state, model, decay=cfg.ema_decay)
            
            if scheduler is not None:
                scheduler.step()
            
            sum_p += float(p_loss.item())
            sum_v += float(v_loss.item())
            batches += 1
            step += 1
        
        train_stats = {"policy_loss": sum_p / max(1, batches), "value_loss": sum_v / max(1, batches)}
        
        # ----- Validation: evaluate EMA weights if present -----
        if ema_state is not None:
            with UseParams(model, ema_state):
                val_stats = evaluate(model, val_loader, device)
        else:
            val_stats = evaluate(model, val_loader, device)
        
        # Weighted best-model metric to match training objective
        val_total = cfg.policy_loss_weight * val_stats["val_policy_loss"] + cfg.value_loss_weight * val_stats["val_value_loss"]
        
        elapsed = time.time() - t0
        row = (
            f"{epoch},"
            f"{train_stats['policy_loss']:.6f},"
            f"{train_stats['value_loss']:.6f},"
            f"{val_stats['val_policy_loss']:.6f},"
            f"{val_stats['val_value_loss']:.6f},"
            f"{val_total:.6f},"
            f"{elapsed:.2f}\n"
        )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(row)
        
        # Save epoch checkpoint (raw weights + EMA)
        payload = {
            "model": model.state_dict(),
            "ema": ema_state,
            "opt": opt.state_dict(),
            "epoch": epoch,
            "cfg": vars(cfg),
        }
        torch.save(payload, ckpt_dir / f"resnet-epoch{epoch:04d}.pt")
        rotate(ckpt_dir, "resnet-epoch*.pt", cfg.keep_last)
        
        # Save best according to the weighted metric (using EMA eval)
        if val_total + cfg.early_stopping_min_delta < best_val:
            best_val = val_total
            epochs_no_improve = 0
            torch.save(payload, ckpt_dir / "resnet-best.pt")
        else:
            epochs_no_improve += 1
        
        if epoch % cfg.save_every == 0:
            print(f"[{epoch}/{cfg.epochs}] "
                    f"train(p={train_stats['policy_loss']:.3f}, v={train_stats['value_loss']:.3f}) "
                    f"val(p={val_stats['val_policy_loss']:.3f}, v={val_stats['val_value_loss']:.3f}) "
                    f"best={best_val:.3f} "
                    f"{elapsed:.1f}s")
        
        # Early stopping
        if cfg.early_stopping_patience is not None and epochs_no_improve >= cfg.early_stopping_patience:
            print(f"Early stopping: no improvement for {epochs_no_improve} epochs.")
            break
    
    return str(ckpt_dir / "resnet-best.pt")

def _set_seed(seed: int | None):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1) Load configs
    tcfg = TrainConfig()
    lcfg = LoaderConfig()

    # (optional) reproducibility for dataset split etc.
    _set_seed(lcfg.seed)

    # 2) Build dataloaders from your self-play NPZs
    train_loader, val_loader = make_dataloaders(lcfg)

    # 3) Infer board size & channels from a sample (safer than hardcoding 9)
    sample = next(iter(train_loader))
    # 4) Create model and fit
    model = ResNet(cfg=ResNetConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = fit(model, train_loader, val_loader, tcfg, device)
    print(f"Training finished. Best checkpoint: {best}")

if __name__ == "__main__":
    main()
