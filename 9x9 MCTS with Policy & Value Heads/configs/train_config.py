from pathlib import Path
from typing import Literal
from dataclasses import dataclass

BASEDIR = Path(__file__).parent.parent.resolve()

@dataclass
class TrainConfig:
    """
    epochs: the number of complete passes through the training set
        - more passes lead to better model fitting but also increase the risk of overfitting and training time.
    lr: learning rate of the AdamW optimizer, determines the step size for each update
        - a larger value leads to faster convergence but possible oscillation
        - a smaller value is more stable but results in slower training
    weight_decay: the L2 regularization coefficient of AdamW, is used to reduce weights and prevent overfitting
        - the larger the value, the stronger the penalty
    grad_clip: gradient norm clipping threshold
        - if set to None, it is disabled
        - otherwise, gradients are limited to this upper bound to prevent exploding gradients
    amp: determine whether to enable automatic mixed precision
        - when enabled, torch.cuda.amp and GradScaler are used to reduce memory usage and accelerate GPU training
        - automatically falls back when using CPU or without CUDA
    policy_loss_weight: the weight of the Policy (π) loss in the total loss
        - increasing it makes the model pay more attention to matching the MCTS visit distribution,
        - decreasing it shifts focus towards accurate value prediction
    value_loss_weight: the Value (v) loss weight
        - balances the estimation of board outcomes and policy matching in the dual-head value-policy network
        - increasing it makes the model focus more on predicting accurate outcomes
        - decreasing it emphasizes matching the MCTS visit distribution
    entropy_reg: the coefficient for policy entropy regularization
        - a positive value rewards smoother distributions (encourages exploration)
        - setting it to 0 removes the entropy term, which may lead to overconfident policies
    ckpt_dir: directory to save model checkpoints during training
    save_every: frequency (in epochs) to save model checkpoints
    keep_last: number of most recent checkpoints to keep, older ones are deleted
    ema_decay: decay rate for Exponential Moving Average (EMA) of model parameters
        - if set to None, EMA is disabled
        - a value close to 1.0 results in slow updates, providing a stable averaged model
        - a smaller value makes the EMA model adapt more quickly to recent changes
    """
    # Optimization
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    
    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    entropy_coeff: float = 0.0005                              #  encourage exploration; set 0.0 to disable
    
    # Checkpointing
    ckpt_dir: str = str(BASEDIR / "train" / "checkpoints")
    save_every: int = 1
    keep_last: int = 5                                         # rotate old ckpts
    
    # EMA
    ema_decay: float | None = 0.997                            # None to disable
    
    # LR scheduler (cosine with warmup)
    use_scheduler: bool = True
    lr_min_factor: float = 0.05                                # final LR = factor * base LR
    warmup_steps: int = 1000                                   # steps
    # alternatively, if warmup_steps <= 0 and warmup_ratio > 0, will compute from total steps
    warmup_ratio: float = 0.0
    
    # Early stoppings
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 1e-4
    
    
    

@dataclass
class LoaderConfig:
    data_dir: Path = BASEDIR / "data/self_play_data"
    pattern: str = "selfplay_*.npz"
    batch_size: int = 256
    val_frac: float = 0.10
    num_workers: int = 8
    seed: int = 25
    in_memory: bool = True                 # True = fastest, uses more RAM
    split_by: Literal["sample", "file"] = "sample"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2