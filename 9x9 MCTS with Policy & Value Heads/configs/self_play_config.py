from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BASEDIR = Path(__file__).parent.parent.resolve()


@dataclass
class SelfPlayConfig:
    seed: int | None = None
    board_size: int = 9
    komi: float = 5.5
    ko: str = "positional"
    allow_suicide: bool = False
    end_on_two_passes: bool = True
    eye_protect: bool = True
    pass_limit: int | None = None
    
    # MCTS
    sims: int = 300
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.15
    root_dirichlet_eps: float = 0.25
    temperature_moves: int = 0
    
    # Parallelism & IO
    num_workers: int = 12
    num_games: int = 3
    parallel: bool = True
    out_dir: Path = BASEDIR / "data" / f"self_play_data_{datetime.now():%Y%m%d-%H%M%S}"
    ckpt_path: Path = BASEDIR / "train" / "checkpoints" / "resnet-best.pt"
    
    in_channels: int = 2

