from dataclasses import dataclass
from pathlib import Path

BASEDIR = Path(__file__).parent.parent.resolve()


@dataclass
class SelfPlayConfig:
    seed: int | None = None
    board_size: int = 9
    komi: float = 5.5
    ko: str = "positional"
    allow_suicide: bool = False
    early_end_pass_alive: bool = True
    two_pass_end: bool = True
    
    # MCTS
    sims: int = 300
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.15
    root_dirichlet_eps: float = 0.25
    temperature_moves: int = 20
    
    # Parallelism & IO
    num_workers: int = 12
    num_games: int = 1000
    parallel: bool = True
    out_dir: Path = BASEDIR / "data" / "self_play_data"
    ckpt_path: Path = BASEDIR / "train" / "checkpoints" / "resnet-best.pt"
    
    in_channels: int = 2

