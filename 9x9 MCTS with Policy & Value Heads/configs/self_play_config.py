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
    sims: int = 214
    c_puct: float = 1.5
    root_dirichlet_alpha: float = 0.15
    root_dirichlet_eps: float = 0.25
    temperature_moves: int = 30
    
    # PASS gating (self-play only)
    pass_gating: bool = True
    pass_gating_moves: int =  40 # block PASS until this move (if non-PASS legal moves exist)
    
    # Parallelism & IO
    num_workers: int = 14
    num_games: int = 1000
    parallel: bool = True
    out_dir: Path = BASEDIR / "data" / "self_play_data"
    
    in_channels: int = 2
