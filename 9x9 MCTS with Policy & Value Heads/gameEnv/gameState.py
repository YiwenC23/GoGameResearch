from __future__ import annotations

import numpy as np

from dataclasses import dataclass, replace
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from gameEnv.rules import Rules

from gameEnv.board import BLACK


@dataclass(frozen=True)
class GameState:
    """
    Immutable game snapshot.
    - board: np.ndarray (N x N) with {BLACK=+1, WHITE=-1, EMPTY=0}
    - to_play: whose turn (+1 / -1)
    - prev_board: previous position (for simple-ko check)
    - passes_count: number of *consecutive* passes so far
    - move_number: ply count
    """
    board: np.ndarray
    to_play: int
    prev_board: Optional[np.ndarray]
    passes_count: int
    move_number: int
    
    # Constructors
    @staticmethod
    def new(size: int = 9, to_play: int = BLACK) -> "GameState":
        board = np.zeros((size, size), dtype=np.int8)
        return GameState(board=board, to_play=to_play, prev_board=None, passes_count=0, move_number=0)
    
    # Replace helper (since frozen=True)
    def _replace(self, **kwargs) -> "GameState":
        return replace(self, **kwargs)
    
    # Thin facade delegating to Rules
    def legal_moves(self, rules: "Rules") -> List[int]:
        return rules.legal_moves(self)
    
    def apply(self, rules: "Rules", move: int) -> "GameState":
        return rules.apply_move(self, move)
    
    def is_terminal(self, rules: "Rules") -> bool:
        return rules.is_terminal(self)
    
    def final_score(self, rules: "Rules") -> float:
        return rules.final_score(self)
    
    # NN encoder (example: 3 planes: player stones, opponent stones, to-play)
    def encode_planes(self) -> np.ndarray:
        """
        Return CxNxN float32 tensor suitable for a tiny ResNet input.
        Planes:
            0: current player's stones (1.0 where my stones, 0 else)
            1: opponent stones
            2: all-ones plane (or could be to-play scalar fed elsewhere)
        """
        player_plane = (self.board == self.to_play).astype(np.float32)
        opponent_plane = (self.board == -self.to_play).astype(np.float32)
        ones_plane = np.ones_like(player_plane, dtype=np.float32)
        return np.stack([player_plane, opponent_plane, ones_plane], axis=0)
