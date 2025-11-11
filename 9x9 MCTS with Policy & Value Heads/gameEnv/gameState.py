from __future__ import annotations

import numpy as np

from dataclasses import dataclass, replace, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from gameEnv.rules import Rules

from gameEnv.board import BLACK


def _hash_board(board: np.ndarray) -> int:
    """
    Fast positional hash for positional-superko checks.
    """
    return hash(board.tobytes())

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
    pos_hashes: frozenset[int] = field(default_factory=frozenset) # positional hashes for superko checking
    
    # Constructors
    @staticmethod
    def new(size: int = 9, to_play: int = BLACK) -> "GameState":
        board = np.zeros((size, size), dtype=np.int8)
        return GameState(board=board, to_play=to_play, prev_board=None, passes_count=0, move_number=0, pos_hashes=frozenset({_hash_board(board)}))
    
    # Replace helper (since frozen=True)
    def _replace(self, **kwargs) -> "GameState":
        data = dict(self.__dict__); data.update(kwargs)
        return type(self)(**data)
    
    # Thin facade delegating to Rules
    def legal_moves(self, rules: "Rules") -> List[int]:
        return rules.legal_moves(self)
    
    def apply(self, rules: "Rules", move: int) -> "GameState":
        return rules.apply_move(self, move)
    
    def is_terminal(self, rules: "Rules") -> bool:
        return rules.is_terminal(self)
    
    def final_score(self, rules: "Rules") -> float:
        return rules.final_score(self)
    
    # NN encoder
    def encode_planes(self) -> np.ndarray:
        """
        Return tensor of shape [C, N, N] suitable for a tiny ResNet input.
        Planes:
            plane 0: stones of player-to-move
            plane 1: stones of opponent
        """
        player_plane = (self.board == self.to_play).astype(np.float32)
        opponent_plane = (self.board == -self.to_play).astype(np.float32)
        return np.stack([player_plane, opponent_plane], axis=0)
