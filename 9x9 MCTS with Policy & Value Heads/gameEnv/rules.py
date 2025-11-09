# rules.py
from __future__ import annotations

import numpy as np
from typing import List
from dataclasses import dataclass

from gameEnv.gameState import GameState
from gameEnv.board import (
    EMPTY,
    board_size, pass_move,
    place_stone_mechanical,
    own_chain_liberties_after,
    resolve_own_suicide,
    empty_regions_with_borders,
    all_points_are_pass_alive_or_territory
)


@dataclass
class Rules:
    """
    Minimal rules for 9x9 prototype.
    - ko: 'simple' - forbids recreating the immediately previous board
    - allow_suicide: False by default
    - scoring: 'territory' (Tromp-Taylor-style: stones + empty regions bordered by one color)
    - end_on_two_passes: True by default
    - komi: 6.0 by default (A number of points awarded to the White player to compensate for Black's advantage of playing first.)
    """
    komi: float = 6.0
    ko: str = "simple"
    allow_suicide: bool = False
    scoring: str = "territory"
    end_on_two_passes: bool = True
    early_end_pass_alive: bool = True
    
    def legal_moves(self, state: "GameState") -> List[int]:
        """Return all legal moves (flattened 0..N*N plus PASS)."""
        board = state.board
        n = board_size(board)
        pass_idx = pass_move(board)
        legal: List[int] = []
        
        # All points
        for index in range(n*n):
            x, y = divmod(index, n)
            if board[x, y] != EMPTY:
                continue
            
            # Try mechanical placement + enemy captures
            next_board, _ = place_stone_mechanical(board, index, state.to_play)
            
            # Suicide check (after capturing neighbors)
            liberty_cnt = own_chain_liberties_after(next_board, index)
            if not self.allow_suicide and liberty_cnt == 0:
                continue   # illegal due to suicide
            
            # Simple-ko check: disallow if equal to the immediately previous position
            if self.ko == "simple" and state.prev_board is not None:
                # If suicide is allowed and liberty_cnt==0, rules would remove own stones.
                # For simple-ko, that removal can matter; emulate it for the comparison.
                if self.allow_suicide and liberty_cnt == 0:
                    # Create a copy for resolution
                    next_board_ko = next_board.copy()
                    resolve_own_suicide(next_board_ko, index)
                    if np.array_equal(next_board_ko, state.prev_board):
                        continue
                else:
                    if np.array_equal(next_board, state.prev_board):
                        continue
            
            legal.append(index)
        legal.append(pass_idx)   # Pass is always legal
        
        return legal
    
    #* State Transition
    def apply_move(self, state: "GameState", move: int) -> "GameState":
        """Assumes `move` was legal per `legal_moves`."""
        board = state.board
        pass_idx = pass_move(board)
        prev_board = board.copy()
        
        if move == pass_idx:
            return state._replace(
                board=board.copy(),
                to_play=-state.to_play,
                prev_board=prev_board,
                passes_count=state.passes_count + 1,
                move_number=state.move_number + 1,
            )
        
        # Point move
        next_board, _ = place_stone_mechanical(board, move, state.to_play)
        liberty_cnt = own_chain_liberties_after(next_board, move)
        if self.allow_suicide and liberty_cnt == 0:
            resolve_own_suicide(next_board, move)
        else:
            # With suicide disallowed, liberty_cnt==0 should have been filtered out in legal_moves.
            assert liberty_cnt > 0, "Illegal suicide move slipped through legality check."
        
        return state._replace(
            board=next_board,
            to_play=-state.to_play,
            prev_board=prev_board,
            passes_count=0,   # reset on non-pass
            move_number=state.move_number + 1,
        )
    
    #* Termination & Scoring
    def is_terminal(self, state: "GameState") -> bool:
        if self.end_on_two_passes and state.passes_count >= 2:
            return True
        if self.early_end_pass_alive and all_points_are_pass_alive_or_territory(state.board):
            return True
        return False
    
    def territory_score(self, board: np.ndarray) -> int:
        """
        Territory score with no komi:
            +1 for each black stone, -1 for each white stone,
            +region_size for owner if an empty region borders ONLY that owner.
        """
        score = int(np.sum(board))   # BLACK=+1 per stone, WHITE=-1 per stone
        for region, borders in empty_regions_with_borders(board):
            if len(borders) == 1:
                owner = next(iter(borders))
                score += owner * len(region)
        return score
    
    def final_score(self, state: "GameState") -> float:
        """Black-positive score, with komi applied (White gets komi)."""
        assert self.is_terminal(state), "Called final_score on non-terminal state"
        return float(self.territory_score(state.board)) - self.komi   # subtract komi -> equivalent to giving White +komi
