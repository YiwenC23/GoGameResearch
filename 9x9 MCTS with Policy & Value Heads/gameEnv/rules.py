# rules.py
from __future__ import annotations

import numpy as np
from typing import Optional, List, Set
from dataclasses import dataclass

from gameEnv.gameState import GameState, _hash_board
from gameEnv.board import (
    BLACK, WHITE,EMPTY,
    board_size, pass_move,
    place_stone_mechanical,
    own_chain_liberties_after,
    resolve_own_suicide,
    empty_regions_with_borders,
    stone_groups, loc_to_index,
)


@dataclass
class Rules:
    """
    Minimal rules for 9x9 prototype.
    - ko: 'positional' - forbids recreating the immediately previous board
    - allow_suicide: False by default
    - scoring: 'territory' (Tromp-Taylor-style: stones + empty regions bordered by one color)
    - end_on_two_passes: True by default
    - komi: 7.0 by default (A number of points awarded to the White player to compensate for Black's advantage of playing first.)
    """
    komi: float = 7.0
    ko: str = "positional"
    allow_suicide: bool = False
    scoring: str = "area"
    end_on_two_passes: bool = True
    pass_limit: Optional[int] = None
    eye_protect: Optional[bool] = True
    
    def legal_moves(self, state: "GameState") -> List[int]:
        """Return all legal moves (flattened 0..N*N plus PASS)."""
        board = state.board
        n = board_size(board)
        pass_idx = pass_move(board)
        legal_mv: List[int] = []
        
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
                continue   # illegal due to suicide (self-capture)
            
            # For ko checks, we must compare against the ACTUAL post-move board.
            # If suicide is allowed and our own chain has 0 liberties, resolve it first.
            next_board_final = next_board
            if self.allow_suicide and liberty_cnt == 0:
                next_board_final = next_board.copy()
                resolve_own_suicide(next_board_final, index)
            
            # Ko rules
            if self.ko == "simple":
                # Disallow if equal to the immediately previous board
                if state.prev_board is not None and np.array_equal(next_board_final, state.prev_board):
                    continue
            elif self.ko == "positional":
                # Disallow if this position has occurred anywhere before
                # (requires GameState to carry a set of prior board hashes)
                if _hash_board(next_board_final) in state.pos_hashes:
                    continue
            else:
                raise ValueError(f"Unknown ko rule: {self.ko}")
            
            legal_mv.append(index)
        
        if self.eye_protect:
            legal_mv = self.legal_moves_eyes_filter(state, legal_mv)
        
        can_pass = self.pass_limit is None or state.move_number >= self.pass_limit
        if pass_idx not in legal_mv and (can_pass or not legal_mv):
            legal_mv.append(pass_idx)
        
        return legal_mv
    
    def legal_moves_eyes_filter(self, state: "GameState", legal_moves: List[int]) -> List[int]:
        """
        For each adjacent region of live groups, filter out moves when the region contains only one empty space (only one liberty)
        
        """
        board = state.board
        n = board_size(board)
        pass_idx = pass_move(board)
        live_groups_dict = stone_groups(board, state.to_play)
        
        eye_locs: Set[int] = set()
        for regions in live_groups_dict.values():
            for region in regions:
                # region is a list of indices, filter only when the region has only one liberty
                if len(region) == 1:
                    (x, y) = next(iter(region))
                    eye_locs.add(loc_to_index(x, y, n))
        
        filtered_legal_moves: List[int] = []
        for move in legal_moves:
            if move == pass_idx:
                filtered_legal_moves.append(move)
                continue
            if move not in eye_locs:
                filtered_legal_moves.append(move)
        
        return filtered_legal_moves
    
    #* State Transition
    def apply_move(self, state: "GameState", move: int) -> "GameState":
        """Assumes `move` was legal per `legal_moves`."""
        board = state.board
        n = board_size(board)
        pass_idx = pass_move(board)
        prev_board = board.copy()
        
        if move == pass_idx:
            return state._replace(
                board=board.copy(),
                to_play=-state.to_play,
                prev_board=prev_board,
                passes_count=state.passes_count + 1,
                move_number=state.move_number + 1,
                pos_hashes=state.pos_hashes,
            )
        
        # Point move
        next_board, _ = place_stone_mechanical(board, move, state.to_play)
        liberty_cnt = own_chain_liberties_after(next_board, move)
        if self.allow_suicide and liberty_cnt == 0:
            resolve_own_suicide(next_board, move)
        else:
            # With suicide disallowed, liberty_cnt==0 should have been filtered out in legal_moves.
            assert liberty_cnt > 0, "Illegal suicide move slipped through legality check."
        
        # Update positional history for positional superko
        new_hashes = state.pos_hashes
        if self.ko == "positional":
            new_hashes = frozenset(set(state.pos_hashes) | {_hash_board(next_board)})
        
        return state._replace(
            board=next_board,
            to_play=-state.to_play,
            prev_board=prev_board,
            passes_count=0,   # reset on non-pass
            move_number=state.move_number + 1,
            pos_hashes=new_hashes,
        )
    
    #* Termination & Scoring
    def is_terminal(self, state: "GameState") -> bool:
        if self.end_on_two_passes and state.passes_count >= 2:
            return True
        return False
    
    def area_score(self, board: np.ndarray) -> tuple[int, int]:
        """
        Area (Tromp-Taylor/Chinese) score with no komi:
            +1 per black stone, -1 per white stone,
            +region_size for owner if an empty region borders ONLY that owner.
        """
        b_score = int(np.count_nonzero(board == BLACK))
        w_score = int(np.count_nonzero(board == WHITE))
        for region, borders in empty_regions_with_borders(board):
            if borders == {BLACK}:
                b_score += len(region)
            elif borders == {WHITE}:
                w_score += len(region)
        return b_score, w_score
    
    def final_scores(self, state: "GameState") -> tuple[float, float, float]:
        assert self.is_terminal(state)
        b_score, w_score = self.area_score(state.board)
        w_score += self.komi
        final_score = b_score - w_score
        return float(b_score), float(w_score), float(final_score)
