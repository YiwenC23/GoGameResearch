from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from gameEnv.rules import Rules
from gameEnv.gameState import GameState
from gameEnv.board import BLACK, WHITE, pass_move, place_stone_mechanical, own_chain_liberties_after


@dataclass
class MCTSNode:
    state: GameState
    parent: Optional["MCTSNode"] = None
    move: Optional[int] = None
    N_visits: int = 0
    value_sum: float = 0.0
    
    def __post_init__(self) -> None:
        """
        children: chdren nodes indexed by move
        _untried_moves: a list of legal moves that have not been tried yet or exploried nodes
        priors: prior probabilities for each move at this node
        """
        self.children: Dict[int, MCTSNode] = {}
        self._untried_moves: Optional[List[int]] = None
        self.priors: Dict[int, float] = {}
    
    def is_terminal(self, rules: Rules) -> bool:
        return self.state.is_terminal(rules)
    
    def untried_moves(self, rules: Rules) -> List[int]:
        """
        Return list of untried moves from this node. 
        """
        if self._untried_moves is None:
            self._untried_moves = self.state.legal_moves(rules)
        return self._untried_moves
    
    def has_untried_moves(self, rules: Rules) -> bool:
        return len(self.untried_moves(rules)) > 0
    
    def _ensure_priors(self, rules: Rules) -> None:
        """
        A lightweight heuristic policy for priors when neural network is not integrated:
            - If `place_stone_mechanical(board, move, to_play)` captures: big prior;
            - If move results more liberties: medium-high prior;
            - If self-atari (liberties become 0) and suicide illegal: near-zero prior;
            - Scale by resulting liberties (encourage healthy connections): e.g., 0.1 + 0.2 * (libs/4);
            - Pass: tiny prior (e.g., 1e-3), unless `passes_count == 1` (make it modest to allow ending).
        """
        if self.priors:
            return
        
        moves = self.untried_moves(rules)
        if not moves:
            return
        
        board = self.state.board
        player = self.state.to_play
        pass_idx = pass_move(board)
        
        scores = []
        for mv in moves:
            if mv == pass_idx:
                # prefer not to pass unless near end (second pass might be ok)
                base = 0.05 if self.state.passes_count == 1 else 1e-3
                scores.append((mv, base))
                continue
            
            nextb, captured_any = place_stone_mechanical(board, mv, player)
            if captured_any:
                scores.append((mv, 1.0))  # big prior for captures
                continue
            
            libs_after = own_chain_liberties_after(nextb, mv)
            if libs_after == 0:
                # self atari (and suicide is illegal in default rules)
                scores.append((mv, 1e-4))
                continue
            
            # crude atari-defense bonus: if libs==1 before, now >1
            # (cheap check: if playing here increases my chain liberty count a lot)
            bonus = 0.3 if libs_after >= 3 else 0.15
            prior = 0.10 + 0.20 * min(libs_after, 4) / 4.0 + bonus
            scores.append((mv, prior))
        
        # Normalize to probabilities
        total = sum(score for _, score in scores) or 1.0
        self.priors = {mv: score / total for mv, score in scores}
    
    def expand_best(self, rules: Rules, c_puct: float, fpu_reduction: float) -> "MCTSNode":
        """
        Use the prior policy to find out the best node to be expand instead of randomly expand.
        """
        self._ensure_priors(rules)
        moves: List[int] = self.untried_moves(rules)
        if not moves:
            raise RuntimeError("expand_best called with no untried moves")
        
        parent_q = 0.0 if self.N_visits == 0 else (self.value_sum / self.N_visits)
        fpu_q = max(-1.0, min(1.0, parent_q - fpu_reduction))
        sqrt_total = math.sqrt(max(1, self.N_visits))
        
        # score every unvisited move using FPU for Q and N_child=0 in the U term
        scored: List[Tuple[int, float]] = [
            (mv, fpu_q + c_puct * self.priors.get(mv, 0.0) * (sqrt_total / 1.0))
            for mv in moves
        ]
        
        best_mv = max(scored, key=lambda t: t[1])[0]
        moves.remove(best_mv)  # updates self._untried_moves
        new_state = self.state.apply(rules, best_mv)
        child = MCTSNode(state=new_state, parent=self, move=best_mv)
        self.children[best_mv] = child
        
        return child
    
    def best_child(self, rules: Rules, c_puct: float, fpu_reduction: float) -> "MCTSNode":
        assert self.children, "Cannot select child from an unexpanded leaf."
        # rules may also be passed in by caller
        self._ensure_priors(rules)
        
        parent_q = 0.0 if self.N_visits == 0 else (self.value_sum / self.N_visits)
        sqrt_total = math.sqrt(max(1, self.N_visits))
        
        best_score = -float("inf")
        best_child = None
        for mv, child in self.children.items():
            p = self.priors.get(mv, 0.0)
            
            if child.N_visits > 0:
                # flip to parent perspective
                q = -child.value_sum / child.N_visits
            else:
                # FPU: assume unvisited child is a bit worse than what we're seeing on average
                q = parent_q - fpu_reduction
                # clamp to [-1,1]
                q = max(-1.0, min(1.0, q))
            
            u = c_puct * p * (sqrt_total / (1.0 + child.N_visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        
        assert best_child is not None
        return best_child

class MCTS:
    def __init__(
        self, sims: int = 800, c_puct: float = 1.5, rollout: Optional[int] = 324,
        fpu_reduction: float = 0.3, root_dirichlet_alpha: float = 0.15, root_noise_frac: float = 0.25
    ) -> None:
        self.sims = sims
        self.c_puct = c_puct
        self.rollout = rollout
        self.fpu_reduction = fpu_reduction
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_noise_frac = root_noise_frac
    
    def best_move(self, state: GameState, rules: Rules) -> int:
        if state.is_terminal(rules):
            return pass_move(state.board)
        
        root = MCTSNode(state=state, parent=None, move=None)
        # init and noise priors at root
        root._ensure_priors(rules)
        if root.priors:
            # mix Dirichlet noise for exploration
            keys = list(root.priors.keys())
            alpha = self.root_dirichlet_alpha
            noise = np.random.dirichlet([alpha] * len(keys))
            eps = self.root_noise_frac
            for k, n in zip(keys, noise):
                root.priors[k] = (1 - eps) * root.priors[k] + eps * float(n)
        
        for _ in range(self.sims):
            node, path = self._tree_policy(root, rules)
            result = self._rollout(node.state, rules)
            self._backpropagate(path, result)
        
        if not root.children:
            return pass_move(state.board)
        
        best = max(root.children.values(), key=lambda child: child.N_visits)
        assert best.move is not None
        return best.move
    
    def _tree_policy(self, root: MCTSNode, rules: Rules) -> tuple[MCTSNode, List[MCTSNode]]:
        node = root
        path = [node]
        while True:
            if node.is_terminal(rules):
                return node, path
            # make sure priors exist before selection from a fully expanded node
            node._ensure_priors(rules)
            if node.has_untried_moves(rules):
                new_node = node.expand_best(rules, self.c_puct, self.fpu_reduction)
                path.append(new_node)
                return new_node, path
            if not node.children:
                return node, path
            
            node = node.best_child(rules, self.c_puct, self.fpu_reduction)
            path.append(node)
    
    def _sample_rollout_move(self, state: GameState, rules: Rules) -> int:
        """
        Pass weighting: add a simple late-game/quiet-board boost so the pass prior is not only tied to `passes_count`, making playouts actually end.
        """
        legal_moves = state.legal_moves(rules)
        pass_idx = pass_move(state.board)
        if not legal_moves:
            return pass_idx
        
        board = state.board
        to_play = state.to_play
        scored = []
        quiet_board_hint = True # flip to False if we see any "tactical" non-pass move
        
        for mv in legal_moves:
            if mv == pass_idx:
                continue
            
            nextb, captured = place_stone_mechanical(board, mv, to_play)
            if captured:
                scored.append((mv, 1.0))
                quiet_board_hint = False
                continue
            
            libs_after = own_chain_liberties_after(nextb, mv)
            if libs_after == 0:
                scored.append((mv, 1e-4))
                continue
            
            if libs_after >= 3:
                quiet_board_hint = False
            
            # add small bonus for healthy connections as in _ensure_priors to keep consistency
            bonus = 0.3 if libs_after >= 3 else 0.15
            prior = 0.10 + 0.20 * min(libs_after, 4) / 4.0 + bonus
            scored.append((mv, prior))
        
        # ___ Pass weighting ___
        # Base pass weight:
        #    - slightly higher if one pass already happened,
        #    - raise it further if the board looks "quiet" (no tactics found)
        pass_w = (0.05 if state.passes_count == 1 else 1e-3)
        if quiet_board_hint:
            pass_w *= 3.0  # gently push playouts to end when nothing's happening
        scored.append((pass_idx, pass_w))
        
        # Normalize & roulette sample (hardened)
        total = sum(score for _, score in scored)
        if total <= 0.0:
            return pass_idx
        
        r = random.random() * total
        accum = 0.0
        for mv, score in scored:
            accum += score
            if accum >= r:
                return mv
        return scored[-1][0]  # fallback
    
    def _rollout(self, state: GameState, rules: Rules) -> float:
        current_state = state
        current = current_state
        steps = 0
        origin_player = state.to_play
        
        # None => "full"; otherwise cap at self.rollout steps
        while not current_state.is_terminal(rules) and (self.rollout is None or steps < self.rollout):
            move = self._sample_rollout_move(current_state, rules)
            current = current_state.apply(rules, move)
            current_state = current
            steps += 1
        
        # force game end if capped early
        if not current.is_terminal(rules):
            p = pass_move(current.board)
            current = current.apply(rules, p)
            current = current.apply(rules, p)
        
        score = rules.final_score(current)
        if abs(score) < 1e-6:
            return 0.0
        
        winner = BLACK if score > 0 else WHITE
        return 1.0 if winner == origin_player else -1.0
    
    @staticmethod
    def _backpropagate(path: List[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.N_visits += 1
            node.value_sum += value
            value = -value
