from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional

from gameEnv.rules import Rules
from gameEnv.board import BLACK, WHITE, pass_move
from gameEnv.gameState import GameState


PolicyValueEvaluator = Callable[[GameState, Rules], Tuple[Dict[int, float], float]]

@dataclass(slots=True)
class MCTSNode:
    state: GameState
    parent: Optional[MCTSNode] = None
    move: Optional[int] = None
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    
    @property
    def Q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_terminal(self, rules: Rules) -> bool:
        return self.state.is_terminal(rules)
    
    def expand(self, rules: Rules, priors: Dict[int, float]) -> None:
        """
        create/update all children nodes from a prior probability distribution over legal moves
        """
        for move, prob in priors.items():
            if move not in self.children:
                next_state = self.state.apply(rules, move)
                self.children[move] = MCTSNode(state=next_state, parent=self, move=move, prior=float(prob))
            else:
                self.children[move].prior = float(prob)
    
    def best_child(self, c_puct: float) -> MCTSNode:
        assert self.children, "No children to select from."
        parent_visits = max(1, self.visit_count)  # O(1) to avoid div by zero
        best_score = -float("inf")
        best_child = None
        
        for child in self.children.values():
            U = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = child.Q + U + 1e-9 * random.random()  # tiny noise to break ties randomly
            if score > best_score:
                best_score = score
                best_child = child
        
        assert best_child is not None
        return best_child


class MCTS:
    def __init__(
        self, rules: Rules, evaluator: PolicyValueEvaluator, sims: int=800, c_puct: float=1.4,
        dirichlet_alpha: float=0.15, dirichlet_epsilon: float=0.25, force_pass_min_visits: int=8,
    ):
        self.rules = rules
        self.eval = evaluator
        self.sims = sims
        self.c_puct = c_puct
        self.dir_alpha = dirichlet_alpha
        self.dir_epsilon = dirichlet_epsilon
        self.force_pass_min_visits = force_pass_min_visits
    
    def _terminal_value(self, state: GameState) -> float:
        _, _, score = self.rules.final_scores(state)  # Black: positive; White: negative
        if score > 0:
            winner = BLACK
        elif score < 0:
            winner = WHITE
        else:
            return 0.0  # draw
        
        if winner == state.to_play:
            return 1.0
        else:
            return -1.0
    
    def _add_dirichlet_noise(self, root: MCTSNode, renormalize: bool=True) -> None:
        if not root.children:
            return
        
        pass_idx = pass_move(root.state.board)
        legal_moves = list(root.children.keys())
        nonpass_moves = [move for move in legal_moves if move != pass_idx]
        
        if not nonpass_moves:
            # nothing to nosise (either only PASS child, or no children)
            return

        noise = np.random.dirichlet([self.dir_alpha] * len(nonpass_moves))
        for i, move in enumerate(nonpass_moves):
            child = root.children[move]
            child.prior = (1 - self.dir_epsilon) * child.prior + self.dir_epsilon * float(noise[i])
        
        # renormalize priors
        if renormalize:
            total_prior = sum(child.prior for child in root.children.values())
            if total_prior > 0:
                inv_total_prior = 1.0 / total_prior
                for child in root.children.values():
                    child.prior *= inv_total_prior
    
    def _backpropagate(self, path: List[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
    
    def _simulate_once(self, root: MCTSNode, forced_first_move: Optional[int] = None) -> None:
        """
        Run one simulation from `root`.
        If `forced_first_move` is set and exists as a child of the root, we descend into it first.
        """
        node = root
        path = [node]
        
        # Optionally force the very first child (used by the pass-vs-play hardening)
        if forced_first_move is not None and forced_first_move in node.children:
            node = node.children[forced_first_move]
            path.append(node)
        
        # Selection
        while node.children and not node.is_terminal(self.rules):
            node = node.best_child(self.c_puct)
            path.append(node)
        
        # Evaluation / Expansion
        if node.is_terminal(self.rules):
            leaf_value = self._terminal_value(node.state)
        else:
            priors, value = self.eval(node.state, self.rules)
            node.expand(self.rules, priors)
            leaf_value = value
        
        # Backup
        self._backpropagate(path, leaf_value)
    
    def _force_explore_pass_vs_play(self, root: MCTSNode) -> int:
        """
        If a PASS by the current root player would immediately end the game (two passes),
        ensure both PASS and a strong non-PASS obtain at least `force_pass_min_visits` visits
        before move selection. Returns the number of simulations consumed.
        """
        if not root.children or not self.rules.end_on_two_passes:
            return 0
        
        pass_idx = pass_move(root.state.board)
        pass_child = root.children.get(pass_idx)
        if pass_child is None:
            return 0  # no PASS available at this node (rules may gate it)
        
        # Two-pass would end the game if there was exactly one previous pass.
        would_end = (root.state.passes_count == 1)
        if not would_end:
            return 0
        
        # Pick a strong non-PASS to compare against (by prior is fine at root).
        nonpass_moves = [m for m in root.children.keys() if m != pass_idx]
        if not nonpass_moves:
            return 0
        
        best_nonpass = max(nonpass_moves, key=lambda m: root.children[m].prior)
        spent = 0
        
        # Run targeted simulations so that both pass and non-pass have some statistics.
        # We alternate to avoid long streaks exclusively to one child.
        targets = [pass_idx, best_nonpass]
        while any(root.children[t].visit_count < self.force_pass_min_visits for t in targets):
            for t in targets:
                if root.children[t].visit_count < self.force_pass_min_visits:
                    self._simulate_once(root, forced_first_move=t)
                    spent += 1
                    if spent >= self.sims:  # don't exceed budget defensively
                        return spent
        return spent
    
    def search(self, root_state: GameState, add_root_noise: bool=True) -> MCTSNode:
        if root_state.is_terminal(self.rules):
            return MCTSNode(state=root_state)
        
        # Root expension
        root = MCTSNode(state=root_state)
        priors, value = self.eval(root_state, self.rules)
        root.expand(self.rules, priors)
        
        # Root noise (exclude PASS)
        if add_root_noise:
            self._add_dirichlet_noise(root, renormalize=True)
        
        # If Pass would end the game, force both PASS and a good non-PASS to get visits
        sims_spent = self._force_explore_pass_vs_play(root)
        
        # Main simulations
        sims_remian = max(0, self.sims - sims_spent)
        for _ in range(sims_remian):
            self._simulate_once(root)
        
        return root
    
    @staticmethod
    def prob_from_root_visits(root: MCTSNode, board_size: int, temp: float=1.0) -> np.ndarray:
        N = board_size * board_size + 1  # including pass
        visits = np.zeros(N, dtype=np.float32)
        for move, child in root.children.items():
            visits[move] = child.visit_count
        
        if temp <= 1e-6:
            one_hot = np.zeros_like(visits)
            best_move = np.argmax(visits)
            one_hot[best_move] = 1.0
            return one_hot
        
        # soften distribution
        with np.errstate(divide="ignore"):
            temp_inverse = 1.0 / temp
            scaled_visits = np.zeros_like(visits, dtype=np.float64)
            np.power(visits, temp_inverse, out=scaled_visits, where=(visits > 0))
        if scaled_visits.sum() <= 0:
            # Degenerate: fall back to uniform over explored children
            mask = (visits > 0).astype(np.float32)
            num_explored = mask.sum()
            if num_explored > 0:
                return mask / num_explored
            else:
                return np.ones_like(visits) / N
        
        return scaled_visits / scaled_visits.sum()
    
    @staticmethod
    def best_move(root: MCTSNode, temp: float=1.0) -> int:
        if not root.children:
            raise ValueError ("best move called on a node with no children (likely terminal).")
        
        moves = np.fromiter(root.children.keys(), dtype=np.int64)
        counts = np.fromiter((child.visit_count for child in root.children.values()), dtype=np.int64)
        
        if temp <= 1e-6:
            return int(moves[np.argmax(counts)])
        
        # sample proportional to visit counts^1/temp
        with np.errstate(divide="ignore"):
            scaled_counts = np.zeros_like(counts, dtype=np.float64)
            np.power(counts, 1.0 / temp, out=scaled_counts, where=(counts > 0))
        if scaled_counts.sum() <= 0:
            # Degenerate: uniform over children
            return int(random.choice(moves))
        
        probs = scaled_counts / scaled_counts.sum()
        return int(np.random.choice(moves, p=probs))
    
    def re_root(
        self,
        old_root: MCTSNode,
        move: int,
        *,
        add_root_noise: bool = False,
        renormalize_priors: bool = True,
    ) -> MCTSNode:
        """
        Promote the played move's child to be the new root (AlphaZero-style tree reuse).
        Keeps the child's subtree statistics (N/W/Q) intact to save work next turn.
        
        If the move wasn't explored, creates a fresh root by applying the move.
        Optionally reapply root Dirichlet noise for self-play training games.
        """
        # Case 1: we already searched this move — reuse subtree & stats
        child = old_root.children.get(move)
        if child is not None:
            child.parent = None
            old_root.children.clear()
            
            if add_root_noise:
                # Use our PASS-excluding noise
                self._add_dirichlet_noise(child, renormalize=renormalize_priors)
            return child
        
        # Case 2: unexplored move — build a new root straightforwardly
        next_state = old_root.state.apply(self.rules, move)
        new_root = MCTSNode(state=next_state)
        
        if add_root_noise:
            priors, _ = self.eval(new_root.state, self.rules)
            new_root.expand(self.rules, priors)
            self._add_dirichlet_noise(new_root, renormalize=renormalize_priors)
        
        return new_root
