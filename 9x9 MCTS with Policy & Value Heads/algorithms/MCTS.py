from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from gameEnv.board import BLACK, WHITE, pass_move
from gameEnv.gameState import GameState
from gameEnv.rules import Rules


@dataclass
class MCTSNode:
    state: GameState
    parent: Optional["MCTSNode"] = None
    move: Optional[int] = None
    visit_count: int = 0
    value_sum: float = 0.0

    def __post_init__(self) -> None:
        self.children: Dict[int, MCTSNode] = {}
        self._untried_moves: Optional[List[int]] = None

    def is_terminal(self, rules: Rules) -> bool:
        return self.state.is_terminal(rules)

    def untried_moves(self, rules: Rules) -> List[int]:
        if self._untried_moves is None:
            moves = self.state.legal_moves(rules)
            random.shuffle(moves)
            self._untried_moves = moves
        return self._untried_moves

    def has_untried_moves(self, rules: Rules) -> bool:
        return len(self.untried_moves(rules)) > 0

    def expand(self, rules: Rules) -> "MCTSNode":
        move = self.untried_moves(rules).pop()
        next_state = self.state.apply(rules, move)
        child = MCTSNode(state=next_state, parent=self, move=move)
        self.children[move] = child
        return child

    def best_child(self, c_param: float) -> "MCTSNode":
        assert self.children, "Cannot select child from an unexpanded leaf."
        log_total = math.log(self.visit_count + 1)
        best_score = -float("inf")
        best_child = None
        for child in self.children.values():
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploit = -child.value_sum / child.visit_count
                explore = math.sqrt(log_total / child.visit_count)
                score = exploit + c_param * explore
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child


class MCTS:
    def __init__(self, sims: int = 800, c_uct: float = 1.4, rollout_limit: int = 256) -> None:
        self.sims = sims
        self.c_uct = c_uct
        self.rollout_limit = rollout_limit

    def best_move(self, state: GameState, rules: Rules) -> int:
        """Return the move index with highest visit count after MCTS search."""
        if state.is_terminal(rules):
            return pass_move(state.board)

        root = MCTSNode(state=state)
        for _ in range(self.sims):
            node, path = self._tree_policy(root, rules)
            result = self._rollout(node.state, rules)
            self._backpropagate(path, result)

        if not root.children:
            return pass_move(state.board)

        best = max(root.children.values(), key=lambda child: child.visit_count)
        assert best.move is not None
        return best.move

    def _tree_policy(self, root: MCTSNode, rules: Rules) -> tuple[MCTSNode, List[MCTSNode]]:
        node = root
        path = [node]
        while True:
            if node.is_terminal(rules):
                return node, path
            if node.has_untried_moves(rules):
                new_node = node.expand(rules)
                path.append(new_node)
                return new_node, path
            if not node.children:
                return node, path
            node = node.best_child(self.c_uct)
            path.append(node)
    
    # Perform a random rollout from the given state to a terminal state
    def _rollout(self, state: GameState, rules: Rules) -> float:
        current = state
        steps = 0
        origin_player = state.to_play

        while not current.is_terminal(rules) and steps < self.rollout_limit:
            legal_moves = current.legal_moves(rules)
            if not legal_moves:
                move = pass_move(current.board)
            else:
                pass_idx = pass_move(current.board)
                non_pass_moves = [m for m in legal_moves if m != pass_idx]
                move = random.choice(non_pass_moves) if non_pass_moves else pass_idx
            current = current.apply(rules, move)
            steps += 1

        if not current.is_terminal(rules):
            pass_idx = pass_move(current.board)
            current = current.apply(rules, pass_idx)
            current = current.apply(rules, pass_idx)

        score = rules.final_score(current)
        if abs(score) < 1e-6:
            return 0.0
        winner = BLACK if score > 0 else WHITE
        return 1.0 if winner == origin_player else -1.0

    @staticmethod
    def _backpropagate(path: List[MCTSNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
