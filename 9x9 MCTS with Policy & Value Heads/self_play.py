from __future__ import annotations

import json
import uuid
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, TypedDict

from algorithms.MCTS import MCTS
from algorithms.resNet import ResNet, ResNetConfig
from gameEnv.rules import Rules
from gameEnv.gameState import GameState

from configs import self_play_config


class PolicyValueEvaluator:
    """
    Wraps a ResNet policy-value network as the evaluator callable expected by MCTS:
        (GameState, Rules) -> (policy_dict, value_float)
    where:
        - policy_dict is {move_index: probability} over LEGAL moves only
        - value_float is in [-1, 1], from the perspective of the *current* player to move
    """
    def __init__(self, ResNet: ResNet, device: torch.device):
        self.ResNet = ResNet.to(device)
        self.device = device
        self.ResNet.eval()  # set eval mode
    
    @torch.no_grad()
    def __call__(self, state: GameState, rules: Rules) -> Tuple[Dict[int, float], float]:
        # encode state to planes [C, N, N] -> tensor [1, C, N, N]
        planes = state.encode_planes().astype(np.float32)                     # np.ndarray [C, N, N]
        input_tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # [1, C, N, N]
        
        policy, value = self.ResNet(input_tensor)  # [1, N*N + 1], [1, 1]
        policy = policy[0]                         # [N*N + 1]
        value = float(value[0, 0].item())          # scalar
        
        # mask logits to only legal moves, then softmax
        legal_moves = state.legal_moves(rules)  # includes PASS (index N*N)
        if not legal_moves:
            raise ValueError("No legal moves available in the current state.")
        
        logits = torch.full((policy.shape[0],), float("-inf"), device=self.device)  # [N*N + 1]
        logits[legal_moves] = policy[legal_moves]
        
        max_logit = torch.max(logits)
        probs = torch.exp(logits - max_logit)  # for numerical stability
        Z = float(probs.sum().item())
        if Z <= 1e-12:
            # all-legal masked logits were -inf -> uniform over legal moves
            prob_dict = {move: 1.0 / len(legal_moves) for move in legal_moves}
        else:
            probs /= Z
            prob_dict = {move: float(probs[move].item()) for move in legal_moves}
        
        return prob_dict, value
    

def one_self_play_game(
    mcts: MCTS, state: GameState, rules: Rules, board_size: int=9,
    temperature_moves: int=20, temp_init: float=1.0, temp_after: float=0.0
) -> Dict[str, np.ndarray]:
    """
    Run one self-play game and return training examples.
    
    Returns a dict containing np arrays:
        planes: [T, C, N, N]
        pi:     [T, N*N+1]  (visit-based policy targets)
        z:      [T]         (+1/-1/0 result from the perspective of the player at each step)
        meta:   JSON-serializable metadata (moves, final_score, winner, komi, etc.)
    """
    state = state
    root = None
    planes_list: List[np.ndarray] = []
    pi_list: List[np.ndarray] = []
    to_play_list: List[int] = []
    moves_played: List[int] = []
    
    move_idx = 0
    last_move = None
    while not state.is_terminal(rules):
        # Temperature schedule
        if move_idx < temperature_moves:
            temp = temp_init
        else:
            temp = temp_after
        
        if root is None:
            root = mcts.search(state, add_root_noise=True)
        
        
        # Policy target from visit counts
        pi = mcts.prob_from_root_visits(root, board_size, temp)
        pi_list.append(pi)
        
        # Record planes and to_play before applying the move
        planes_list.append(state.encode_planes().astype(np.float32))
        to_play_list.append(state.to_play)
        
        # Choose a move using visit counts with temperature
        move = MCTS.best_move(root, temp)
        moves_played.append(move)
        
        # Apply move to game state
        state = state.apply(rules, move)
        last_move = move
        move_idx += 1
        
        # Prepare the next root (tree reuse) now for the next iteration
        if not state.is_terminal(rules):
            root = mcts.re_root(root, move, add_root_noise=False, renormalize_priors=True)  # noise added at loop top
            
            # Optional fallback if the MCTS might return None or an unexplored child
            if root is None or (not root.children):
                root = mcts.search(state, add_root_noise=True)
        
    # Game over: compute file result
    final_score = rules.final_score(state)  # positive: Black win; negative: White win
    if final_score > 0:
        winner = 1  # Black
    elif final_score < 0:
        winner = -1  # White
    else:
        winner = 0  # Draw
    
    # Build value targets from the perspective of each player at each move
    z_list: List[float] = []
    for player in to_play_list:
        if winner == 0:
            z = 0.0
        else:
            if player == winner:
                z = 1.0
            else:
                z = -1.0
        z_list.append(float(z))
    
    data = {
        "planes": np.array(planes_list, dtype=np.float32),        # [T, C, N, N]
        "pi": np.array(pi_list, dtype=np.float32),                # [T, N*N + 1]
        "z": np.array(z_list, dtype=np.float32),                  # [T]
        "meta": {
            "board_size": int(board_size),
            "final_score": float(final_score),
            "winner": int(winner),
            "komi": float(rules.komi),
            "temperature_moves": int(temperature_moves),
            "moves": moves_played,
        }
    }
    
    return data

class SelfPlaySample(TypedDict):
    planes: np.ndarray  # [T, C, N, N]
    pi: np.ndarray      # [T, N*N + 1]
    z: np.ndarray       # [T]
    meta: Dict[str, Any]          # JSON-serializable metadata

def save_game_npz(sample: Dict[str, np.ndarray], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"selfplay_{timestamp}_{uid}.npz"
    file_path = output_dir / file_name
    
    met_bytes = json.dumps(sample["meta"]).encode("utf-8")
    payload = {
        "planes": sample["planes"],
        "pi": sample["pi"],
        "z": sample["z"],
        "meta": np.frombuffer(met_bytes, dtype=np.uint8),
    }
    
    np.savez_compressed(file_path, **payload)
    
    return file_path

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon uses the MPS (Metal) backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    np.random.seed(self_play_config.SEED)
    torch.manual_seed(self_play_config.SEED)
    
    device = pick_device()
    
    print(f"Using device: {device}")
    
    # Build rules
    rules = Rules()
    rules.komi = self_play_config.KOMI
    rules.allow_suicide = self_play_config.ALLOW_SUICIDE
    rules.ko = self_play_config.KO
    
    # Build network with random weights (first start)
    cfg = ResNetConfig(board_size=self_play_config.BOARD_SIZE, in_channels=self_play_config.IN_CHANNELS)
    model = ResNet(cfg)
    evaluator = PolicyValueEvaluator(model, device)
    
    # MCTS
    mcts = MCTS(
        rules, evaluator, sims=self_play_config.MCTS_SIMS, c_puct=self_play_config.C_PUCT,
        dirichlet_alpha=self_play_config.DIRICHLET_ALPHA, dirichlet_epsilon=self_play_config.DIRICHLET_EPS
    )
    
    # Run self-play games
    output_dir = self_play_config.OUTPUT_DIR
    for game in range(self_play_config.NUM_SELF_PLAY_GAMES):
        init_state = GameState.new(size=self_play_config.BOARD_SIZE)
        sample = one_self_play_game(
            mcts, state=init_state, rules=rules, board_size=self_play_config.BOARD_SIZE,
            temperature_moves=self_play_config.TEMPERATURE_MOVES, temp_init=1.0, temp_after=0.0
        )
        output_path = save_game_npz(sample, output_dir)
        print(f"[Game {game+1}/{self_play_config.NUM_SELF_PLAY_GAMES}] length={len(sample['z'])} moves | score={sample['meta']['final_score']:.1f} | winner={sample['meta']['winner']} | saved -> {output_path}")
    
    print("Self-play completed.")

if __name__ == "__main__":
    main()
