from __future__ import annotations

import sys
import json
import time
import uuid
import torch
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.MCTS import MCTS
from algorithms.resNet import ResNet, ResNetConfig
from gameEnv.rules import Rules
from gameEnv.gameState import GameState

from configs.self_play_config import SelfPlayConfig

sp_cfg = SelfPlayConfig()
_DEVICE_STR: str | None = None
_EVALUATOR: PolicyValueEvaluator | None = None
_RULES: Rules | None = None

def _pool_initializer(device_str: str, model_state_dict: Dict[str, torch.Tensor]) -> None:
    global _DEVICE_STR, _EVALUATOR, _RULES
    device = torch.device(device_str)
    
    cfg = ResNetConfig(board_size=sp_cfg.board_size, in_channels=sp_cfg.in_channels)
    model = ResNet(cfg)
    model.load_state_dict(model_state_dict)
    _EVALUATOR = PolicyValueEvaluator(model, device)
    
    _RULES = Rules(
        komi=sp_cfg.komi,
        ko=sp_cfg.ko,
        allow_suicide=sp_cfg.allow_suicide,
        end_on_two_passes=sp_cfg.end_on_two_passes,
        pass_limit=sp_cfg.pass_limit,
        eye_protect=sp_cfg.eye_protect,
    )
    _DEVICE_STR = device_str

class PolicyValueEvaluator:
    """
    Wraps a ResNet policy-value network as the evaluator callable expected by MCTS:
        (GameState, Rules) -> (policy_dict, value_float)
    where:
        - policy_dict is {move_index: probability} over LEGAL moves only
        - value is in [-1, 1], from the perspective of the *current* player to move
    """
    def __init__(self, net: ResNet, device: torch.device):
        self.net = net.to(device)
        self.device = device
        self.net.eval()  # set eval mode
    
    @torch.no_grad()
    def __call__(self, state: GameState, rules: Rules):
        # encode state to planes [C, N, N] -> tensor [1, C, N, N]
        planes = state.encode_planes().astype(np.float32)                     # np.ndarray [C, N, N]
        input_tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # [1, C, N, N]
        
        policy, value = self.net(input_tensor)  # [1, N*N + 1], [1, 1]
        policy = policy[0]                         # [N*N + 1]
        value = float(value[0, 0].item())          # scalar
        
        # mask logits to only legal moves, then softmax
        legal_moves = state.legal_moves(rules)  # includes PASS per Rules
        if not legal_moves:
            raise RuntimeError("No legal moves available in PolicyValueEvaluator.")
        
        logits = torch.full((policy.shape[0],), float("-inf"), device=self.device)  # [N*N + 1]
        logits[legal_moves] = policy[legal_moves]
        
        max_logit = torch.max(logits)
        probs = torch.exp(logits - max_logit)  # for numerical stability
        Z = float(probs.sum().item())
        if Z <= 1e-12:
            # all-legal masked logits were -inf -> uniform over legal moves
            policy_dict = {move: 1.0 / len(legal_moves) for move in legal_moves}
        else:
            probs /= Z
            policy_dict = {move: float(probs[move].item()) for move in legal_moves}
        
        return policy_dict, value

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
    while not rules.is_terminal(state):
        # Temperature schedule
        if move_idx < temperature_moves:
            temp = temp_init
        else:
            temp = temp_after
        
        root = mcts.search(state, add_root_noise=(temp > 0.0))
        
        # Policy target from visit counts
        pi = MCTS.prob_from_root_visits(root, board_size, temp)
        pi_list.append(pi)
        
        # Record planes and to_play before applying the move
        planes_list.append(state.encode_planes().astype(np.float32))
        to_play_list.append(state.to_play)
        
        # Choose a move using visit counts with temperature
        move = MCTS.best_move(root, temp)
        moves_played.append(move)
        
        # Apply move to game state
        state = state.apply(rules, move)
        move_idx += 1
    
    # Game over: compute file result
    black_score, white_score, final_score = rules.final_scores(state)
    if final_score > 0:
        winner = 1  # Black
    elif final_score < 0:
        winner = -1  # White
    else:
        winner = 0  # Draw
    
    if state.passes_count >= 2:
        terminal_reason = "two_passes"
    
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
            "final_score": float(final_score),                                          # Black-positive score, including komi
            "winner": int(winner),                                                      # 1=Black, -1=White, 0=Draw
            "black_score": float(black_score),
            "white_score": float(white_score),
            "komi": float(rules.komi),
            "terminal_reason": terminal_reason,
            "temperature_moves": int(temperature_moves),
            "moves": moves_played,
        }
    }
    
    return data

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

def load_model_state(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("ema") or payload["model"]
    return {k: v.detach().clone() for k, v in state.items()}

def self_play_worker(game_idx: int) -> tuple[int, Path, Dict[str, Any], int]:
    if _DEVICE_STR is None or _EVALUATOR is None or _RULES is None:
        raise RuntimeError("Pool initializer not run before self_play_worker.")
    if sp_cfg.seed is not None:
        np.random.seed(sp_cfg.seed + game_idx)
        torch.manual_seed(sp_cfg.seed + game_idx)
    
    rules = _RULES
    evaluator = _EVALUATOR
    mcts = MCTS(
        rules,
        evaluator,
        sims=sp_cfg.sims,
        c_puct=sp_cfg.c_puct,
        dirichlet_alpha=sp_cfg.root_dirichlet_alpha,
        dirichlet_epsilon=sp_cfg.root_dirichlet_eps,
    )
    
    init_state = GameState.new(size=sp_cfg.board_size)
    sample = one_self_play_game(
        mcts, state=init_state, rules=rules, board_size=sp_cfg.board_size,
        temperature_moves=sp_cfg.temperature_moves, temp_init=1.0, temp_after=0.0
    )
    output_path = save_game_npz(sample, sp_cfg.out_dir)
    length = int(sample["z"].shape[0])
    return game_idx, output_path, sample, length

def main():
    if sp_cfg.seed is not None:
        np.random.seed(sp_cfg.seed)
        torch.manual_seed(sp_cfg.seed)
    
    parallel = sp_cfg.parallel
    
    device = pick_device()
    print(f"Using device: {device}")
    
    ckpt_path = Path(sp_cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt_path.exists():
        # Generate a dummy checkpoint if none exists
        cfg = ResNetConfig(board_size=sp_cfg.board_size, in_channels=sp_cfg.in_channels)
        dummy_model = ResNet(cfg)
        torch.save({"model": dummy_model.state_dict()}, ckpt_path)
        print(f"No checkpoint found, created new random checkpoint at {ckpt_path} \n")
    model_state_dict = load_model_state(ckpt_path)
    
    if parallel:
        # Run self-play games in parallel using multiprocess
        print("Running self-play games in parallel...")
        ctx = mp.get_context("spawn")
        work = range(sp_cfg.num_games)
        with ctx.Pool(
            processes=sp_cfg.num_workers,
            initializer=_pool_initializer,
            initargs=(str(device), model_state_dict),
        ) as pool:
            for game_idx, output_path, sample, length in pool.imap_unordered(self_play_worker, work):
                winner_str = "BLACK" if sample["meta"]["winner"] == 1 else "WHITE" if sample["meta"]["winner"] == -1 else "DRAW"
                if sample["meta"]["winner"] == 1:
                    winner_str = "BLACK"
                    score = sample["meta"]["final_score"]
                elif sample["meta"]["winner"] == -1:
                    winner_str = "WHITE"
                    score = -sample["meta"]["final_score"]
                else:
                    winner_str = "DRAW"
                    score = sample["meta"]["final_score"]
                black_score = sample["meta"]["black_score"]
                white_score = sample["meta"]["white_score"]
                terminal_reason = sample["meta"]["terminal_reason"]
                print(
                    f"[Game {game_idx+1}/{sp_cfg.num_games}] "
                    f"length={length} moves | score={score:.1f} | "
                    f"winner={winner_str} | black_score={black_score:.1f} | "
                    f"white_score={white_score:.1f} (komi={sp_cfg.komi}) | "
                    f"terminal_reason={terminal_reason} | saved -> {output_path}"
                )
    else:
        # Run self-play games sequentially
        print("Running self-play games sequentially...")
        _pool_initializer(str(device), model_state_dict)
        for game_idx in range(sp_cfg.num_games):
            _, output_path, sample, length = self_play_worker(game_idx)
            
            if sample["meta"]["winner"] == 1:
                winner_str = "BLACK"
                score = sample["meta"]["final_score"]
            elif sample["meta"]["winner"] == -1:
                winner_str = "WHITE"
                score = -sample["meta"]["final_score"]
            else:
                winner_str = "DRAW"
                score = sample["meta"]["final_score"]
            black_score = sample["meta"]["black_score"]
            white_score = sample["meta"]["white_score"]
            terminal_reason = sample["meta"]["terminal_reason"]
            print(
                f"[Game {game_idx+1}/{sp_cfg.num_games}] "
                f"length={length} moves | score={score:.1f} | "
                f"winner={winner_str} | black_score={black_score:.1f} | "
                f"white_score={white_score:.1f} (komi={sp_cfg.komi}) | "
                f"terminal_reason={terminal_reason} | saved -> {output_path}"
            )
    
    print("Self-play completed.")

if __name__ == "__main__":
    main()