from __future__ import annotations

import numpy as np
from typing import Iterable, Set, Tuple, List


BLACK, WHITE, EMPTY = 1, -1, 0   # Stone Colors
PASS_STR = "PASS"
COLS = "ABCDEFGHJ"
COL2IDX = {col:i for i,col in enumerate(COLS)}
ORTHO_DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))   # Directions (4-neighborhood: up, down, left, right)

def board_size(board: np.ndarray) -> int:
    """Return N for an N x N board."""
    return int(board.shape[0])

def loc_to_index(x: int, y: int, n: int = 9) -> int:
    """
    Convert a 2D board location to a 1D engine index in [0, n*n - 1].
    Parameters:
        x   (int): zero-based row (0 = top)
        y   (int): zero-based column (0 = left)
        n   (int): board size (e.g., 9 for 9x9)
    Returns:
        int: flat index for the intersection
    """
    board_size = n
    return x * board_size + y

def index_to_loc(m: int, n: int = 9) -> tuple[int, int]:
    """
    Convert a 1D engine index to (x, y).
    Parameters:
        m   (int): flat index of an intersection (0..n*n-1)
        n   (int): board size (e.g., 9 for 9x9)
    Returns:
        (x, y): zero-based location
    """
    index = m
    board_size = n
    return divmod(index, board_size)

def loc_to_gtp(x: int, y: int, n: int = 9) -> str:
    """
    Convert internal (x, y) with row=0 at TOP to GTP/human coordinate like 'D4'.
    Notes:
        - Letters run left -> right using COLS (A..J without I for 9x9).
        - Numbers count from BOTTOM (A1 lower-left), so y is flipped.
    """
    board_size = n
    letter = COLS[y]             # 0..8 -> A..J (no I)
    number = board_size - x      # flip because GUI uses top=0, GTP uses bottom=1
    return f"{letter}{number}"

def gtp_to_loc(s: str, n: int = 9) -> tuple[int, int] | None:
    """
    'D4' -> loc(x, y) using our COLS string; 'PASS'/'pass' -> None.
    Accepts case-insensitive letters.
    """
    board_size = n
    gtp_str = s.strip()
    if not gtp_str:
        raise ValueError("empty coord")
    if gtp_str.lower() == "pass":
        return None
    
    letter = gtp_str[0].upper()
    try:
        number = int(gtp_str[1:])
    except ValueError:
        raise ValueError(f"bad row number: {gtp_str[1:]}")
    
    if letter not in COL2IDX:
        raise ValueError(f"bad column: {letter}")
    if not (1 <= number <= board_size):
        raise ValueError(f"bad row: {number}")
    
    y = COL2IDX[letter]
    x = board_size - number
    return (x, y)

def gtp_to_index(s: str, n: int = 9) -> int:
    """
    'D4' -> engine index; 'PASS' -> n*n.
    """
    board_size = n
    loc = gtp_to_loc(s, board_size)
    return board_size * board_size if loc is None else loc_to_index(*loc, n=board_size)

def index_to_gtp(m: int, n: int = 9) -> str:
    """
    engine index or pass -> 'D4' / 'PASS'.
    """
    index = m
    board_size = n
    if index == board_size * board_size:
        return PASS_STR
    
    x, y = index_to_loc(index, board_size)
    return loc_to_gtp(x, y, board_size)

def pass_move(board: np.ndarray) -> int:
    """Encode PASS as the linear index N*N for the board size."""
    n = board_size(board)
    return n * n

def on_board(x: int, y: int, n: int) -> bool:
    size = n
    return 0 <= x < size and 0 <= y < size

def neighbors(x: int, y: int, n: int) -> Iterable[Tuple[int, int]]:
    for delta_x, delta_y in ORTHO_DIRS:
        neighbor_x = x + delta_x
        neighbor_y = y + delta_y
        if on_board(neighbor_x, neighbor_y, n):
            yield neighbor_x, neighbor_y

def flood_group(board: np.ndarray, x: int, y: int) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]]]:
    """
    Return (stones, liberties) of the chain at (x,y).
    NOTE: caller should guarantee board[x,y] != EMPTY.
    """
    n = board_size(board)
    color = int(board[x, y])
    stones: Set[Tuple[int,int]] = set()
    liberties: Set[Tuple[int,int]] = set()
    stack = [(x, y)]
    
    while stack:
        current_x, current_y = stack.pop()
        if (current_x, current_y) in stones:
            continue
        stones.add((current_x, current_y))
        
        for neighbor_x, neighbor_y in neighbors(current_x, current_y, n):
            pos_value = int(board[neighbor_x, neighbor_y])
            if pos_value == EMPTY:
                liberties.add((neighbor_x, neighbor_y))
            elif pos_value == color and (neighbor_x, neighbor_y) not in stones:
                stack.append((neighbor_x, neighbor_y))
    
    return stones, liberties

def remove_group(board: np.ndarray, stones: Iterable[Tuple[int,int]]) -> None:
    """Capture: in-place removal of a set of stones."""
    for x, y in stones:
        board[x, y] = EMPTY

def place_stone_mechanical(board: np.ndarray, move: int, color: int) -> Tuple[np.ndarray, bool]:
    """
    'Mechanical' placement that does NOT make any legality decisions.
    - Places a stone of `color` at `move` (linear 0..N*N-1).
    - Captures any adjacent opponent chains reduced to 0 liberties.
    - DOES NOT remove own chain even if at 0 liberties (so suicide legality stays outside).
    Returns (next_board, captured_any).
    """
    n = board_size(board)
    x, y = divmod(move, n)
    assert board[x, y] == EMPTY, "Square must be empty for mechanical placement"
    next_board = board.copy()
    next_board[x, y] = color
    
    captured = False
    # Capture adjacent opponent chains that lost last liberty
    for neighbor_x, neighbor_y in neighbors(x, y, n):
        if next_board[neighbor_x, neighbor_y] == -color:
            stones, libs = flood_group(next_board, neighbor_x, neighbor_y)
            if not libs:
                remove_group(next_board, stones)
                captured = True
    
    return next_board, captured

def own_chain_liberties_after(nextb: np.ndarray, move: int) -> int:
    """Liberty count for the newly placed chain at `move` on board `nextb`."""
    n = board_size(nextb)
    x, y = divmod(move, n)
    _, libs = flood_group(nextb, x, y)
    return len(libs)

def resolve_own_suicide(next_board: np.ndarray, move: int) -> None:
    """
    If the just-placed chain at `move` has 0 liberties, remove it (suicide resolution).
    Call this ONLY if the rules allow suicide.
    """
    n = board_size(next_board)
    x, y = divmod(move, n)
    stones, libs = flood_group(next_board, x, y)
    if not libs:
        remove_group(next_board, stones)

def empty_regions_with_borders(board: np.ndarray) -> List[Tuple[Set[Tuple[int,int]], Set[int]]]:
    """
    Find all empty regions. For each, return (empties_set, bordering_colors_set).
    Bordering colors are subset of {BLACK, WHITE}.
    """
    n = board_size(board)
    seen_pos: Set[Tuple[int,int]] = set()
    regions_with_borders: List[Tuple[Set[Tuple[int,int]], Set[int]]] = []
    
    for x in range(n):
        for y in range(n):
            if board[x, y] != EMPTY or (x, y) in seen_pos:
                continue
            
            # Flood-fill empties
            region: Set[Tuple[int,int]] = set()
            borders: Set[int] = set()
            
            stack = [(x, y)]
            while stack:
                current_x, current_y = stack.pop()
                if (current_x, current_y) in region:
                    continue
                region.add((current_x, current_y))
                seen_pos.add((current_x, current_y))
                
                for neighbor_x, neighbor_y in neighbors(current_x, current_y, n):
                    pos_value = int(board[neighbor_x, neighbor_y])
                    if pos_value == EMPTY and (neighbor_x, neighbor_y) not in region:
                        stack.append((neighbor_x, neighbor_y))
                    elif pos_value != EMPTY:
                        borders.add(pos_value)
            
            regions_with_borders.append((region, borders))
    
    return regions_with_borders
