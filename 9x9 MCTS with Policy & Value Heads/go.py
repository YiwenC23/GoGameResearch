import time
import queue
import threading
import tkinter as tk

from tkinter import ttk

from gameEnv.rules import Rules
from gameEnv.gameState import GameState
from gameEnv.board import BLACK, WHITE, EMPTY, pass_move

from algorithms.MCTS import MCTS


#* ---------------- Board Settings ---------------- *#
N = 9                                                # board size
PASS_STR = "PASS"                                    # string to display for pass move
COLS = "ABCDEFGHJ"                                   # board coordinates: Go coords skip I
COL2IDX = {col:i for i,col in enumerate(COLS)}       # board coordinates to index mapping


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

class GoGUI(tk.Tk):
    def __init__(self, rules: Rules, gtp_color=WHITE):
        super().__init__()
        self.title("9x9 Go")
        self.geometry("900x650")   # Initial window size
        
        # model
        self.rules = rules
        self.state = GameState.new(size=N)
        self.gtp_color = gtp_color
        
        # threading mailbox (worker -> GUI)
        self.result_q = queue.Queue()
        self.worker = None
        self.gtp_busy = False
        
        # Dynamic sizing variables (initialized with defaults)
        self.canvas_width = 500
        self.canvas_height = 500
        self.cell_size = 50
        self.pad = 40
        self.offset_x = 0
        self.offset_y = 0
        
        # Create main layout
        self.setup_layout()
        
        # MCTS engine
        self.mcts = MCTS(sims=128)
        
        # Initial draw and start GTP polling
        self.draw_all()
        self.after(50, self.poll_gtp_queue)
        self.add_message("Game started. Black to play")
        self.maybe_gtp_turn()
    
    @property
    def board_size(self) -> int:
        return self.state.board.shape[0]
    
    def setup_layout(self):
        # Main container
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for board
        self.board_frame = ttk.Frame(self.main_frame)
        self.board_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for the board
        self.canvas = tk.Canvas(self.board_frame, bg="#f4e6b0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Right frame for controls (fixed width)
        self.control_frame = ttk.Frame(self.main_frame, width=280)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.control_frame.pack_propagate(False)   # Maintain fixed width
        
        # Button section
        button_frame = ttk.LabelFrame(self.control_frame, text="Game Controls", padding=10)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_newGame = ttk.Button(button_frame, text="New Game", command=self.new_game)
        self.btn_newGame.pack(fill=tk.X, pady=5)
        
        self.btn_pass = ttk.Button(button_frame, text="Pass", command=self.on_pass)
        self.btn_pass.pack(fill=tk.X, pady=5)
        
        # Game info section
        info_frame = ttk.LabelFrame(self.control_frame, text="Game Info", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Board Size:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 5), pady=2)
        ttk.Label(info_frame, text=f"{N}x{N}").grid(row=0, column=1, sticky="w", padx=(0, 15), pady=2)
        
        ttk.Label(info_frame, text="Komi:", font=("Helvetica", 10, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 5), pady=2)
        ttk.Label(info_frame, text=f"{self.rules.komi}").grid(row=0, column=3, sticky="w", pady=2)
        
        ttk.Label(info_frame, text="Allow Suicide:", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=2)
        ttk.Label(info_frame, text="Yes" if self.rules.allow_suicide else "No").grid(row=1, column=1, sticky="w", padx=(0, 15), pady=2)
        
        ttk.Label(info_frame, text="End with 2 Passes:", font=("Helvetica", 10, "bold")).grid(row=1, column=2, sticky="w", padx=(0, 5), pady=2)
        ttk.Label(info_frame, text="Yes" if self.rules.end_on_two_passes else "No").grid(row=1, column=3, sticky="w", pady=2)
        
        # Message/Status window section
        message_frame = ttk.LabelFrame(self.control_frame, text="Game Status", padding=5)
        message_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrolled text widget
        text_frame = ttk.Frame(message_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.message_text = tk.Text(text_frame, wrap=tk.WORD, height=10, width=30, font=("Helvetica", 10), state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(text_frame, command=self.message_text.yview)
        self.message_text.config(yscrollcommand=scrollbar.set)
        
        self.message_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        
        # Calculate cell size based on available space
        # Use the smaller dimension to keep board square
        available_size = min(self.canvas_width, self.canvas_height)
        
        # Reserve space for padding (coordinates)
        self.pad = max(40, available_size * 0.08)
        board_size_px = max(1, available_size - 2 * self.pad)
        
        # Calculate cell size
        self.cell_size = max(1, board_size_px / (N - 1))
        
        # Calculate offset to center the board
        board_total_width = self.pad * 2 + self.cell_size * (N - 1)
        board_total_height = self.pad * 2 + self.cell_size * (N - 1)
        self.offset_x = (self.canvas_width - board_total_width) / 2
        self.offset_y = (self.canvas_height - board_total_height) / 2
        
        # Redraw everything with new dimensions
        self.draw_all()
    
    def add_message(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.message_text.config(state=tk.NORMAL)
        self.message_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.message_text.see(tk.END)   # Auto-scroll to bottom
        self.message_text.config(state=tk.DISABLED)
    
    #* Canvas Drawing
    def draw_all(self):
        """Redraw the entire board"""
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_coords()
        self.draw_star_points()
        self.draw_stones()
    
    def draw_grid(self):
        """Draw grid lines with dynamic sizing"""
        for i in range(N):
            x = self.pad + self.offset_x + i * self.cell_size
            y = self.pad + self.offset_y + i * self.cell_size
            
            # Horizontal lines
            self.canvas.create_line(
                self.pad + self.offset_x,
                y,
                self.pad + self.offset_x + self.cell_size * (N - 1),
                y,
                fill="black",
                width=1
            )
            
            # Vertical lines
            self.canvas.create_line(
                x,
                self.pad + self.offset_y,
                x,
                self.pad + self.offset_y + self.cell_size * (N - 1),
                fill="black",
                width=1
            )
    
    def draw_coords(self):
        """Draw coordinates with proper spacing to prevent overlap"""
        # Adjust font size based on cell size
        font_size = max(10, min(14, int(self.cell_size * 0.28)))
        coord_font = ("Helvetica", font_size)
        
        # Calculate safe spacing for coordinates
        coord_spacing = self.pad * 0.4
        
        for i in range(N):
            letter = COLS[i]
            x = self.pad + self.offset_x + i * self.cell_size
            
            # Top letters
            self.canvas.create_text(
                x, 
                self.offset_y + coord_spacing,
                text=letter,
                font=coord_font,
                fill="black"
            )
            # Bottom letters
            self.canvas.create_text(
                x, 
                self.offset_y + self.pad * 2 + self.cell_size * (N - 1) - coord_spacing,
                text=letter,
                font=coord_font,
                fill="black"
            )
        
        for j in range(N):
            num = N - j   # Correct numbering (9 at top, 1 at bottom)
            y = self.pad + self.offset_y + j * self.cell_size
            
            # Left numbers
            self.canvas.create_text(
                self.offset_x + coord_spacing, 
                y, 
                text=str(num), 
                font=coord_font,
                fill="black"
            )
            # Right numbers
            self.canvas.create_text(
                self.offset_x + self.pad * 2 + self.cell_size * (N - 1) - coord_spacing,
                y,
                text=str(num),
                font=coord_font,
                fill="black"
            )
    
    def draw_star_points(self):
        """Draw star points (hoshi) on the board"""
        stars = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        radius = max(2, min(4, self.cell_size * 0.08))
        
        for (x_index, y_index) in stars:
            x = self.pad + self.offset_x + y_index * self.cell_size
            y = self.pad + self.offset_y + x_index * self.cell_size
            self.canvas.create_oval(
                x - radius, y - radius,
                x + radius, y + radius,
                fill="black", outline="black"
            )
    
    def draw_stones(self):
        board = self.state.board
        stone_pad_ratio = 0.25   # Stone takes 50% of cell size
        stone_radius = self.cell_size * (0.5 - stone_pad_ratio)
        
        for x in range(N):
            for y in range(N):
                pos_value = board[x, y]
                if pos_value == EMPTY:
                    continue
                
                center_x = self.pad + self.offset_x + y * self.cell_size
                center_y = self.pad + self.offset_y + x * self.cell_size
                
                stone_color = "black" if pos_value == BLACK else "white"
                self.canvas.create_oval(
                    center_x - stone_radius, center_y - stone_radius,
                    center_x + stone_radius, center_y + stone_radius,
                    fill=stone_color,
                    outline=stone_color,
                    width=1
                )
    
    #* Interaction Handlers
    def on_click(self, event):
        if self.gtp_busy or self.state.is_terminal(self.rules):
            return
            
        loc = self.xy_to_loc(event.x, event.y)
        if loc is None:
            return
            
        x, y = loc
        index = loc_to_index(x, y, n=self.board_size)
        
        if index not in self.state.legal_moves(self.rules):
            self.add_message(f"Illegal move at {loc_to_gtp(x, y, n=self.board_size)}")
            return
            
        # Record the move
        player = "Black" if self.state.to_play == BLACK else "White"
        self.add_message(f"{player} played {loc_to_gtp(x, y, n=self.board_size)}")
        self.play_move(index)
    
    def on_pass(self):
        if self.gtp_busy or self.state.is_terminal(self.rules):
            return
            
        player = "Black" if self.state.to_play == BLACK else "White"
        self.add_message(f"{player} passed")
        self.play_move(pass_move(self.state.board))
    
    def new_game(self):
        self.state = GameState.new(size=N)
        self.draw_all()
        
        self.message_text.config(state=tk.NORMAL)
        self.message_text.delete(1.0, tk.END)
        self.message_text.config(state=tk.DISABLED)
        
        self.add_message("New game started. Black to play")
        self.maybe_gtp_turn()
    
    def xy_to_loc(self, x, y):
        """Convert canvas pixel coordinates to board row/col"""
        # Adjust for offset
        adjusted_x = x - self.offset_x
        adjusted_y = y - self.offset_y
        
        # Check if click is within the general board area first
        if adjusted_x < self.pad - self.cell_size/2 or adjusted_x > self.pad + self.cell_size * (N - 1) + self.cell_size/2:
            return None
        if adjusted_y < self.pad - self.cell_size/2 or adjusted_y > self.pad + self.cell_size * (N - 1) + self.cell_size/2:
            return None
        
        # Find nearest intersection
        y = round((adjusted_x - self.pad) / self.cell_size)
        x = round((adjusted_y - self.pad) / self.cell_size)
        
        # Check if within board bounds
        if not (0 <= x < N and 0 <= y < N):
            return None
        
        # Calculate the exact position of this intersection
        intersection_x = self.pad + y * self.cell_size
        intersection_y = self.pad + x * self.cell_size
        
        # Calculate distance from click to intersection
        distance = ((adjusted_x - intersection_x) ** 2 + (adjusted_y - intersection_y) ** 2) ** 0.5
        
        # Only accept if click is close enough to the intersection
        # Use a radius of about 20% of cell size (adjust 0.20 for different sensitivity)
        click_radius = self.cell_size * 0.20
        
        if distance <= click_radius:
            return (x, y)
        
        return None
    
    #* Game Flow
    def play_move(self, move):
        """Execute a move and update the game state"""
        self.state = self.state.apply(self.rules, move)
        self.draw_all()
        
        # Debug: show pass counter
        # self.add_message(f"(Consecutive passes: {self.state.passes_count})")
        
        if self.state.is_terminal(self.rules):
            self.show_final()
        else:
            self.maybe_gtp_turn()
    
    def maybe_gtp_turn(self):
        """Check if it's GTP's turn and launch GTP thinking if needed"""
        if self.state.to_play == self.gtp_color and not self.state.is_terminal(self.rules):
            self.launch_gtp()
    
    def launch_gtp(self):
        """Launch GTP computation in a background thread"""
        if self.gtp_busy: 
            return
            
        self.gtp_busy = True
        self.add_message("GTP is thinking...")
        self.btn_pass.state(["disabled"])
        self.btn_newGame.state(["disabled"])
        
        def worker():
            try:
                mv = self.mcts.best_move(self.state, self.rules)
                self.result_q.put(("gtp_move", mv))
            except Exception as e:
                self.result_q.put(("error", e))
        
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()
    
    def poll_gtp_queue(self):
        """Poll the GTP result queue and update GUI when GTP move is ready"""
        try:
            while True:
                tag, payload = self.result_q.get_nowait()
                
                if tag == "gtp_move":
                    mv = payload
                    label = index_to_gtp(mv, n=self.board_size)
                    
                    player = "Black" if self.state.to_play == BLACK else "White"
                    self.add_message(f"{player} played {label}")
                    
                    self.gtp_busy = False
                    self.btn_pass.state(["!disabled"])
                    self.btn_newGame.state(["!disabled"])
                    self.play_move(mv)
                    
                elif tag == "error":
                    self.gtp_busy = False
                    self.btn_pass.state(["!disabled"])
                    self.btn_newGame.state(["!disabled"])
                    self.add_message(f"GTP error: {payload}")
                    
        except queue.Empty:
            pass
        
        # Reschedule polling
        self.after(50, self.poll_gtp_queue)
    
    def show_final(self):
        """Display final game result"""
        score = self.rules.final_score(self.state)
        if score > 0:
            msg = f"Game over. Black wins by {score:.1f}"
        elif score < 0:
            msg = f"Game over. White wins by {(-score):.1f}"
        else:
            msg = "Game over. Draw."
        self.add_message(msg)
        self.add_message("Click 'New Game' to play again")


if __name__ == "__main__":
    rules = Rules()
    app = GoGUI(rules, gtp_color=WHITE)
    app.mainloop()
