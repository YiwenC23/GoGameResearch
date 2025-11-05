
import pathlib

BASEDIR = pathlib.Path(__file__).parent.parent.resolve()

SEED = 25
BOARD_SIZE = 9
KOMI = 6.5
ALLOW_SUICIDE = False
KO = "simple"
MCTS_SIMS = 800
C_PUCT = 1.4
DIRICHLET_ALPHA = 0.15
DIRICHLET_EPS = 0.25
TEMPERATURE_MOVES = 20
NUM_SELF_PLAY_GAMES = 100
OUTPUT_DIR = BASEDIR/"data/self_play_data"
IN_CHANNELS = 3