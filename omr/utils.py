
# omr/utils.py
import json
from pathlib import Path

# canonical warp size (width x height) - chosen arbitrarily; coordinates are mapped to this
CANON_WIDTH = 1400
CANON_HEIGHT = 2000

# layout for bubbles: 5 sections of 20 questions => 100 questions. We'll define a grid
# For prototype, we assume 5 columns (subjects) and 20 rows; each bubble cell will be computed.
# These constants determine where bubble centers lie on the warped image.
GRID_LEFT = 150
GRID_TOP = 300
GRID_RIGHT = CANON_WIDTH - 150
GRID_BOTTOM = CANON_HEIGHT - 200

Q_PER_SUBJECT = 20
NUM_SUBJECTS = 5
TOTAL_Q = Q_PER_SUBJECT * NUM_SUBJECTS

def load_answer_keys(path="answer_keys.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found.")
    with open(p, "r") as f:
        return json.load(f)

# returns list of (x,y,w,h) cells for every question in order 0..99
def generate_grid_cells():
    # We layout subject columns horizontally, each column has 20 rows stacked
    width = GRID_RIGHT - GRID_LEFT
    height = GRID_BOTTOM - GRID_TOP
    col_width = width / NUM_SUBJECTS
    row_height = height / Q_PER_SUBJECT
    cells = []
    for subj in range(NUM_SUBJECTS):
        for row in range(Q_PER_SUBJECT):
            x = int(GRID_LEFT + subj * col_width)
            y = int(GRID_TOP + row * row_height)
            w = int(col_width * 0.9)   # slightly inside margin
            h = int(row_height * 0.9)
            # center the cell around (x,y) by adjusting top-left
            cell_x = x + int((col_width - w)/2)
            cell_y = y + int((row_height - h)/2)
            cells.append((cell_x, cell_y, w, h))
    return cells
