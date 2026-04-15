"""
Challenger Agent — CS3600 Tournament Bot

Strategy:
  1. HMM rat tracker — only guess when P(best_cell) > 0.45 (EV > 0.5)
  2. Greedy line-building: prime in straight lines, then carpet for big points
  3. Expectiminimax for mid-game decisions with alpha-beta
  4. Time-managed iterative deepening
"""

from collections.abc import Callable
from typing import Tuple, List
import numpy as np

from game.enums import (
    Direction, MoveType, Cell, BOARD_SIZE,
    CARPET_POINTS_TABLE, loc_after_direction,
)
from game.board import Board
from game.move import Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAT_BONUS = 4
RAT_PENALTY = 2

NOISE_EMISSION = np.array([
    [0.7,  0.15, 0.15],   # SPACE
    [0.1,  0.8,  0.1 ],   # PRIMED
    [0.1,  0.1,  0.8 ],   # CARPET
    [0.5,  0.3,  0.2 ],   # BLOCKED
], dtype=np.float64)

DIST_OFFSETS = np.array([-1, 0, 1, 2])
DIST_PROBS = np.array([0.12, 0.70, 0.12, 0.06], dtype=np.float64)

ALL_X = np.array([i % BOARD_SIZE for i in range(64)])
ALL_Y = np.array([i // BOARD_SIZE for i in range(64)])

DIRECTIONS = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
DIR_DELTAS = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}

OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


# ---------------------------------------------------------------------------
# Rat Tracker (HMM)
# ---------------------------------------------------------------------------
class RatTracker:
    def __init__(self, T: np.ndarray):
        self.T = np.asarray(T, dtype=np.float64)
        belief = np.zeros(64, dtype=np.float64)
        belief[0] = 1.0
        for _ in range(1000):
            belief = belief @ self.T
        self.belief = belief

    def update(self, board: Board, noise: int, reported_dist: int, wx: int, wy: int):
        predicted = self.belief @ self.T

        floor_types = self._get_floor_types(board)
        noise_lik = NOISE_EMISSION[floor_types, noise]

        actual_dists = np.abs(ALL_X - wx) + np.abs(ALL_Y - wy)
        implied = actual_dists[:, None] + DIST_OFFSETS[None, :]

        if reported_dist == 0:
            match = (implied <= 0).astype(np.float64)
        else:
            match = (implied == reported_dist).astype(np.float64)
        dist_lik = match @ DIST_PROBS

        posterior = predicted * noise_lik * dist_lik
        total = posterior.sum()
        if total > 0:
            self.belief = posterior / total
        else:
            self.belief = predicted / predicted.sum()

    def reset_after_catch(self):
        belief = np.zeros(64, dtype=np.float64)
        belief[0] = 1.0
        for _ in range(1000):
            belief = belief @ self.T
        self.belief = belief

    def best_guess(self) -> Tuple[Tuple[int, int], float]:
        idx = int(np.argmax(self.belief))
        prob = float(self.belief[idx])
        return (idx % BOARD_SIZE, idx // BOARD_SIZE), prob

    def search_ev(self) -> float:
        _, prob = self.best_guess()
        return prob * RAT_BONUS - (1 - prob) * RAT_PENALTY

    @staticmethod
    def _get_floor_types(board: Board) -> np.ndarray:
        floor = np.zeros(64, dtype=np.int32)
        for i in range(64):
            bit = 1 << i
            if board._primed_mask & bit:
                floor[i] = 1
            elif board._carpet_mask & bit:
                floor[i] = 2
            elif board._blocked_mask & bit:
                floor[i] = 3
        return floor


# ---------------------------------------------------------------------------
# Board analysis helpers
# ---------------------------------------------------------------------------

def is_open(board: Board, x: int, y: int) -> bool:
    """Check if (x,y) is in-bounds, not blocked, not primed, not carpeted."""
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return not bool((board._blocked_mask | board._primed_mask | board._carpet_mask) & bit)


def is_space(board: Board, x: int, y: int) -> bool:
    """Check if (x,y) is a SPACE cell (not blocked/primed/carpeted)."""
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return bool(board._space_mask & bit)


def is_primed(board: Board, x: int, y: int) -> bool:
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return bool(board._primed_mask & bit)


def count_primed_run(board: Board, x: int, y: int, dx: int, dy: int) -> int:
    """Count consecutive primed squares starting from (x+dx, y+dy)."""
    run = 0
    cx, cy = x + dx, y + dy
    while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
        if is_primed(board, cx, cy):
            run += 1
            cx += dx
            cy += dy
        else:
            break
    return run


def can_move_to(board: Board, x: int, y: int, opp_x: int, opp_y: int) -> bool:
    """Check if a worker can move to (x,y) — space or carpet, not occupied."""
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return False
    if (x, y) == (opp_x, opp_y):
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    if board._blocked_mask & bit:
        return False
    if board._primed_mask & bit:
        return False
    return True


def best_carpet_from(board: Board, x: int, y: int) -> Tuple[Direction, int, int]:
    """Find the best carpet roll from position (x,y). Returns (dir, length, points)."""
    best_dir = None
    best_len = 0
    best_pts = -999
    opp = board.opponent_worker.get_location()

    for d in DIRECTIONS:
        dx, dy = DIR_DELTAS[d]
        run = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
            bit = 1 << (cy * BOARD_SIZE + cx)
            if (board._primed_mask & bit) and (cx, cy) != opp:
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            pts = CARPET_POINTS_TABLE.get(min(run, 7), 0)
            if pts > best_pts:
                best_pts = pts
                best_len = run
                best_dir = d

    return best_dir, best_len, best_pts


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def evaluate(board: Board, rat_ev: float = 0.0) -> float:
    my_pts = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    mx, my_ = board.player_worker.get_location()
    ox, oy = board.opponent_worker.get_location()

    score = (my_pts - opp_pts) * 1.0

    # Carpet potential from my position
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        run = count_primed_run(board, mx, my_, dx, dy)
        if run >= 2:
            score += CARPET_POINTS_TABLE.get(min(run, 7), 0) * 0.4

    # Carpet potential from opponent (threat)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        run = count_primed_run(board, ox, oy, dx, dy)
        if run >= 2:
            score -= CARPET_POINTS_TABLE.get(min(run, 7), 0) * 0.25

    # Value being on/near open spaces for future priming
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = mx + dx, my_ + dy
        if is_space(board, nx, ny):
            score += 0.2
        if is_primed(board, nx, ny):
            score += 0.15

    return score


# ---------------------------------------------------------------------------
# Minimax
# ---------------------------------------------------------------------------

def get_ordered_moves(board: Board) -> List[Move]:
    moves = board.get_valid_moves(exclude_search=True)

    def priority(m: Move) -> int:
        if m.move_type == MoveType.CARPET:
            return 1000 + CARPET_POINTS_TABLE.get(m.roll_length, 0) * 10
        elif m.move_type == MoveType.PRIME:
            mx, my_ = board.player_worker.get_location()
            dx, dy = DIR_DELTAS[m.direction]
            # Prefer priming that extends a line
            behind = count_primed_run(board, mx, my_, -dx, -dy)
            return 100 + behind * 10
        else:
            return 1

    moves.sort(key=priority, reverse=True)
    return moves


def minimax(board: Board, depth: int, alpha: float, beta: float,
            maximizing: bool, rat_ev: float, time_left: Callable) -> float:
    if depth == 0 or board.is_game_over():
        if maximizing:
            return evaluate(board, rat_ev)
        else:
            return -evaluate(board, rat_ev)

    if time_left() < 0.5:
        if maximizing:
            return evaluate(board, rat_ev)
        else:
            return -evaluate(board, rat_ev)

    moves = get_ordered_moves(board)
    if not moves:
        if maximizing:
            return evaluate(board, rat_ev)
        else:
            return -evaluate(board, rat_ev)

    if maximizing:
        value = -float('inf')
        for m in moves:
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            score = minimax(child, depth - 1, alpha, beta, False, rat_ev, time_left)
            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value if value != -float('inf') else evaluate(board, rat_ev)
    else:
        value = float('inf')
        for m in moves:
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            score = minimax(child, depth - 1, alpha, beta, True, rat_ev, time_left)
            if score < value:
                value = score
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value if value != float('inf') else -evaluate(board, rat_ev)


def find_best_move_minimax(board: Board, rat_ev: float,
                           time_left: Callable, max_depth: int) -> Move:
    moves = get_ordered_moves(board)
    if not moves:
        return Move.plain(Direction.RIGHT)

    best_move = moves[0]

    for depth in range(1, max_depth + 1):
        if time_left() < 2.0:
            break

        current_best = None
        current_best_score = -float('inf')

        for m in moves:
            if time_left() < 1.0:
                break
            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()
            score = minimax(child, depth - 1, -float('inf'), float('inf'),
                          False, rat_ev, time_left)
            if score > current_best_score:
                current_best_score = score
                current_best = m

        if current_best is not None:
            best_move = current_best

    return best_move


# ---------------------------------------------------------------------------
# Greedy strategy helpers
# ---------------------------------------------------------------------------

def greedy_move(board: Board, rat_ev: float) -> Move:
    """Fast greedy move selection when we don't have time for minimax."""
    moves = board.get_valid_moves(exclude_search=True)
    if not moves:
        return Move.plain(Direction.RIGHT)

    best_score = -float('inf')
    best_move = moves[0]

    for m in moves:
        score = score_move(board, m)
        if score > best_score:
            best_score = score
            best_move = m

    return best_move


def score_move(board: Board, m: Move) -> float:
    """Score a single move greedily."""
    mx, my_ = board.player_worker.get_location()
    ox, oy = board.opponent_worker.get_location()
    dx, dy = DIR_DELTAS.get(m.direction, (0, 0)) if m.direction else (0, 0)

    if m.move_type == MoveType.CARPET:
        pts = CARPET_POINTS_TABLE.get(m.roll_length, 0)
        # Landing position after carpet
        lx = mx + dx * m.roll_length
        ly = my_ + dy * m.roll_length
        # Bonus for carpet potential from landing spot
        future_bonus = 0
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            run = count_primed_run(board, lx, ly, ddx, ddy)
            if run >= 2:
                future_bonus += CARPET_POINTS_TABLE.get(min(run, 7), 0) * 0.2
        return pts + future_bonus + 50  # strong preference for carpeting

    elif m.move_type == MoveType.PRIME:
        nx, ny = mx + dx, my_ + dy
        # Value: 1 point for priming, plus line-building bonus
        line_bonus = 0
        # Check how this prime extends lines in the direction we're moving
        ahead = count_primed_run(board, nx, ny, dx, dy)
        behind = count_primed_run(board, mx, my_, -dx, -dy)
        # Total line through current pos after priming: behind + 1 (this cell)
        total_line = behind + 1  # The cell we're priming
        if total_line >= 2:
            line_bonus += CARPET_POINTS_TABLE.get(min(total_line, 7), 0) * 0.3

        # Also check perpendicular directions from the departure cell
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (ddx, ddy) == (dx, dy) or (ddx, ddy) == (-dx, -dy):
                continue
            run = count_primed_run(board, mx, my_, ddx, ddy)
            if run >= 1:
                line_bonus += run * 0.15

        # Check if destination has good future carpet potential
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            run = count_primed_run(board, nx, ny, ddx, ddy)
            if run >= 2:
                line_bonus += CARPET_POINTS_TABLE.get(min(run, 7), 0) * 0.25

        return 1 + line_bonus + 5  # priming is generally good

    else:  # PLAIN
        nx, ny = mx + dx, my_ + dy
        # Value plain moves that position us near primeable squares or carpet opportunities
        pos_value = 0
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            adj_x, adj_y = nx + ddx, ny + ddy
            if is_space(board, adj_x, adj_y):
                pos_value += 0.3
            run = count_primed_run(board, nx, ny, ddx, ddy)
            if run >= 2:
                pos_value += CARPET_POINTS_TABLE.get(min(run, 7), 0) * 0.3
        return pos_value


# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------
class PlayerAgent:
    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.rat_tracker = None
        if transition_matrix is not None:
            T = np.asarray(transition_matrix, dtype=np.float64)
            self.rat_tracker = RatTracker(T)
        self.observations_since_reset = 0
        self.search_cooldown = 0  # turns to wait before next search

    def commentate(self):
        return "Challenger GG"

    def play(self, board: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        noise, reported_dist = sensor_data
        wx, wy = board.player_worker.get_location()

        # Handle rat resets from searches
        rat_reset = False
        if board.opponent_search[0] is not None and board.opponent_search[1]:
            rat_reset = True
        if board.player_search[0] is not None and board.player_search[1]:
            rat_reset = True

        if rat_reset and self.rat_tracker:
            self.rat_tracker.reset_after_catch()
            self.observations_since_reset = 0
            self.search_cooldown = 3  # wait a few turns after a reset

        # Handle failed search — add cooldown
        if board.player_search[0] is not None and not board.player_search[1]:
            self.search_cooldown = 3  # wait after a miss

        if self.search_cooldown > 0:
            self.search_cooldown -= 1

        self.observations_since_reset += 1

        # Update rat belief
        rat_ev = -2.0
        if self.rat_tracker:
            self.rat_tracker.update(board, int(noise), reported_dist, wx, wy)
            rat_ev = self.rat_tracker.search_ev()

        # Search for rat only when:
        # 1. High EV (> 1.0)
        # 2. Enough observations (>= 4 since last reset)
        # 3. Not in cooldown
        # 4. Max belief probability > 0.5
        if (self.rat_tracker and self.search_cooldown == 0
                and self.observations_since_reset >= 4):
            guess, prob = self.rat_tracker.best_guess()
            if prob > 0.5 and rat_ev > 1.0:
                return Move.search(guess)

        # Time management
        turns_left = board.player_worker.turns_left
        time_remaining = time_left()

        if turns_left <= 0:
            return Move.plain(Direction.RIGHT)

        time_per_turn = time_remaining / max(turns_left, 1)

        # Check for immediate carpet opportunities (always take these)
        carpet_dir, carpet_len, carpet_pts = best_carpet_from(board, wx, wy)
        if carpet_dir is not None and carpet_pts >= 2:
            return Move.carpet(carpet_dir, carpet_len)

        # Determine search depth
        if time_per_turn > 3.0:
            max_depth = 4
        elif time_per_turn > 1.5:
            max_depth = 3
        elif time_per_turn > 0.5:
            max_depth = 2
        else:
            max_depth = 1

        # Use minimax if we have time, greedy otherwise
        if max_depth >= 2 and time_remaining > 5.0:
            return find_best_move_minimax(board, rat_ev, time_left, max_depth)
        else:
            return greedy_move(board, rat_ev)
