"""
Challenger Agent v2 - CS3600 Tournament Bot

Upgrades from v1 (motivated by ByteFight match analysis):
  - Negamax + Alpha-Beta + Principal Variation Search (PVS)
  - Iterative deepening to depth 12 with aspiration windows
  - Transposition table (persistent across turns, LRU eviction)
  - Killer moves + history heuristic for move ordering
  - Quiescence search to avoid horizon-effect blunders
  - 8-component phase-aware evaluation focused on PRIME->CARPET chains
  - Ultra-aggressive rat search (target 6+ catches per game)
  - Smart time management targeting 140-180s of 240s budget
  - Auto-carpet for huge payoffs (>=10 pts) to avoid wasting them
"""

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Tuple, List, Optional, Dict
import numpy as np

from game.enums import (
    Direction, MoveType, BOARD_SIZE,
    CARPET_POINTS_TABLE,
)
from game.board import Board
from game.move import Move

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAT_BONUS = 4
RAT_PENALTY = 2
INF = float('inf')
NEG_INF = float('-inf')

# CARPET_POINTS[length] -> points. length 0 unused; length 1 = -1 (penalty).
CARPET_POINTS = [0, -1, 2, 4, 6, 10, 15, 21]

# Phase weights: (early, mid, late) -- phase indices 0, 1, 2
W_SCORE_DIFF   = (1.0,  2.0,  3.5)
W_MY_CARPET    = (0.8,  1.0,  1.2)
W_OPP_CARPET   = (0.4,  0.6,  0.9)
W_MY_CHAIN     = (0.7,  0.5,  0.3)
W_OPP_CHAIN    = (0.35, 0.3,  0.2)
W_PRIMEABLE    = (0.5,  0.4,  0.2)
W_TERRITORY    = (0.15, 0.1,  0.05)
W_MOBILITY     = (0.1,  0.08, 0.05)

# TT flags
TT_EXACT = 0
TT_LOWER = 1  # value is lower bound (beta cutoff in subtree)
TT_UPPER = 2  # value is upper bound (fail-low in subtree)

# Rat HMM model
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
DELTAS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


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

    def top_k_beliefs(self, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        idxs = np.argsort(self.belief)[::-1][:k]
        return [((int(i) % BOARD_SIZE, int(i) // BOARD_SIZE), float(self.belief[i]))
                for i in idxs]

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
# Board helpers
# ---------------------------------------------------------------------------
def _in_bounds(x: int, y: int) -> bool:
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def is_primed(board: Board, x: int, y: int) -> bool:
    if not _in_bounds(x, y):
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return bool(board._primed_mask & bit)


def is_space(board: Board, x: int, y: int) -> bool:
    if not _in_bounds(x, y):
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return bool(board._space_mask & bit)


def is_open(board: Board, x: int, y: int) -> bool:
    if not _in_bounds(x, y):
        return False
    bit = 1 << (y * BOARD_SIZE + x)
    return not bool((board._blocked_mask | board._primed_mask | board._carpet_mask) & bit)


def count_primed_run(board: Board, x: int, y: int, dx: int, dy: int) -> int:
    run = 0
    cx, cy = x + dx, y + dy
    while _in_bounds(cx, cy):
        if is_primed(board, cx, cy):
            run += 1
            cx += dx
            cy += dy
        else:
            break
    return run


def best_carpet_from(board: Board, x: int, y: int) -> Tuple[Optional[Direction], int, int]:
    """Find best carpet roll from (x, y); returns (direction, length, points)."""
    best_dir = None
    best_len = 0
    best_pts = -999
    opp = board.opponent_worker.get_location()

    for d in DIRECTIONS:
        dx, dy = DIR_DELTAS[d]
        run = 0
        cx, cy = x + dx, y + dy
        while _in_bounds(cx, cy):
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
# Phase detection
# ---------------------------------------------------------------------------
def _phase_index(board: Board) -> int:
    """0=early, 1=mid, 2=late."""
    t = min(board.player_worker.turns_left, board.opponent_worker.turns_left)
    if t > 25:
        return 0
    elif t > 10:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _best_carpet_value(board: Board, x: int, y: int, avoid: Tuple[int, int]) -> float:
    """Best carpet points achievable from (x, y), avoiding cell occupied by `avoid`."""
    best = 0
    for dx, dy in DELTAS_4:
        run = 0
        cx, cy = x + dx, y + dy
        while _in_bounds(cx, cy):
            bit = 1 << (cy * BOARD_SIZE + cx)
            if (board._primed_mask & bit) and (cx, cy) != avoid:
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            pts = CARPET_POINTS[min(run, 7)]
            if pts > best:
                best = pts
    return float(best)


def _best_carpet_value_basic(board: Board, x: int, y: int) -> float:
    """Best carpet value from (x, y), ignoring opponent position."""
    best = 0
    for dx, dy in DELTAS_4:
        run = 0
        cx, cy = x + dx, y + dy
        while _in_bounds(cx, cy):
            bit = 1 << (cy * BOARD_SIZE + cx)
            if board._primed_mask & bit:
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            pts = CARPET_POINTS[min(run, 7)]
            if pts > best:
                best = pts
    return float(best)


def _count_chain_potential(board: Board, x: int, y: int, avoid: Tuple[int, int]) -> float:
    """
    Chain potential: sum of future carpet values reachable from (x, y) by
    moving/priming 1-2 steps. Captures the "I can carpet next turn or the turn
    after" pattern.
    """
    score = 0.0
    # For each direction, evaluate "walk once, then look for carpet"
    for dx, dy in DELTAS_4:
        nx, ny = x + dx, y + dy
        if not _in_bounds(nx, ny):
            continue
        # If next cell is open (not blocked/primed/carpet), it's a reachable neighbor
        bit = 1 << (ny * BOARD_SIZE + nx)
        if board._blocked_mask & bit:
            continue
        if (nx, ny) == avoid:
            continue
        # Carpet value from this neighboring square (1-turn lookahead)
        nv = _best_carpet_value(board, nx, ny, avoid)
        if nv > 0:
            # Discounted because it's 1 turn away
            score += nv * 0.5
    return score


def _primeable_bonus(board: Board, x: int, y: int) -> float:
    """
    If my current cell is SPACE and adjacent to a primed cell, priming here
    directly enables a 2-carpet next turn. Return the expected carpet value.
    """
    bit = 1 << (y * BOARD_SIZE + x)
    if (board._primed_mask | board._carpet_mask | board._blocked_mask) & bit:
        return 0.0
    total = 0.0
    for dx, dy in DELTAS_4:
        # Check primed cell adjacent in (dx, dy) direction
        ax, ay = x + dx, y + dy
        if not _in_bounds(ax, ay):
            continue
        abit = 1 << (ay * BOARD_SIZE + ax)
        if not (board._primed_mask & abit):
            continue
        # There's a primed neighbor. If I prime my cell and step the other way,
        # I'd have 2 adjacent primed cells (mine + neighbor). Worth ~2 pts.
        # Also count chain length behind this neighbor for potential.
        behind_run = count_primed_run(board, ax, ay, dx, dy)
        total_line = 1 + 1 + behind_run  # my cell + neighbor + further primed
        pts = CARPET_POINTS[min(total_line, 7)] if total_line >= 2 else 0
        total += pts * 0.5  # Half weight because it's 1 turn away
    return total


def _count_territory(board: Board, x: int, y: int, radius: int = 2) -> int:
    """Count open cells (not blocked/carpet) within radius."""
    count = 0
    occupied = board._blocked_mask | board._carpet_mask
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny):
                continue
            bit = 1 << (ny * BOARD_SIZE + nx)
            if not (occupied & bit):
                count += 1
    return count


def _count_mobility(board: Board, x: int, y: int, other_x: int, other_y: int) -> int:
    """Count directions the worker at (x, y) can step to."""
    blocked = board._blocked_mask | board._primed_mask
    count = 0
    for d in DIRECTIONS:
        dx, dy = DIR_DELTAS[d]
        nx, ny = x + dx, y + dy
        if not _in_bounds(nx, ny):
            continue
        bit = 1 << (ny * BOARD_SIZE + nx)
        if blocked & bit:
            continue
        if (nx, ny) == (other_x, other_y):
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# Main evaluation - always from board.player_worker's perspective
# ---------------------------------------------------------------------------
def evaluate(board: Board) -> float:
    # Terminal: use pure point differential, scaled
    if board.is_game_over():
        diff = board.player_worker.get_points() - board.opponent_worker.get_points()
        if diff > 0:
            return 10000.0 + diff
        elif diff < 0:
            return -10000.0 + diff
        return 0.0

    my_pts = board.player_worker.get_points()
    opp_pts = board.opponent_worker.get_points()
    mx, my_ = board.player_worker.get_location()
    ox, oy = board.opponent_worker.get_location()

    phase = _phase_index(board)
    w_score = W_SCORE_DIFF[phase]
    w_mc    = W_MY_CARPET[phase]
    w_oc    = W_OPP_CARPET[phase]
    w_mch   = W_MY_CHAIN[phase]
    w_och   = W_OPP_CHAIN[phase]
    w_prim  = W_PRIMEABLE[phase]
    w_terr  = W_TERRITORY[phase]
    w_mob   = W_MOBILITY[phase]

    # 1. Score differential
    s = (my_pts - opp_pts) * w_score

    # 2. My immediate carpet value (biggest term when available)
    my_cv = _best_carpet_value(board, mx, my_, (ox, oy))
    s += my_cv * w_mc

    # 3. Opponent immediate carpet threat (subtract)
    opp_cv = _best_carpet_value(board, ox, oy, (mx, my_))
    s -= opp_cv * w_oc

    # 4. My chain potential (1-turn lookahead carpets)
    my_chain = _count_chain_potential(board, mx, my_, (ox, oy))
    s += my_chain * w_mch

    # 5. Opponent chain threat
    opp_chain = _count_chain_potential(board, ox, oy, (mx, my_))
    s -= opp_chain * w_och

    # 6. Primeable bonus (can I enable a 2-carpet by priming here?)
    prim = _primeable_bonus(board, mx, my_) - _primeable_bonus(board, ox, oy) * 0.5
    s += prim * w_prim

    # 7. Territory differential
    my_t = _count_territory(board, mx, my_, 2)
    opp_t = _count_territory(board, ox, oy, 2)
    s += (my_t - opp_t) * w_terr

    # 8. Mobility differential
    my_m = _count_mobility(board, mx, my_, ox, oy)
    opp_m = _count_mobility(board, ox, oy, mx, my_)
    s += (my_m - opp_m) * w_mob

    return s


# ---------------------------------------------------------------------------
# Transposition Table (LRU)
# ---------------------------------------------------------------------------
class TranspositionTable:
    def __init__(self, max_size: int = 2_000_000):
        self.table: OrderedDict = OrderedDict()
        self.max_size = max_size

    @staticmethod
    def _key(board: Board) -> tuple:
        return (
            board._primed_mask,
            board._carpet_mask,
            board.player_worker.position,
            board.opponent_worker.position,
            board.player_worker.points,
            board.opponent_worker.points,
            board.turn_count,
        )

    def get(self, board: Board):
        k = self._key(board)
        entry = self.table.get(k)
        if entry is not None:
            self.table.move_to_end(k)
        return entry

    def put(self, board: Board, depth: int, flag: int, value: float, best_move_key):
        k = self._key(board)
        self.table[k] = (depth, flag, value, best_move_key)
        self.table.move_to_end(k)
        # Evict 25% oldest if over capacity
        if len(self.table) > self.max_size:
            evict = len(self.table) // 4
            for _ in range(evict):
                self.table.popitem(last=False)

    def clear(self):
        self.table.clear()


# ---------------------------------------------------------------------------
# Killer table + history heuristic
# ---------------------------------------------------------------------------
class KillerTable:
    def __init__(self, max_ply: int = 32):
        self.killers: List[List[Optional[tuple]]] = [[None, None] for _ in range(max_ply)]

    def store(self, ply: int, mkey: tuple):
        if ply >= len(self.killers):
            return
        k = self.killers[ply]
        if k[0] == mkey or k[1] == mkey:
            return
        k[1] = k[0]
        k[0] = mkey

    def get(self, ply: int) -> Tuple[Optional[tuple], Optional[tuple]]:
        if ply >= len(self.killers):
            return (None, None)
        return (self.killers[ply][0], self.killers[ply][1])


class HistoryTable:
    def __init__(self):
        self.table: Dict[tuple, int] = {}

    def add(self, mkey: tuple, bonus: int):
        self.table[mkey] = self.table.get(mkey, 0) + bonus

    def get(self, mkey: tuple) -> int:
        return self.table.get(mkey, 0)

    def decay(self, factor: int = 2):
        for k in list(self.table.keys()):
            self.table[k] //= factor
            if self.table[k] == 0:
                del self.table[k]


def move_key(m: Move) -> tuple:
    if m.move_type == MoveType.SEARCH:
        return (int(m.move_type), m.search_loc)
    return (int(m.move_type), int(m.direction) if m.direction is not None else -1, m.roll_length)


# ---------------------------------------------------------------------------
# Search context
# ---------------------------------------------------------------------------
class SearchContext:
    def __init__(self, tt: TranspositionTable, killers: KillerTable,
                 history: HistoryTable, deadline: float):
        self.tt = tt
        self.killers = killers
        self.history = history
        self.deadline = deadline
        self.nodes = 0


class SearchTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Move ordering (7 tiers)
# ---------------------------------------------------------------------------
def order_moves(board: Board, tt_best_key, killers: Tuple,
                history: HistoryTable) -> List[Move]:
    moves = board.get_valid_moves(exclude_search=True)
    if not moves:
        return moves

    mx, my_ = board.player_worker.get_location()

    k1, k2 = killers

    def priority(m: Move) -> float:
        mk = move_key(m)

        # Tier 1: TT best move
        if tt_best_key is not None and mk == tt_best_key:
            return 10_000_000.0

        # Tier 2: carpets by immediate payoff
        if m.move_type == MoveType.CARPET:
            pts = CARPET_POINTS[min(m.roll_length, 7)]
            return 5_000_000.0 + pts * 1000.0

        # Tier 3: killer moves
        if mk == k1:
            return 4_000_000.0
        if mk == k2:
            return 3_900_000.0

        # Tier 4: high-history moves
        hist = history.get(mk)

        # Tier 5-7: primes and plains scored by chain potential
        if m.move_type == MoveType.PRIME:
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = mx + dx, my_ + dy
            # Chain value: line length if I prime here + run behind
            behind = count_primed_run(board, mx, my_, -dx, -dy)
            total_line = 1 + behind
            line_pts = CARPET_POINTS[min(total_line, 7)] if total_line >= 2 else 0
            # Also consider destination's carpet potential
            dest_val = _best_carpet_value_basic(board, nx, ny)
            nearby_primed = 0
            for ddx, ddy in DELTAS_4:
                if is_primed(board, mx + ddx, my_ + ddy):
                    nearby_primed += 1
            return 1_000_000.0 + line_pts * 200.0 + dest_val * 50.0 + nearby_primed * 30.0 + hist * 0.01

        if m.move_type == MoveType.PLAIN:
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = mx + dx, my_ + dy
            dest_val = _best_carpet_value_basic(board, nx, ny)
            nearby_primed = 0
            for ddx, ddy in DELTAS_4:
                if is_primed(board, nx + ddx, ny + ddy):
                    nearby_primed += 1
            return 500_000.0 + dest_val * 30.0 + nearby_primed * 20.0 + hist * 0.01

        return hist * 0.01

    moves.sort(key=priority, reverse=True)
    return moves


# ---------------------------------------------------------------------------
# Quiescence search
# ---------------------------------------------------------------------------
def quiescence(board: Board, alpha: float, beta: float,
               ctx: SearchContext, extensions_left: int) -> float:
    ctx.nodes += 1
    if ctx.nodes & 0x7FF == 0:  # every 2048 nodes
        if time.time() + 0.1 > ctx.deadline:
            raise SearchTimeout()

    stand_pat = evaluate(board)

    if extensions_left <= 0 or board.is_game_over():
        return stand_pat

    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    # Only tactical (carpet) moves
    all_moves = board.get_valid_moves(exclude_search=True)
    tactical = [m for m in all_moves
                if m.move_type == MoveType.CARPET
                and CARPET_POINTS[min(m.roll_length, 7)] >= 2]

    if not tactical:
        return stand_pat

    # Order by points desc
    tactical.sort(key=lambda m: CARPET_POINTS[min(m.roll_length, 7)], reverse=True)

    best = stand_pat
    for m in tactical:
        child = board.forecast_move(m, check_ok=False)
        if child is None:
            continue
        child.reverse_perspective()
        try:
            value = -quiescence(child, -beta, -alpha, ctx, extensions_left - 1)
        except SearchTimeout:
            raise
        if value > best:
            best = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            return value
    return best


# ---------------------------------------------------------------------------
# Negamax with PVS + TT + killers + history
# ---------------------------------------------------------------------------
def negamax_pvs(board: Board, depth: int, alpha: float, beta: float,
                ply: int, ctx: SearchContext) -> float:
    ctx.nodes += 1
    if ctx.nodes & 0x7FF == 0:
        if time.time() + 0.1 > ctx.deadline:
            raise SearchTimeout()

    orig_alpha = alpha

    # TT probe
    tt_entry = ctx.tt.get(board)
    tt_best_key = None
    if tt_entry is not None:
        tt_depth, tt_flag, tt_value, tt_move_key = tt_entry
        tt_best_key = tt_move_key
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_value
            if tt_flag == TT_LOWER and tt_value >= beta:
                return tt_value
            if tt_flag == TT_UPPER and tt_value <= alpha:
                return tt_value

    if board.is_game_over():
        return evaluate(board)

    if depth <= 0:
        return quiescence(board, alpha, beta, ctx, 3)

    killers = ctx.killers.get(ply)
    moves = order_moves(board, tt_best_key, killers, ctx.history)
    if not moves:
        return evaluate(board)

    best_value = NEG_INF
    best_mkey = None
    first = True

    for m in moves:
        child = board.forecast_move(m, check_ok=False)
        if child is None:
            continue
        child.reverse_perspective()

        if first:
            value = -negamax_pvs(child, depth - 1, -beta, -alpha, ply + 1, ctx)
            first = False
        else:
            # Null-window (scout) search
            value = -negamax_pvs(child, depth - 1, -alpha - 1, -alpha, ply + 1, ctx)
            if alpha < value < beta:
                # Re-search with full window
                value = -negamax_pvs(child, depth - 1, -beta, -alpha, ply + 1, ctx)

        # Length-1 carpet opportunity-cost penalty: burning a primed cell
        # that could have formed a length-2+ chain (worth +2) is an implicit
        # cost beyond the -1 immediate point hit. Only take length-1 when
        # tactically clearly better (rat catch, forced navigation, blocking).
        if m.move_type == MoveType.CARPET and m.roll_length == 1:
            value -= 2.0

        if value > best_value:
            best_value = value
            best_mkey = move_key(m)

        if value > alpha:
            alpha = value

        if alpha >= beta:
            # Beta cutoff: store killer (non-carpet) + history bonus
            mk = move_key(m)
            if m.move_type != MoveType.CARPET:
                ctx.killers.store(ply, mk)
            ctx.history.add(mk, depth * depth)
            break

    # Store in TT
    if best_value <= orig_alpha:
        flag = TT_UPPER
    elif best_value >= beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT
    ctx.tt.put(board, depth, flag, best_value, best_mkey)

    return best_value


# ---------------------------------------------------------------------------
# Iterative deepening with aspiration windows
# ---------------------------------------------------------------------------
def iterative_deepening_search(board: Board, tt: TranspositionTable,
                               history: HistoryTable, time_budget: float,
                               max_depth: int = 12) -> Tuple[Move, int, int, float]:
    start = time.time()
    deadline = start + time_budget
    killers = KillerTable()
    ctx = SearchContext(tt, killers, history, deadline)

    # Initial ordering at root
    tt_entry = tt.get(board)
    root_tt_key = tt_entry[3] if tt_entry else None
    initial_moves = order_moves(board, root_tt_key, killers.get(0), history)
    if not initial_moves:
        return Move.plain(Direction.RIGHT), 0, 0, 0.0

    best_move = initial_moves[0]
    best_score = NEG_INF
    depth_reached = 0
    last_score = 0.0

    for depth in range(1, max_depth + 1):
        if time.time() + 0.15 > deadline:
            break

        iteration_best_move = None
        iteration_best_score = NEG_INF
        completed = True

        # Re-read TT for ordering (may have updated from previous iteration)
        tt_entry = tt.get(board)
        root_tt_key = tt_entry[3] if tt_entry else None
        ordered = order_moves(board, root_tt_key, killers.get(0), history)

        # Aspiration windows
        if depth >= 3:
            window = 50.0
            alpha = last_score - window
            beta = last_score + window
        else:
            alpha = NEG_INF
            beta = INF

        retries = 0
        max_retries = 3
        while True:
            iteration_best_move = None
            iteration_best_score = NEG_INF
            cur_alpha = alpha
            cur_beta = beta
            first = True
            aborted = False

            try:
                for m in ordered:
                    if time.time() + 0.05 > deadline:
                        aborted = True
                        break
                    child = board.forecast_move(m, check_ok=False)
                    if child is None:
                        continue
                    child.reverse_perspective()

                    if first:
                        score = -negamax_pvs(child, depth - 1, -cur_beta, -cur_alpha, 1, ctx)
                        first = False
                    else:
                        score = -negamax_pvs(child, depth - 1, -cur_alpha - 1, -cur_alpha, 1, ctx)
                        if cur_alpha < score < cur_beta:
                            score = -negamax_pvs(child, depth - 1, -cur_beta, -cur_alpha, 1, ctx)

                    # Length-1 carpet opportunity-cost penalty (see negamax_pvs).
                    if m.move_type == MoveType.CARPET and m.roll_length == 1:
                        score -= 2.0

                    if score > iteration_best_score:
                        iteration_best_score = score
                        iteration_best_move = m
                    if score > cur_alpha:
                        cur_alpha = score
            except SearchTimeout:
                aborted = True

            if aborted:
                completed = False
                break

            # Aspiration re-search if outside window
            if iteration_best_score <= alpha and retries < max_retries:
                alpha = NEG_INF  # widen alpha
                retries += 1
                continue
            if iteration_best_score >= beta and retries < max_retries:
                beta = INF
                retries += 1
                continue
            break

        if iteration_best_move is not None and completed:
            best_move = iteration_best_move
            best_score = iteration_best_score
            last_score = iteration_best_score
            depth_reached = depth
            ctx.tt.put(board, depth, TT_EXACT, iteration_best_score, move_key(iteration_best_move))
        elif iteration_best_move is not None and iteration_best_score > best_score + 100:
            # Partial result is clearly better, accept
            best_move = iteration_best_move
            best_score = iteration_best_score

        if not completed:
            break

    return best_move, depth_reached, ctx.nodes, time.time() - start


# ---------------------------------------------------------------------------
# Greedy fallback
# ---------------------------------------------------------------------------
def score_move(board: Board, m: Move) -> float:
    mx, my_ = board.player_worker.get_location()
    if m.direction is not None:
        dx, dy = DIR_DELTAS.get(m.direction, (0, 0))
    else:
        dx, dy = 0, 0

    if m.move_type == MoveType.CARPET:
        pts = CARPET_POINTS[min(m.roll_length, 7)]
        lx = mx + dx * m.roll_length
        ly = my_ + dy * m.roll_length
        future = _best_carpet_value_basic(board, lx, ly) * 0.3
        return pts * 10 + future + 100

    if m.move_type == MoveType.PRIME:
        nx, ny = mx + dx, my_ + dy
        behind = count_primed_run(board, mx, my_, -dx, -dy)
        total_line = 1 + behind
        line_pts = CARPET_POINTS[min(total_line, 7)] if total_line >= 2 else 0
        dest_val = _best_carpet_value_basic(board, nx, ny)
        adj_primed = 0
        for ddx, ddy in DELTAS_4:
            if is_primed(board, mx + ddx, my_ + ddy):
                adj_primed += 1
            if is_primed(board, nx + ddx, ny + ddy):
                adj_primed += 1
        return 1 + line_pts * 2 + dest_val * 1.5 + adj_primed * 0.8 + 10

    # PLAIN
    nx, ny = mx + dx, my_ + dy
    dest_val = _best_carpet_value_basic(board, nx, ny)
    adj = 0
    for ddx, ddy in DELTAS_4:
        if is_space(board, nx + ddx, ny + ddy):
            adj += 1
        if is_primed(board, nx + ddx, ny + ddy):
            adj += 2
    return dest_val * 1.0 + adj * 0.3


def greedy_move(board: Board) -> Move:
    moves = board.get_valid_moves(exclude_search=True)
    if not moves:
        return Move.plain(Direction.RIGHT)
    best = moves[0]
    best_s = NEG_INF
    for m in moves:
        s = score_move(board, m)
        if s > best_s:
            best_s = s
            best = m
    return best


# ---------------------------------------------------------------------------
# Player Agent
# ---------------------------------------------------------------------------
class PlayerAgent:
    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.rat_tracker: Optional[RatTracker] = None
        if transition_matrix is not None:
            T = np.asarray(transition_matrix, dtype=np.float64)
            self.rat_tracker = RatTracker(T)
        self.observations_since_reset = 0
        self.search_cooldown = 0

        self.tt = TranspositionTable(max_size=2_000_000)
        self.history = HistoryTable()

        # Stats
        self.last_depth = 0
        self.last_nodes = 0
        self.last_time = 0.0
        self.total_time_used = 0.0
        self.turn_number = 0

    def commentate(self):
        return (f"Challenger v2 | d={self.last_depth} n={self.last_nodes} "
                f"t={self.last_time:.2f}s total={self.total_time_used:.1f}s")

    # ---------- Time management ----------
    def _compute_time_budget(self, board: Board, time_remaining: float) -> float:
        turns_left = max(board.player_worker.turns_left, 1)
        base = time_remaining / turns_left

        if turns_left > 25:
            budget = base * 0.7
        elif turns_left > 10:
            budget = base * 1.6
        else:
            budget = base * 0.9

        budget = max(0.5, min(10.0, budget))

        # Safety floors
        if time_remaining < 30:
            budget = min(budget, 1.0)
        if time_remaining < 15:
            budget = min(budget, 0.5)
        if time_remaining < 5:
            budget = min(budget, 0.3)

        return budget

    # ---------- Rat search decision ----------
    def _rat_thresholds(self, board: Board) -> Tuple[float, float]:
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        turns_left = min(board.player_worker.turns_left, board.opponent_worker.turns_left)

        if diff <= -10:
            p, e = 0.20, -0.5
        elif diff <= -5:
            p, e = 0.25, 0.0
        elif diff < 0:
            p, e = 0.30, 0.3
        elif diff < 5:
            p, e = 0.35, 0.6
        else:
            p, e = 0.45, 1.2

        if turns_left <= 5:
            p = max(0.15, p - 0.1)
            e = max(-1.0, e - 0.3)

        return p, e

    def _should_search_rat(self, board: Board, rat_ev: float, best_carpet_pts: int) -> bool:
        if self.rat_tracker is None:
            return False
        if self.search_cooldown > 0:
            return False

        turns_left = min(board.player_worker.turns_left, board.opponent_worker.turns_left)
        min_obs = 2 if turns_left <= 5 else 3
        if self.observations_since_reset < min_obs:
            return False

        _, prob = self.rat_tracker.best_guess()
        p_thr, e_thr = self._rat_thresholds(board)
        if prob < p_thr:
            return False
        if rat_ev < e_thr:
            return False

        # Compare opportunity cost against a big carpet
        if best_carpet_pts >= 6 and rat_ev < best_carpet_pts - 1:
            return False

        return True

    # ---------- Play ----------
    def play(self, board: Board, sensor_data: Tuple, time_left: Callable) -> Move:
        play_start = time.time()
        self.turn_number += 1

        # Decay history each turn
        self.history.decay(2)

        noise, reported_dist = sensor_data
        wx, wy = board.player_worker.get_location()

        # Handle rat catches/resets
        rat_reset = False
        if board.opponent_search[0] is not None and board.opponent_search[1]:
            rat_reset = True
        if board.player_search[0] is not None and board.player_search[1]:
            rat_reset = True

        if rat_reset and self.rat_tracker:
            self.rat_tracker.reset_after_catch()
            self.observations_since_reset = 0
            self.search_cooldown = 2  # shorter cooldown

        if board.player_search[0] is not None and not board.player_search[1]:
            self.search_cooldown = 2

        if self.search_cooldown > 0:
            self.search_cooldown -= 1

        self.observations_since_reset += 1

        # Update rat belief
        rat_ev = -2.0
        if self.rat_tracker:
            self.rat_tracker.update(board, int(noise), reported_dist, wx, wy)
            rat_ev = self.rat_tracker.search_ev()

        # Find best immediate carpet
        carpet_dir, carpet_len, carpet_pts = best_carpet_from(board, wx, wy)
        best_carpet_pts = max(0, carpet_pts if carpet_dir is not None else 0)

        # Rat search decision
        if self._should_search_rat(board, rat_ev, best_carpet_pts):
            guess, _ = self.rat_tracker.best_guess()
            move = Move.search(guess)
            self.last_time = time.time() - play_start
            self.total_time_used += self.last_time
            return move

        # Auto-take huge carpets (length 5+)
        if carpet_dir is not None and carpet_pts >= 10:
            move = Move.carpet(carpet_dir, carpet_len)
            self.last_time = time.time() - play_start
            self.total_time_used += self.last_time
            return move

        turns_left = board.player_worker.turns_left
        time_remaining = time_left()

        if turns_left <= 0:
            move = Move.plain(Direction.RIGHT)
            self.last_time = time.time() - play_start
            self.total_time_used += self.last_time
            return move

        # Very low time: greedy
        if time_remaining < 5.0:
            move = greedy_move(board)
            self.last_time = time.time() - play_start
            self.total_time_used += self.last_time
            return move

        # Compute search budget
        time_budget = self._compute_time_budget(board, time_remaining)

        # Endgame: deeper search (smaller tree)
        max_depth = 12
        if turns_left <= 5:
            max_depth = min(20, turns_left * 2 + 6)

        # Iterative deepening
        best_move, depth_reached, nodes, _ = iterative_deepening_search(
            board, self.tt, self.history, time_budget, max_depth=max_depth
        )

        if best_move is None:
            best_move = greedy_move(board)

        self.last_depth = depth_reached
        self.last_nodes = nodes
        self.last_time = time.time() - play_start
        self.total_time_used += self.last_time

        return best_move
