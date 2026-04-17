"""
Challenger Agent v7.3 - CS3600 Tournament Bot

v7.3 changes (on v7.1):
  1. Numba JIT compilation of all evaluation helpers (~20-50x eval speedup).
     Graceful fallback to pure Python if numba unavailable.
  2. Shared-memory transposition table (numpy uint64 array, lockless XOR trick)
     replaces OrderedDict TT. Enables cross-process sharing.
  3. Lazy SMP: 2 worker processes search in parallel, writing to shared TT.
     Workers improve TT quality; main process always selects the move.
  4. Board serialization for IPC (14 primitives, bypass __init__).

v7.1: Make/unmake, NMP, LMR.
v7: Board-wide primed infra eval, flattened time, HMM belief zeroing.

Core engine: Negamax+PVS, iterative deepening (depth 12+), shared TT (2M),
killer+history+LMR+NMP, quiescence, make/unmake, Lazy SMP, JIT eval.
"""

import time
import os
import struct
import multiprocessing
import multiprocessing.shared_memory
from collections import OrderedDict
from collections.abc import Callable
from typing import Tuple, List, Optional, Dict
import numpy as np

# Conditional numba import — graceful fallback to pure Python
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
    def njit(*args, **kwargs):
        kwargs.setdefault('cache', False)
        return _njit(*args, **kwargs)
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        """No-op decorator when numba unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

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
INF = 1e18       # numba-safe (no float('inf'))
NEG_INF = -1e18

# CARPET_POINTS[length] -> points. length 0 unused; length 1 = -1 (penalty).
CARPET_POINTS = [0, -1, 2, 4, 6, 10, 15, 21]

# Numpy constant arrays for JIT functions
_CARPET_POINTS_NP = np.array([0, -1, 2, 4, 6, 10, 15, 21], dtype=np.float64)

# Pack all 11 weight tuples into an 11×3 matrix for numba
# Row order: SCORE_DIFF, MY_CARPET, OPP_CARPET, MY_CHAIN, OPP_CHAIN,
#            PRIMEABLE, TERRITORY, MOBILITY, BELIEF, MY_INFRA, OPP_INFRA

# Phase weights: (early, mid, late) -- phase indices 0, 1, 2
W_SCORE_DIFF   = (1.0,  2.0,  3.5)
W_MY_CARPET    = (2.0,  3.0,  4.0)   # v6: was (0.8, 1.0, 1.2) — carpet pts r=+0.692 with score
W_OPP_CARPET   = (1.0,  1.5,  2.5)   # v6: was (0.4, 0.6, 0.9)
W_MY_CHAIN     = (0.7,  0.5,  0.3)
W_OPP_CHAIN    = (0.35, 0.3,  0.2)
W_PRIMEABLE    = (0.5,  0.4,  0.2)
W_TERRITORY    = (0.15, 0.1,  0.05)
W_MOBILITY     = (0.1,  0.08, 0.05)
# CS-1c: belief-guided movement (small — closer to high-P(rat) is mildly good)
W_BELIEF       = (0.15, 0.20, 0.10)
# v7: Board-wide primed infrastructure (structural fix for prime→leave→return pattern)
W_MY_INFRA     = (0.15, 0.25, 0.10)
W_OPP_INFRA    = (0.08, 0.15, 0.08)

_WEIGHTS_NP = np.array([
    W_SCORE_DIFF, W_MY_CARPET, W_OPP_CARPET, W_MY_CHAIN, W_OPP_CHAIN,
    W_PRIMEABLE, W_TERRITORY, W_MOBILITY, W_BELIEF, W_MY_INFRA, W_OPP_INFRA,
], dtype=np.float64)  # shape (11, 3)

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
# Fast direction→(dx,dy) via IntEnum index (UP=0, RIGHT=1, DOWN=2, LEFT=3)
_DIR_DXY = ((0, -1), (1, 0), (0, 1), (-1, 0))


# ---------------------------------------------------------------------------
# CS-1c: Belief-guided movement
# Precompute one proximity score per (x, y) per turn so evaluate() is O(1).
# _BELIEF_PROX[i] = sum over cells c of belief[c] * max(0, 8 - manhattan(i, c))
# Higher when worker at i is close to high-probability rat cells.
# ---------------------------------------------------------------------------
_BELIEF_PROX: Optional[np.ndarray] = None


def set_belief_proximity(belief: Optional[np.ndarray]) -> None:
    global _BELIEF_PROX
    if belief is None:
        _BELIEF_PROX = None
        return
    result = np.zeros(64, dtype=np.float64)
    for i in range(64):
        x = i % BOARD_SIZE
        y = i // BOARD_SIZE
        dists = np.abs(ALL_X - x) + np.abs(ALL_Y - y)
        proximity_vals = np.maximum(0.0, 8.0 - dists)
        result[i] = float((belief * proximity_vals).sum())
    _BELIEF_PROX = result


def _belief_prox_at(x: int, y: int) -> float:
    if _BELIEF_PROX is None:
        return 0.0
    return float(_BELIEF_PROX[y * BOARD_SIZE + x])


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
# JIT evaluation helpers (numba @njit — pure int64/float64 args, no Python objects)
# ---------------------------------------------------------------------------
@njit(cache=False)
def _count_primed_run_jit(primed_mask, x, y, dx, dy):
    """Count consecutive primed cells starting from (x+dx, y+dy)."""
    run = 0
    cx, cy = x + dx, y + dy
    while 0 <= cx < 8 and 0 <= cy < 8:
        if primed_mask & (1 << (cy * 8 + cx)):
            run += 1
            cx += dx
            cy += dy
        else:
            break
    return run


@njit(cache=False)
def _best_carpet_value_jit(primed_mask, x, y, avoid_x, avoid_y, carpet_pts):
    """Best carpet points from (x,y) avoiding cell (avoid_x, avoid_y)."""
    best = 0.0
    for di in range(4):
        if di == 0:
            dx, dy = 1, 0
        elif di == 1:
            dx, dy = -1, 0
        elif di == 2:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        run = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < 8 and 0 <= cy < 8:
            if (primed_mask & (1 << (cy * 8 + cx))) and not (cx == avoid_x and cy == avoid_y):
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            pts = carpet_pts[min(run, 7)]
            if pts > best:
                best = pts
    return best


@njit(cache=False)
def _best_carpet_value_basic_jit(primed_mask, x, y, carpet_pts):
    """Best carpet value from (x,y), no avoid."""
    best = 0.0
    for di in range(4):
        if di == 0:
            dx, dy = 1, 0
        elif di == 1:
            dx, dy = -1, 0
        elif di == 2:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        run = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < 8 and 0 <= cy < 8:
            if primed_mask & (1 << (cy * 8 + cx)):
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            pts = carpet_pts[min(run, 7)]
            if pts > best:
                best = pts
    return best


@njit(cache=False)
def _count_chain_potential_jit(primed_mask, blocked_mask, x, y, avoid_x, avoid_y, carpet_pts):
    """Chain potential: sum of discounted carpet values reachable 1 step away."""
    score = 0.0
    for di in range(4):
        if di == 0:
            dx, dy = 1, 0
        elif di == 1:
            dx, dy = -1, 0
        elif di == 2:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 8 and 0 <= ny < 8):
            continue
        bit = 1 << (ny * 8 + nx)
        if blocked_mask & bit:
            continue
        if nx == avoid_x and ny == avoid_y:
            continue
        nv = _best_carpet_value_jit(primed_mask, nx, ny, avoid_x, avoid_y, carpet_pts)
        if nv > 0:
            score += nv * 0.5
    return score


@njit(cache=False)
def _primeable_bonus_jit(primed_mask, carpet_mask, blocked_mask, x, y, carpet_pts):
    """Bonus for standing on SPACE adjacent to primed cells (enables future carpet)."""
    bit = 1 << (y * 8 + x)
    if (primed_mask | carpet_mask | blocked_mask) & bit:
        return 0.0
    total = 0.0
    for di in range(4):
        if di == 0:
            dx, dy = 1, 0
        elif di == 1:
            dx, dy = -1, 0
        elif di == 2:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        ax, ay = x + dx, y + dy
        if not (0 <= ax < 8 and 0 <= ay < 8):
            continue
        abit = 1 << (ay * 8 + ax)
        if not (primed_mask & abit):
            continue
        behind_run = _count_primed_run_jit(primed_mask, ax, ay, dx, dy)
        total_line = 1 + 1 + behind_run
        if total_line >= 2:
            pts = carpet_pts[min(total_line, 7)]
            total += pts * 0.5
    return total


@njit(cache=False)
def _count_territory_jit(blocked_mask, carpet_mask, x, y, radius):
    """Count open cells within manhattan radius."""
    count = 0
    occupied = blocked_mask | carpet_mask
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            bit = 1 << (ny * 8 + nx)
            if not (occupied & bit):
                count += 1
    return count


@njit(cache=False)
def _count_mobility_jit(blocked_mask, primed_mask, x, y, other_x, other_y):
    """Count directions worker at (x,y) can move."""
    blocked = blocked_mask | primed_mask
    count = 0
    for di in range(4):
        if di == 0:
            dx, dy = 1, 0
        elif di == 1:
            dx, dy = -1, 0
        elif di == 2:
            dx, dy = 0, 1
        else:
            dx, dy = 0, -1
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 8 and 0 <= ny < 8):
            continue
        bit = 1 << (ny * 8 + nx)
        if blocked & bit:
            continue
        if nx == other_x and ny == other_y:
            continue
        count += 1
    return count


@njit(cache=False)
def _primed_infrastructure_jit(primed_mask, mx, my_, ox, oy, carpet_pts):
    """Board-wide primed run valuation. Returns (my_value, opp_value)."""
    if primed_mask == 0:
        return 0.0, 0.0
    my_val = 0.0
    opp_val = 0.0
    # Horizontal runs
    for row in range(8):
        base = row * 8
        col = 0
        while col < 8:
            if primed_mask & (1 << (base + col)):
                start = col
                col += 1
                while col < 8 and (primed_mask & (1 << (base + col))):
                    col += 1
                rlen = col - start
                if rlen >= 2:
                    pts = carpet_pts[min(rlen, 7)]
                    md = 99
                    od = 99
                    if start > 0:
                        md = abs(mx - start + 1) + abs(my_ - row)
                        od = abs(ox - start + 1) + abs(oy - row)
                    if col < 8:
                        d = abs(mx - col) + abs(my_ - row)
                        if d < md:
                            md = d
                        d = abs(ox - col) + abs(oy - row)
                        if d < od:
                            od = d
                    disc_m = 1.0 - md * 0.12
                    if disc_m < 0.05:
                        disc_m = 0.05
                    disc_o = 1.0 - od * 0.12
                    if disc_o < 0.05:
                        disc_o = 0.05
                    my_val += pts * disc_m
                    opp_val += pts * disc_o
            else:
                col += 1
    # Vertical runs
    for col in range(8):
        row = 0
        while row < 8:
            if primed_mask & (1 << (row * 8 + col)):
                start = row
                row += 1
                while row < 8 and (primed_mask & (1 << (row * 8 + col))):
                    row += 1
                rlen = row - start
                if rlen >= 2:
                    pts = carpet_pts[min(rlen, 7)]
                    md = 99
                    od = 99
                    if start > 0:
                        md = abs(mx - col) + abs(my_ - start + 1)
                        od = abs(ox - col) + abs(oy - start + 1)
                    if row < 8:
                        d = abs(mx - col) + abs(my_ - row)
                        if d < md:
                            md = d
                        d = abs(ox - col) + abs(oy - row)
                        if d < od:
                            od = d
                    disc_m = 1.0 - md * 0.12
                    if disc_m < 0.05:
                        disc_m = 0.05
                    disc_o = 1.0 - od * 0.12
                    if disc_o < 0.05:
                        disc_o = 0.05
                    my_val += pts * disc_m
                    opp_val += pts * disc_o
            else:
                row += 1
    return my_val, opp_val


@njit(cache=False)
def _evaluate_jit(primed_mask, carpet_mask, space_mask, blocked_mask,
                  mx, my_, ox, oy, my_pts, opp_pts, my_turns, opp_turns,
                  belief_prox, carpet_pts, weights):
    """Full evaluation in JIT. weights is (11, 3) matrix."""
    # Phase
    t = my_turns if my_turns < opp_turns else opp_turns
    if t > 25:
        phase = 0
    elif t > 10:
        phase = 1
    else:
        phase = 2

    w_score = weights[0, phase]
    w_mc    = weights[1, phase]
    w_oc    = weights[2, phase]
    w_mch   = weights[3, phase]
    w_och   = weights[4, phase]
    w_prim  = weights[5, phase]
    w_terr  = weights[6, phase]
    w_mob   = weights[7, phase]
    w_bel   = weights[8, phase]
    w_mi    = weights[9, phase]
    w_oi    = weights[10, phase]

    # 1. Score diff
    s = (my_pts - opp_pts) * w_score

    # 2. My carpet value
    my_cv = _best_carpet_value_jit(primed_mask, mx, my_, ox, oy, carpet_pts)
    s += my_cv * w_mc

    # 3. Opp carpet threat
    opp_cv = _best_carpet_value_jit(primed_mask, ox, oy, mx, my_, carpet_pts)
    s -= opp_cv * w_oc

    # Prime amplification
    prime_amp = 2.0 if my_cv < 4.0 else 1.0

    # 4. My chain potential
    my_chain = _count_chain_potential_jit(primed_mask, blocked_mask, mx, my_, ox, oy, carpet_pts)
    s += my_chain * w_mch * prime_amp

    # 5. Opp chain
    opp_chain = _count_chain_potential_jit(primed_mask, blocked_mask, ox, oy, mx, my_, carpet_pts)
    s -= opp_chain * w_och

    # 6. Primeable bonus
    my_prim = _primeable_bonus_jit(primed_mask, carpet_mask, blocked_mask, mx, my_, carpet_pts)
    opp_prim = _primeable_bonus_jit(primed_mask, carpet_mask, blocked_mask, ox, oy, carpet_pts)
    s += (my_prim - opp_prim * 0.5) * w_prim * prime_amp

    # 7. Territory
    my_t = _count_territory_jit(blocked_mask, carpet_mask, mx, my_, 2)
    opp_t = _count_territory_jit(blocked_mask, carpet_mask, ox, oy, 2)
    s += (my_t - opp_t) * w_terr

    # 8. Mobility
    my_m = _count_mobility_jit(blocked_mask, primed_mask, mx, my_, ox, oy)
    opp_m = _count_mobility_jit(blocked_mask, primed_mask, ox, oy, mx, my_)
    s += (my_m - opp_m) * w_mob

    # 9. Belief proximity
    my_bp = belief_prox[my_ * 8 + mx]
    opp_bp = belief_prox[oy * 8 + ox]
    s += (my_bp - opp_bp * 0.4) * w_bel

    # 10. Primed infrastructure
    my_infra, opp_infra = _primed_infrastructure_jit(primed_mask, mx, my_, ox, oy, carpet_pts)
    s += my_infra * w_mi * prime_amp
    s -= opp_infra * w_oi

    return s


# ---------------------------------------------------------------------------
# Evaluation helpers (pure Python — used by order_moves/score_move fallback)
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
# Main evaluation - JIT wrapper (falls back to pure Python if numba unavailable)
# ---------------------------------------------------------------------------
_ZERO_BELIEF = np.zeros(64, dtype=np.float64)


def evaluate(board: Board) -> float:
    if board.is_game_over():
        diff = board.player_worker.get_points() - board.opponent_worker.get_points()
        if diff > 0:
            return 10000.0 + diff
        elif diff < 0:
            return -10000.0 + diff
        return 0.0
    mx, my_ = board.player_worker.get_location()
    ox, oy = board.opponent_worker.get_location()
    bp = _BELIEF_PROX if _BELIEF_PROX is not None else _ZERO_BELIEF
    return _evaluate_jit(
        board._primed_mask, board._carpet_mask, board._space_mask, board._blocked_mask,
        mx, my_, ox, oy,
        board.player_worker.get_points(), board.opponent_worker.get_points(),
        board.player_worker.turns_left, board.opponent_worker.turns_left,
        bp, _CARPET_POINTS_NP, _WEIGHTS_NP)


# ---------------------------------------------------------------------------
# Transposition Table — shared-memory numpy array for Lazy SMP
# Layout: TT_SIZE entries × 4 uint64 columns in flat numpy array
#   [i*4+0] = verification_key (hash XOR packed_data — lockless XOR trick)
#   [i*4+1] = packed_data: depth(4) | flag(2) | score_fixed(26) | move_key_hash(32)
#   [i*4+2] = full_hash (64-bit board hash for collision detection)
#   [i*4+3] = generation counter (for aging)
# ---------------------------------------------------------------------------
TT_SIZE = 1 << 21  # 2M entries
TT_BYTES = TT_SIZE * 4 * 8  # 64MB

def _board_hash(board: Board) -> int:
    """Fast board hash for TT indexing. Returns positive 64-bit int."""
    pm = board._primed_mask & 0xFFFFFFFFFFFFFFFF
    cm = board._carpet_mask & 0xFFFFFFFFFFFFFFFF
    px, py = board.player_worker.position
    ox, oy = board.opponent_worker.position
    pp = board.player_worker.points
    op_ = board.opponent_worker.points
    tc = board.turn_count
    # Mix via FNV-1a-like scheme
    h = 0xcbf29ce484222325
    for v in (pm, cm, px, py, ox, oy, pp, op_, tc):
        h ^= (v & 0xFFFFFFFFFFFFFFFF)
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _pack_tt_data(depth: int, flag: int, value: float, mkh: int) -> int:
    """Pack TT entry into a single uint64."""
    # depth: 0-15 (4 bits), flag: 0-2 (2 bits), score: fixed-point 26 bits, mkh: 32 bits
    score_fixed = int(value * 1000.0) + (1 << 25)  # offset to make positive
    score_fixed = max(0, min((1 << 26) - 1, score_fixed))
    return ((depth & 0xF) << 60) | ((flag & 0x3) << 58) | ((score_fixed & 0x3FFFFFF) << 32) | (mkh & 0xFFFFFFFF)


def _unpack_tt_data(packed: int):
    """Unpack TT entry. Returns (depth, flag, value, move_key_hash)."""
    depth = (packed >> 60) & 0xF
    flag = (packed >> 58) & 0x3
    score_fixed = (packed >> 32) & 0x3FFFFFF
    value = (score_fixed - (1 << 25)) / 1000.0
    mkh = packed & 0xFFFFFFFF
    return depth, flag, value, mkh


def _move_key_hash(mkey) -> int:
    """Hash a move_key tuple to 32-bit int."""
    if mkey is None:
        return 0
    return hash(mkey) & 0xFFFFFFFF


class TranspositionTable:
    """Shared-memory TT usable across processes for Lazy SMP."""

    def __init__(self, shm_name: str = None, create: bool = True):
        self._owns_shm = create
        self.generation = 0
        try:
            if create:
                if shm_name is None:
                    shm_name = f"challenger_tt_{os.getpid()}"
                try:
                    # Clean up stale shm from previous runs
                    old = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False)
                    old.close()
                    old.unlink()
                except FileNotFoundError:
                    pass
                self._shm = multiprocessing.shared_memory.SharedMemory(
                    name=shm_name, create=True, size=TT_BYTES)
            else:
                self._shm = multiprocessing.shared_memory.SharedMemory(
                    name=shm_name, create=False)
            self._arr = np.ndarray((TT_SIZE * 4,), dtype=np.uint64, buffer=self._shm.buf)
            if create:
                self._arr[:] = 0
            self.shm_name = self._shm.name
            self._shared = True
        except Exception:
            # Fallback: local numpy array (no shared memory)
            self._shm = None
            self._arr = np.zeros(TT_SIZE * 4, dtype=np.uint64)
            self.shm_name = None
            self._shared = False

    def get(self, board: Board):
        h = _board_hash(board)
        idx = (h & (TT_SIZE - 1)) * 4
        stored_hash = int(self._arr[idx + 2])
        if stored_hash != h:
            return None
        packed = int(self._arr[idx + 1])
        verify = int(self._arr[idx])
        if verify != (h ^ packed) & 0xFFFFFFFFFFFFFFFF:
            return None  # concurrent write corrupted entry
        depth, flag, value, mkh = _unpack_tt_data(packed)
        return (depth, flag, value, mkh)

    def put(self, board: Board, depth: int, flag: int, value: float, best_move_key):
        h = _board_hash(board)
        idx = (h & (TT_SIZE - 1)) * 4
        mkh = _move_key_hash(best_move_key)
        packed = _pack_tt_data(depth, flag, value, mkh)
        # Replace-by-depth: only overwrite if new entry is deeper or newer gen
        old_packed = int(self._arr[idx + 1])
        old_depth = (old_packed >> 60) & 0xF
        old_gen = int(self._arr[idx + 3])
        if old_gen == self.generation and old_depth > depth:
            return  # keep deeper entry from same generation
        self._arr[idx + 2] = h
        self._arr[idx + 1] = packed
        self._arr[idx] = (h ^ packed) & 0xFFFFFFFFFFFFFFFF
        self._arr[idx + 3] = self.generation

    def clear(self):
        self._arr[:] = 0
        self.generation += 1

    def new_generation(self):
        self.generation += 1

    def close(self):
        if self._shm is not None:
            self._shm.close()

    def unlink(self):
        if self._shm is not None and self._owns_shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


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
# Make/unmake — zero-allocation in-place board mutation for search
# Eliminates Board() + 4 Worker + 64 Move objects per node.
# ---------------------------------------------------------------------------
def _make_move(board: Board, move: Move) -> tuple:
    """Apply move in-place, swap perspective. Returns undo tuple."""
    pw = board.player_worker
    old_pos = pw.position
    old_pts = pw.points
    old_turns = pw.turns_left
    old_tc = board.turn_count
    old_winner = board.winner
    old_primed = board._primed_mask
    old_carpet = board._carpet_mask
    old_space = board._space_mask

    mt = move.move_type
    if mt == 1:  # PRIME
        bit = 1 << (old_pos[1] * BOARD_SIZE + old_pos[0])
        board._space_mask = old_space & ~bit
        board._primed_mask = old_primed | bit
        dx, dy = _DIR_DXY[move.direction]
        pw.position = (old_pos[0] + dx, old_pos[1] + dy)
        pw.points = old_pts + 1
    elif mt == 0:  # PLAIN
        dx, dy = _DIR_DXY[move.direction]
        pw.position = (old_pos[0] + dx, old_pos[1] + dy)
    elif mt == 2:  # CARPET
        dx, dy = _DIR_DXY[move.direction]
        cx, cy = old_pos
        pm = old_primed
        cm = old_carpet
        for _ in range(move.roll_length):
            cx += dx
            cy += dy
            bit = 1 << (cy * BOARD_SIZE + cx)
            pm &= ~bit
            cm |= bit
        board._primed_mask = pm
        board._carpet_mask = cm
        pw.position = (cx, cy)
        pw.points = old_pts + CARPET_POINTS[min(move.roll_length, 7)]

    board.turn_count = old_tc + 1
    pw.turns_left = old_turns - 1

    # Game-over check (both players out of turns)
    if old_turns - 1 == 0 and board.opponent_worker.turns_left == 0:
        op = board.opponent_worker.points
        mp = pw.points
        if mp > op:
            board.winner = 0
        elif mp < op:
            board.winner = 1
        else:
            board.winner = 2

    board.player_worker, board.opponent_worker = board.opponent_worker, board.player_worker
    return (old_pos, old_pts, old_turns, old_tc, old_winner, old_primed, old_carpet, old_space)


def _unmake_move(board: Board, undo: tuple) -> None:
    """Undo a move applied by _make_move."""
    old_pos, old_pts, old_turns, old_tc, old_winner, old_primed, old_carpet, old_space = undo
    board.player_worker, board.opponent_worker = board.opponent_worker, board.player_worker
    pw = board.player_worker
    pw.position = old_pos
    pw.points = old_pts
    pw.turns_left = old_turns
    board.turn_count = old_tc
    board.winner = old_winner
    board._primed_mask = old_primed
    board._carpet_mask = old_carpet
    board._space_mask = old_space


def _make_null_move(board: Board) -> tuple:
    """Null move: skip turn, swap perspective. For NMP."""
    pw = board.player_worker
    old_turns = pw.turns_left
    old_tc = board.turn_count
    old_winner = board.winner

    board.turn_count = old_tc + 1
    pw.turns_left = old_turns - 1

    if old_turns - 1 == 0 and board.opponent_worker.turns_left == 0:
        op = board.opponent_worker.points
        mp = pw.points
        if mp > op:
            board.winner = 0
        elif mp < op:
            board.winner = 1
        else:
            board.winner = 2

    board.player_worker, board.opponent_worker = board.opponent_worker, board.player_worker
    return (old_turns, old_tc, old_winner)


def _unmake_null_move(board: Board, undo: tuple) -> None:
    """Undo null move."""
    old_turns, old_tc, old_winner = undo
    board.player_worker, board.opponent_worker = board.opponent_worker, board.player_worker
    board.player_worker.turns_left = old_turns
    board.turn_count = old_tc
    board.winner = old_winner


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

        # Tier 1: TT best move (compare via hash since TT stores mkh)
        if tt_best_key is not None and (_move_key_hash(mk) == tt_best_key):
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

        # Tier 5-7: primes and plains scored by chain potential (JIT helpers)
        pm = board._primed_mask
        if m.move_type == MoveType.PRIME:
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = mx + dx, my_ + dy
            best_run_pts = 0
            h_fwd = _count_primed_run_jit(pm, mx, my_, 1, 0)
            h_bwd = _count_primed_run_jit(pm, mx, my_, -1, 0)
            h_total = 1 + h_fwd + h_bwd
            if h_total >= 2:
                p = CARPET_POINTS[min(h_total, 7)]
                if p > best_run_pts:
                    best_run_pts = p
            v_fwd = _count_primed_run_jit(pm, mx, my_, 0, 1)
            v_bwd = _count_primed_run_jit(pm, mx, my_, 0, -1)
            v_total = 1 + v_fwd + v_bwd
            if v_total >= 2:
                p = CARPET_POINTS[min(v_total, 7)]
                if p > best_run_pts:
                    best_run_pts = p
            dest_val = _best_carpet_value_basic_jit(pm, nx, ny, _CARPET_POINTS_NP)
            nearby_primed = 0
            for ddx, ddy in DELTAS_4:
                ax, ay = mx + ddx, my_ + ddy
                if 0 <= ax < 8 and 0 <= ay < 8 and (pm & (1 << (ay * 8 + ax))):
                    nearby_primed += 1
            return 1_000_000.0 + best_run_pts * 200.0 + dest_val * 50.0 + nearby_primed * 30.0 + hist * 0.01

        if m.move_type == MoveType.PLAIN:
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = mx + dx, my_ + dy
            dest_val = _best_carpet_value_basic_jit(pm, nx, ny, _CARPET_POINTS_NP)
            nearby_primed = 0
            for ddx, ddy in DELTAS_4:
                ax, ay = nx + ddx, ny + ddy
                if 0 <= ax < 8 and 0 <= ay < 8 and (pm & (1 << (ay * 8 + ax))):
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
        undo = _make_move(board, m)
        try:
            value = -quiescence(board, -beta, -alpha, ctx, extensions_left - 1)
        except SearchTimeout:
            _unmake_move(board, undo)
            raise
        _unmake_move(board, undo)
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
                ply: int, ctx: SearchContext, allow_null: bool = True) -> float:
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

    # NMP: Null Move Pruning — skip our turn; if score still >= beta,
    # this position is too good and we can prune.
    # Guards: allow_null (no consecutive nulls), depth >= 3, enough turns left.
    if allow_null and depth >= 3 and board.player_worker.turns_left > 3:
        R = 2 if depth <= 5 else 3
        null_undo = _make_null_move(board)
        try:
            null_score = -negamax_pvs(board, depth - 1 - R, -beta, -beta + 1,
                                      ply + 1, ctx, allow_null=False)
        except SearchTimeout:
            _unmake_null_move(board, null_undo)
            raise
        _unmake_null_move(board, null_undo)
        if null_score >= beta:
            return null_score

    killers = ctx.killers.get(ply)
    moves = order_moves(board, tt_best_key, killers, ctx.history)
    if not moves:
        return evaluate(board)

    best_value = NEG_INF
    best_mkey = None
    moves_searched = 0

    for m in moves:
        undo = _make_move(board, m)

        # LMR: Late Move Reductions — reduce depth on late quiet moves.
        # First 4 moves searched at full depth; later non-carpet moves reduced.
        reduction = 0
        if moves_searched >= 4 and depth >= 3 and m.move_type != MoveType.CARPET:
            reduction = 1
            if moves_searched >= 8:
                reduction = 2

        try:
            if moves_searched == 0:
                # PV node: full window search
                value = -negamax_pvs(board, depth - 1, -beta, -alpha, ply + 1, ctx)
            else:
                # Scout search with LMR
                value = -negamax_pvs(board, depth - 1 - reduction, -alpha - 1,
                                     -alpha, ply + 1, ctx)
                if value > alpha:
                    # LMR re-search at full depth (null window)
                    if reduction > 0:
                        value = -negamax_pvs(board, depth - 1, -alpha - 1,
                                             -alpha, ply + 1, ctx)
                    # PVS re-search with full window
                    if value > alpha and value < beta:
                        value = -negamax_pvs(board, depth - 1, -beta, -alpha,
                                             ply + 1, ctx)
        except SearchTimeout:
            _unmake_move(board, undo)
            raise
        _unmake_move(board, undo)

        moves_searched += 1

        # Length-1 carpet opportunity-cost penalty
        if m.move_type == MoveType.CARPET and m.roll_length == 1:
            value -= 2.0

        if value > best_value:
            best_value = value
            best_mkey = move_key(m)

        if value > alpha:
            alpha = value

        if alpha >= beta:
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
        # CS-0 safety: search is always legal; plain(RIGHT) can be invalid → forfeit
        wx, wy = board.player_worker.get_location()
        return Move.search((wx, wy)), 0, 0, 0.0

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
                    undo = _make_move(board, m)

                    try:
                        if first:
                            score = -negamax_pvs(board, depth - 1, -cur_beta,
                                                 -cur_alpha, 1, ctx)
                            first = False
                        else:
                            score = -negamax_pvs(board, depth - 1, -cur_alpha - 1,
                                                 -cur_alpha, 1, ctx)
                            if cur_alpha < score < cur_beta:
                                score = -negamax_pvs(board, depth - 1, -cur_beta,
                                                     -cur_alpha, 1, ctx)
                    except SearchTimeout:
                        _unmake_move(board, undo)
                        raise

                    _unmake_move(board, undo)

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
    pm = board._primed_mask
    if m.direction is not None:
        dx, dy = DIR_DELTAS.get(m.direction, (0, 0))
    else:
        dx, dy = 0, 0

    if m.move_type == MoveType.CARPET:
        pts = CARPET_POINTS[min(m.roll_length, 7)]
        lx = mx + dx * m.roll_length
        ly = my_ + dy * m.roll_length
        future = _best_carpet_value_basic_jit(pm, lx, ly, _CARPET_POINTS_NP) * 0.3
        return pts * 10 + future + 100

    if m.move_type == MoveType.PRIME:
        nx, ny = mx + dx, my_ + dy
        behind = _count_primed_run_jit(pm, mx, my_, -dx, -dy)
        total_line = 1 + behind
        line_pts = CARPET_POINTS[min(total_line, 7)] if total_line >= 2 else 0
        dest_val = _best_carpet_value_basic_jit(pm, nx, ny, _CARPET_POINTS_NP)
        adj_primed = 0
        for ddx, ddy in DELTAS_4:
            ax, ay = mx + ddx, my_ + ddy
            if 0 <= ax < 8 and 0 <= ay < 8 and (pm & (1 << (ay * 8 + ax))):
                adj_primed += 1
            ax, ay = nx + ddx, ny + ddy
            if 0 <= ax < 8 and 0 <= ay < 8 and (pm & (1 << (ay * 8 + ax))):
                adj_primed += 1
        return 1 + line_pts * 2 + dest_val * 1.5 + adj_primed * 0.8 + 10

    # PLAIN
    nx, ny = mx + dx, my_ + dy
    dest_val = _best_carpet_value_basic_jit(pm, nx, ny, _CARPET_POINTS_NP)
    adj = 0
    sm = board._space_mask
    for ddx, ddy in DELTAS_4:
        ax, ay = nx + ddx, ny + ddy
        if 0 <= ax < 8 and 0 <= ay < 8:
            bit = 1 << (ay * 8 + ax)
            if sm & bit:
                adj += 1
            if pm & bit:
                adj += 2
    return dest_val * 1.0 + adj * 0.3


def greedy_move(board: Board) -> Move:
    moves = board.get_valid_moves(exclude_search=True)
    if not moves:
        # CS-0 safety: search is always legal; plain(RIGHT) can be invalid → forfeit
        wx, wy = board.player_worker.get_location()
        return Move.search((wx, wy))
    best = moves[0]
    best_s = NEG_INF
    for m in moves:
        s = score_move(board, m)
        if s > best_s:
            best_s = s
            best = m
    return best


# ---------------------------------------------------------------------------
# Board serialization for SMP worker processes
# ---------------------------------------------------------------------------
def _serialize_board(board: Board) -> tuple:
    """Serialize board to 14 primitives for IPC."""
    px, py = board.player_worker.position
    ox, oy = board.opponent_worker.position
    return (board._primed_mask, board._carpet_mask, board._space_mask, board._blocked_mask,
            px, py, ox, oy,
            board.player_worker.points, board.opponent_worker.points,
            board.player_worker.turns_left, board.opponent_worker.turns_left,
            board.turn_count, board.winner)


def _deserialize_board(state: tuple) -> Board:
    """Reconstruct Board from serialized state — bypass __init__."""
    (pm, cm, sm, bm, px, py, ox, oy, pp, op_, pt, ot, tc, winner) = state
    from game.worker import Worker
    board = Board.__new__(Board)
    board._primed_mask = pm
    board._carpet_mask = cm
    board._space_mask = sm
    board._blocked_mask = bm
    board.turn_count = tc
    board.winner = winner
    board.history = None
    board.build_history = False
    board.opponent_search = (None, False)
    board.player_search = (None, False)
    board.is_player_a_turn = True
    board.time_to_play = 20
    board.MAX_TURNS = 80
    # Reconstruct workers
    pw = Worker.__new__(Worker)
    pw.position = (px, py)
    pw.is_player_a = True
    pw.is_player_b = False
    pw.points = pp
    pw.turns_left = pt
    pw.time_left = 240.0
    board.player_worker = pw
    ow = Worker.__new__(Worker)
    ow.position = (ox, oy)
    ow.is_player_a = False
    ow.is_player_b = True
    ow.points = op_
    ow.turns_left = ot
    ow.time_left = 240.0
    board.opponent_worker = ow
    # Precompute valid_search_moves
    board.valid_search_moves = [Move.search((x, y)) for y in range(8) for x in range(8)]
    return board


# ---------------------------------------------------------------------------
# Numba warmup — JIT compile all functions once (1-3s)
# ---------------------------------------------------------------------------
def _warmup_numba():
    """Call each JIT function once to trigger compilation. No-op if numba unavailable."""
    if not _HAS_NUMBA:
        return
    pm = np.uint64(0).item()  # int
    cm = np.uint64(0).item()
    sm = np.uint64(0xFFFFFFFFFFFFFFFF).item()
    bm = np.uint64(0).item()
    bp = np.zeros(64, dtype=np.float64)
    cp = _CARPET_POINTS_NP
    w = _WEIGHTS_NP
    # Trigger compilation of all JIT functions
    _count_primed_run_jit(pm, 0, 0, 1, 0)
    _best_carpet_value_jit(pm, 0, 0, 7, 7, cp)
    _best_carpet_value_basic_jit(pm, 0, 0, cp)
    _count_chain_potential_jit(pm, bm, 0, 0, 7, 7, cp)
    _primeable_bonus_jit(pm, cm, bm, 0, 0, cp)
    _count_territory_jit(bm, cm, 0, 0, 2)
    _count_mobility_jit(bm, pm, 0, 0, 7, 7)
    _primed_infrastructure_jit(pm, 0, 0, 7, 7, cp)
    _evaluate_jit(pm, cm, sm, bm, 0, 0, 7, 7, 0, 0, 40, 40, bp, cp, w)


# ---------------------------------------------------------------------------
# SMP worker process
# ---------------------------------------------------------------------------
class _SMPCommand:
    __slots__ = ('board_state', 'time_budget', 'max_depth', 'generation')
    def __init__(self, board_state, time_budget, max_depth, generation):
        self.board_state = board_state
        self.time_budget = time_budget
        self.max_depth = max_depth
        self.generation = generation


def _smp_worker(pipe, shm_name, worker_id):
    """SMP worker process: runs search in parallel, writes to shared TT."""
    try:
        tt = TranspositionTable(shm_name=shm_name, create=False)
        history = HistoryTable()

        # Warmup numba in worker process too
        _warmup_numba()

        while True:
            try:
                cmd = pipe.recv()
            except (EOFError, BrokenPipeError):
                break
            if cmd is None:
                break

            board = _deserialize_board(cmd.board_state)
            tt.generation = cmd.generation

            # Workers use depth offset for search diversity
            depth_offset = worker_id + 1  # worker 0 starts at depth 2, worker 1 at depth 3

            # Run iterative deepening with 90% of time budget (safety margin)
            budget = cmd.time_budget * 0.90
            try:
                best_move, depth_reached, nodes, elapsed = iterative_deepening_search(
                    board, tt, history, budget, max_depth=cmd.max_depth
                )
                mk = move_key(best_move) if best_move else None
                pipe.send((mk, depth_reached, nodes, elapsed))
            except Exception:
                pipe.send((None, 0, 0, 0.0))

            history.decay(2)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


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

        # Warmup numba JIT (1-3s one-time cost)
        _warmup_numba()

        # Shared-memory TT for Lazy SMP
        self.tt = TranspositionTable(create=True)
        self.history = HistoryTable()

        # SMP: fork 2 worker processes
        self._smp_ok = False
        self._workers = []
        self._pipes = []
        self._num_workers = 2
        try:
            if self.tt._shared:
                for wid in range(self._num_workers):
                    parent_conn, child_conn = multiprocessing.Pipe()
                    p = multiprocessing.Process(
                        target=_smp_worker,
                        args=(child_conn, self.tt.shm_name, wid),
                        daemon=True
                    )
                    p.start()
                    child_conn.close()  # parent doesn't use child end
                    self._workers.append(p)
                    self._pipes.append(parent_conn)
                self._smp_ok = True
        except Exception:
            # Fallback: single-threaded
            self._smp_ok = False
            for pipe in self._pipes:
                try:
                    pipe.close()
                except Exception:
                    pass
            self._pipes = []
            self._workers = []

        # Stats
        self.last_depth = 0
        self.last_nodes = 0
        self.last_time = 0.0
        self.total_time_used = 0.0
        self.turn_number = 0
        self._smp_worker_nodes = 0

    def __del__(self):
        """Cleanup SMP workers and shared memory."""
        for pipe in self._pipes:
            try:
                pipe.send(None)
            except Exception:
                pass
        for p in self._workers:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
        for pipe in self._pipes:
            try:
                pipe.close()
            except Exception:
                pass
        try:
            self.tt.close()
            self.tt.unlink()
        except Exception:
            pass

    def commentate(self):
        smp = f"SMP={1 + len(self._workers)}" if self._smp_ok else "ST"
        jit = "JIT" if _HAS_NUMBA else "Py"
        return (f"Challenger v7.3 {jit} {smp} | d={self.last_depth} n={self.last_nodes} "
                f"wn={self._smp_worker_nodes} "
                f"t={self.last_time:.2f}s total={self.total_time_used:.1f}s")

    # ---------- Time management ----------
    def _compute_time_budget(self, board: Board, time_remaining: float) -> float:
        turns_left = max(board.player_worker.turns_left, 1)

        # v7: Consistent ~4s/turn (Carrie: 4.01s fixed, Michael: 3.25s avg).
        # v6 used only 113s of 240s. Target: 150-170s total usage.
        if turns_left > 25:
            budget = 4.5   # Early: invest in deep TT entries
        elif turns_left > 10:
            budget = 4.0   # Mid: maintain good search depth
        elif turns_left > 5:
            budget = 3.5   # Late: slightly less
        else:
            # Endgame: distribute remaining time
            budget = time_remaining / turns_left * 0.85

        budget = max(0.5, min(8.0, budget))

        # Safety floors
        if time_remaining < 30:
            budget = min(budget, time_remaining / turns_left * 0.8)
        if time_remaining < 15:
            budget = min(budget, 0.8)
        if time_remaining < 5:
            budget = min(budget, 0.3)

        return budget

    # ---------- Rat search decision ----------
    def _rat_thresholds(self, board: Board) -> Tuple[float, float]:
        my_pts = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        diff = my_pts - opp_pts
        turns_left = min(board.player_worker.turns_left, board.opponent_worker.turns_left)

        # v6: Raised thresholds to target ~6 searches/game (was 13.3).
        # Deep analysis showed r=-0.471 between search% and score.
        if diff <= -10:
            p, e = 0.40, 0.5
        elif diff <= -5:
            p, e = 0.42, 0.8
        elif diff < 0:
            p, e = 0.45, 1.0
        elif diff < 5:
            p, e = 0.50, 1.2
        else:
            p, e = 0.55, 1.5

        # v6: No endgame lowering — v5 lowered by 0.1 in last 5 turns, added junk searches
        return p, e

    def _should_search_rat(self, board: Board, rat_ev: float, best_carpet_pts: int) -> bool:
        if self.rat_tracker is None:
            return False
        if self.search_cooldown > 0:
            return False

        turns_left = min(board.player_worker.turns_left, board.opponent_worker.turns_left)
        min_obs = 3 if turns_left <= 5 else 5  # v6: was 2/3
        if self.observations_since_reset < min_obs:
            return False

        _, prob = self.rat_tracker.best_guess()
        p_thr, e_thr = self._rat_thresholds(board)
        if prob < p_thr:
            return False
        if rat_ev < e_thr:
            return False

        # Compare opportunity cost against a big carpet
        if best_carpet_pts >= 4 and rat_ev < best_carpet_pts:  # v6: lowered from 6/+1 margin
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
            self.search_cooldown = 3  # v6: was 2

        if board.player_search[0] is not None and not board.player_search[1]:
            self.search_cooldown = 3  # v6: was 2
            # v7: Certainty update — rat is NOT at the missed cell
            if not rat_reset and self.rat_tracker:
                sx, sy = board.player_search[0]
                self.rat_tracker.belief[sy * BOARD_SIZE + sx] = 0.0
                bsum = self.rat_tracker.belief.sum()
                if bsum > 0:
                    self.rat_tracker.belief /= bsum

        # v7: Also incorporate opponent miss info
        if (not rat_reset and self.rat_tracker and
                board.opponent_search[0] is not None and
                not board.opponent_search[1]):
            sx, sy = board.opponent_search[0]
            self.rat_tracker.belief[sy * BOARD_SIZE + sx] = 0.0
            bsum = self.rat_tracker.belief.sum()
            if bsum > 0:
                self.rat_tracker.belief /= bsum

        if self.search_cooldown > 0:
            self.search_cooldown -= 1

        self.observations_since_reset += 1

        # Update rat belief
        rat_ev = -2.0
        if self.rat_tracker:
            self.rat_tracker.update(board, int(noise), reported_dist, wx, wy)
            rat_ev = self.rat_tracker.search_ev()
            # CS-1c: refresh belief-proximity table used by evaluate()
            set_belief_proximity(self.rat_tracker.belief)
        else:
            set_belief_proximity(None)

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

        # New generation for TT aging
        self.tt.new_generation()

        # Dispatch to SMP workers (they search in parallel, writing to shared TT)
        if self._smp_ok:
            board_state = _serialize_board(board)
            cmd = _SMPCommand(board_state, time_budget, max_depth, self.tt.generation)
            for pipe in self._pipes:
                try:
                    pipe.send(cmd)
                except Exception:
                    pass

        # Main process search (uses same shared TT — benefits from worker writes)
        best_move, depth_reached, nodes, _ = iterative_deepening_search(
            board, self.tt, self.history, time_budget, max_depth=max_depth
        )

        # Collect worker results (50ms timeout — don't block)
        worker_nodes = 0
        if self._smp_ok:
            for pipe in self._pipes:
                try:
                    if pipe.poll(0.05):
                        result = pipe.recv()
                        if result and result[2]:
                            worker_nodes += result[2]
                except Exception:
                    pass
        self._smp_worker_nodes = worker_nodes

        if best_move is None:
            best_move = greedy_move(board)

        self.last_depth = depth_reached
        self.last_nodes = nodes
        self.last_time = time.time() - play_start
        self.total_time_used += self.last_time

        return best_move
