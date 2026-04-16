from collections.abc import Callable
from typing import Tuple
import random
import jax.numpy as jnp
import numpy as np
import jax
from game import board as board_module, move
from game.enums import Cell, Noise, BOARD_SIZE, Direction, CARPET_POINTS_TABLE, MoveType

EMISSION_PROBS = jnp.array([
    [0.7,  0.15, 0.15],  # space
    [0.1,  0.8,  0.1 ],  # primed
    [0.1,  0.1,  0.8 ],  # carpet
    [0.5,  0.3,  0.2 ],  # blocked
])

DIST_OFFSETS = jnp.array([-1, 0, 1, 2])
DIST_PROBS   = jnp.array([0.12, 0.70, 0.12, 0.06])
ALL_X = jnp.array([i % BOARD_SIZE for i in range(BOARD_SIZE * BOARD_SIZE)])
ALL_Y = jnp.array([i // BOARD_SIZE for i in range(BOARD_SIZE * BOARD_SIZE)])

@jax.jit
def bayesian_update(belief, T, floor_types, noise_idx, reported_dist, wx, wy):
    predicted = belief @ T
    noise_lik = EMISSION_PROBS[floor_types, noise_idx]
    actual_dists = jnp.abs(ALL_X - wx) + jnp.abs(ALL_Y - wy)
    implied_reported = actual_dists[:, None] + DIST_OFFSETS[None, :]
    exact_match = (implied_reported == reported_dist)
    dist_lik = jnp.sum(exact_match * DIST_PROBS[None, :], axis=1)
    clamped_match = (implied_reported <= 0)
    clamped_lik = jnp.sum(clamped_match * DIST_PROBS[None, :], axis=1)
    dist_lik = jnp.where(reported_dist == 0, clamped_lik, dist_lik)
    posterior = predicted * noise_lik * dist_lik
    return posterior / jnp.sum(posterior)

def dir_helper(direction) -> Tuple[int, int]:
    return {
        Direction.RIGHT: (1, 0),
        Direction.LEFT:  (-1, 0),
        Direction.DOWN:  (0, 1),
        Direction.UP:    (0, -1),
    }[direction]

def get_floor_type_array(board) -> jnp.ndarray:
    floor = np.zeros(64, dtype=np.int32)
    for i in range(64):
        bit = 1 << i
        if board._primed_mask & bit:
            floor[i] = 1
        elif board._carpet_mask & bit:
            floor[i] = 2
        elif board._blocked_mask & bit:
            floor[i] = 3
    return jnp.array(floor)

def carpet_potential(board, x: int, y: int) -> float:
    total = 0.0
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        run = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
            bit = 1 << (cy * BOARD_SIZE + cx)
            if board._primed_mask & bit:
                run += 1
                cx += dx
                cy += dy
            else:
                break
        if run >= 2:
            total += CARPET_POINTS_TABLE[min(run, BOARD_SIZE - 1)]
    return total

def prime_potential(board, x: int, y: int) -> float:
    score = 0.0
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board._primed_mask & bit:
                score += 1.0
    return score



def score_move(board, m) -> float:
    wx, wy = board.player_worker.get_location()

    if m.move_type == MoveType.CARPET:
        immediate = CARPET_POINTS_TABLE[m.roll_length]
        dx, dy = dir_helper(m.direction)
        lx, ly = wx + dx * m.roll_length, wy + dy * m.roll_length
        return immediate + 0.3 * carpet_potential(board, lx, ly)

    if m.move_type == MoveType.PRIME:
        dx, dy = dir_helper(m.direction)
        dest_x, dest_y = wx + dx, wy + dy
        extension_bonus = 0.5 * prime_potential(board, wx, wy)
        positional = 0.3 * carpet_potential(board, dest_x, dest_y)
        return 1.0 + extension_bonus + positional

    if m.move_type == MoveType.PLAIN:
        dx, dy = dir_helper(m.direction)
        dest_x, dest_y = wx + dx, wy + dy
        return 0.3 * carpet_potential(board, dest_x, dest_y) \
             + 0.2 * prime_potential(board, dest_x, dest_y)

    return 0.0

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.T = None
        self.belief = None
        self.prior = None
        self.max_depth = 2

        if transition_matrix is not None:
            self.T = jnp.array(transition_matrix)
            start = jnp.zeros(64).at[0].set(1.0)

            @jax.jit
            def compute_prior(T):
                def step(p, _):
                    return p @ T, None
                final, _ = jax.lax.scan(step, start, xs=None, length=1000)
                return final

            self.prior = compute_prior(self.T)
            self.belief = self.prior

    def update_belief(self, board, sensor_data: Tuple):
        noise, dist = sensor_data
        noise_idx = int(noise)
        wx, wy = board.player_worker.get_location()
        floor_types = get_floor_type_array(board)
        self.belief = bayesian_update(self.belief, self.T, floor_types, noise_idx, dist, wx, wy)

    def maybe_reset_belief(self, board):
        for loc, result in [board.opponent_search, board.player_search]:
            if loc is not None and result:
                self.belief = self.prior
                break

    def search_ev(self):
        p = float(jnp.max(self.belief))
        return p * 4 - (1 - p) * 2
    
    def rat_guess(self):
        idx = int(jnp.argmax(self.belief))
        return move.Move.search((idx % BOARD_SIZE, idx // BOARD_SIZE))
    
    def expectiminimax_root(self, board, time_left):
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return None
        return max(moves, key=lambda m: self.chance_node(
            board.forecast_move(m), self.max_depth - 1
        ))

    def chance_node(self, board, depth):
        return self.min_node(board, depth)

    def min_node(self, board, depth):
        if depth == 0 or board is None or board.is_game_over():
            return self.heuristic(board)
        board.reverse_perspective()
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            board.reverse_perspective()
            return self.heuristic(board)
        best = min(
            self.max_node(board.forecast_move(m), depth - 1)
            for m in moves
        )
        board.reverse_perspective()
        return best

    def max_node(self, board, depth):
        if depth == 0 or board is None or board.is_game_over():
            return self.heuristic(board)
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            return self.heuristic(board)
        return max(
            self.chance_node(board.forecast_move(m), depth - 1)
            for m in moves
        )

    def heuristic(self, board) -> float:
        if board is None:
            return 0.0
        x, y = board.player_worker.get_location()
        score_diff = board.player_worker.get_points() \
                   - board.opponent_worker.get_points()
        carpet_score = carpet_potential(board, x, y)
        rat_ev = self.search_ev()
        return score_diff + 0.3 * carpet_score + rat_ev
    
    def commentate(self):
        if self.belief is not None:
            idx = int(jnp.argmax(self.belief))
            x, y = idx % BOARD_SIZE, idx // BOARD_SIZE
            return f"Rat most likely at ({x},{y}), p={float(self.belief[idx]):.4f}"
        return ""

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.maybe_reset_belief(board)
        self.update_belief(board, sensor_data)

        if self.search_ev() > 0:
            return self.rat_guess()

        best = self.expectiminimax_root(board, time_left)
        if best is not None:
            return best

        # fallback to random move if something breaks
        print("Falling back to random move")
        return random.choice(board.get_valid_moves())