from collections.abc import Callable
from typing import List, Set, Tuple
import random
import jax.numpy as jnp
import numpy as np
import jax
from engine.game.enums import Cell, Noise, BOARD_SIZE

from engine.game import board, move, enums

EMISSION_PROBS = jnp.array([
    [0.7,  0.15, 0.15],  # SPACE
    [0.1,  0.8,  0.1 ],  # PRIMED
    [0.1,  0.1,  0.8 ],  # CARPET
    [0.5,  0.3,  0.2 ],  # BLOCKED
])

CELL_TO_IDX = {
    Cell.SPACE:   0,
    Cell.PRIMED:  1,
    Cell.CARPET:  2,
    Cell.BLOCKED: 3,
}

# Distance error offsets and their probabilities
DIST_OFFSETS = jnp.array([-1, 0, 1, 2])
DIST_PROBS   = jnp.array([0.12, 0.70, 0.12, 0.06])

# All 64 (x, y) positions as arrays
ALL_X = jnp.array([i % BOARD_SIZE for i in range(BOARD_SIZE * BOARD_SIZE)])
ALL_Y = jnp.array([i // BOARD_SIZE for i in range(BOARD_SIZE * BOARD_SIZE)])



class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        # we start by calculating a prior of where the rat is
        self.belief = None
        self.T = None
        if transition_matrix is not None:
            self.T = jnp.array(transition_matrix)
            prior = jnp.zeros(64).at[0].set(1.0)

            @jax.jit
            def compute_prior(T):
                def step(p, _):
                    return p @ T, None
                final, _ = jax.lax.scan(step, prior, xs=None, length=1000)
                return final

            self.belief = compute_prior(self.T)

        self.floor_types = jnp.zeros(64, dtype=int)

    def _get_floor_type_array(self, board) -> jnp.ndarray:
        """Return a (64,) int array of floor type indices from the current board state."""
        # Read directly from bitboards — much faster than 64 get_cell() calls
        floor = np.zeros(64, dtype=jnp.int32)  # default: SPACE=0
        for i in range(64):
            bit = 1 << i
            if board._primed_mask & bit:
                floor[i] = 1  # PRIMED
            elif board._carpet_mask & bit:
                floor[i] = 2  # CARPET
            elif board._blocked_mask & bit:
                floor[i] = 3  # BLOCKED
        return jnp.array(floor)
    

    @staticmethod
    @jax.jit
    def _bayesian_update(
        belief: jnp.ndarray,       # (64,)
        T: jnp.ndarray,            # (64, 64)
        floor_types: jnp.ndarray,  # (64,) int — floor type index per cell
        noise_idx: int,            # scalar: 0=squeak, 1=scratch, 2=squeal
        reported_dist: int,        # scalar: reported manhattan distance
        wx: int,                   # worker x
        wy: int,                   # worker y
    ) -> jnp.ndarray:

        # 1. Rat movement
        predicted = belief @ T 

        # 2. P(noise | floor type at each cell)
        noise_lik = EMISSION_PROBS[floor_types, noise_idx]

        # 3. P(reported dist | actual dist to each cell)
        actual_dists = jnp.abs(ALL_X - wx) + jnp.abs(ALL_Y - wy)

        # For each cell, sum P(offset) where actual + offset == reported,
        implied_reported = actual_dists[:, None] + DIST_OFFSETS[None, :]

        # exact match
        exact_match = (implied_reported == reported_dist)
        dist_lik = jnp.sum(exact_match * DIST_PROBS[None, :], axis=1)

        # reported_dist == 0 means any roll that would go <= 0
        clamped_match = (implied_reported <= 0) 
        clamped_lik = jnp.sum(clamped_match * DIST_PROBS[None, :], axis=1)

        dist_lik = jnp.where(reported_dist == 0, clamped_lik, dist_lik)

        # 4. posterior = predicted * noise_lik * dist_lik
        posterior = predicted * noise_lik * dist_lik
        posterior = posterior / jnp.sum(posterior)  # normalize

        return posterior
    
    def update_belief(self, board, sensor_data: Tuple):
        noise, reported_dist = sensor_data  # Noise enum, int
        noise_idx = int(noise)  # Noise(0)=squeak, (1)=scratch, (2)=squeal

        wx, wy = board.player_worker.get_location()
        floor_types = self._get_floor_type_array(board)

        self.belief = self._bayesian_update(
            self.belief,
            self.T,
            floor_types,
            noise_idx,
            reported_dist,
            wx,
            wy,
        )

    def search_ev(self) -> float:
        p = float(jnp.max(self.belief))
        return p * 4 - (1 - p) * 2
        
    def rat_guess(self):
        idx = int(jnp.argmax(self.belief))
        x, y = idx % BOARD_SIZE, idx // BOARD_SIZE
        return move.Move.search((x, y))
    
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        self.update_belief(board, sensor_data)

        if self.search_ev() > 0:
            return self.rat_guess()
        

        moves = board.get_valid_moves()
        return random.choice(moves)
