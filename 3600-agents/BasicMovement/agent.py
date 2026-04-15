from collections.abc import Callable
from typing import List, Set, Tuple
import random

from engine.game import board, move, enums
from engine.game.enums import Direction, BOARD_SIZE, CARPET_POINTS_TABLE


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """
        pass
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return ""
    
    def expectminimax_root(self, board, time_left):
        moves = board.get_valid_moves()
        return max(moves, key=lambda m: self.expect(
            board.forecast_move(m), self.max_depth - 1
        ))
    
    def expect(self, board, depth):
        return self.min_node(board, depth)
    
    def min_node(self, board, depth):
        if depth == 0 or board.is_game_over():
            return self._heuristic(board)
        board.reverse_perspective()
        moves = board.get_valid_moves()
        best = min(self.max_node(board.forecast_move(m), depth - 1) for m in moves)
        board.reverse_perspective()
        return best
    
    def max_node(self, board, depth):
        if depth == 0 or board.is_game_over():
            return self._heuristic(board)
        moves = board.get_valid_moves()
        return max(self.expect(board.forecast_move(m), depth - 1) for m in moves)
    
    
    
    def carpet_potential(self, board, x: int, y: int) -> float:
        total = 0.0
        for dx, dy, direction in [
            (1, 0, Direction.RIGHT),
            (-1, 0, Direction.LEFT),
            (0, 1, Direction.DOWN),
            (0, -1, Direction.UP),
        ]:
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
    
    def prime_potential(self, board, x: int, y: int) -> float:
        score = 0.0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                bit = 1 << (ny * BOARD_SIZE + nx)
                if board._primed_mask & bit:
                    score += 1.0
        return score
    
    def _heuristic(self, board):
        # THIS is where your carpet scoring logic lives
        # now it's evaluating a future board state, not just one move
        carpet_score = self.carpet_potential(board, *board.player_worker.get_location())
        rat_ev = self.search_ev()
        return board.player_worker.get_points() - board.opponent_worker.get_points() \
             + 0.3 * carpet_score + rat_ev


    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):

        moves = board.get_valid_moves()
        return random.choice(moves)
