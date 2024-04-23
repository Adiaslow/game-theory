"""
Module for Adam's Markov Chain Monte Carlo (MCMC) player for Drei Gewinnt (Tic Tac Toe).
"""

# External Modules
import numpy as np
import random

class DGAdamMCMC:
    """Represents a Drei Gewinnt player that uses Adam's MCMC algorithm."""

    def __init__(self, temperature=0.5):
        """
        Initializes the Drei Gewinnt player.

        Args:
            temperature (float): The temperature parameter for the player.
        """
        self._temperature = temperature
        self._board = np.zeros((3, 3), dtype=int)
        self._moves = []
        self._num_wins = 0
        self._num_losses = 0
        self._num_ties = 0

    def _metropolis_hastings(self, board, move):
        """
        Applies the Metropolis-Hastings algorithm to determine if a move should be accepted.

        Args:
            board (numpy.ndarray): The current state of the board.
            move (tuple): The row and column of the proposed move.

        Returns:
            bool: True if the move should be accepted, False otherwise.
        """
        # Calculate the probability of accepting the move based on the current board state
        current_score = self._evaluate_board(board)
        new_board = board.copy()
        new_board[move[0]][move[1]] = 1
        new_score = self._evaluate_board(new_board)
        acceptance_probability = min(1, np.exp((new_score - current_score) / self._temperature))
        return random.random() < acceptance_probability

    def _evaluate_board(self, board):
        """
        Evaluates the score of the current board state.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            float: The score of the board state.
        """
        score = 0

        # Check rows
        for row in board:
            if np.sum(row) == 2:
                score += 1
            elif np.sum(row) == -2:
                score -= 1

        # Check columns
        for col in board.T:
            if np.sum(col) == 2:
                score += 1
            elif np.sum(col) == -2:
                score -= 1

        # Check diagonals
        if np.trace(board) == 2:
            score += 1
        elif np.trace(board) == -2:
            score -= 1

        if np.trace(np.fliplr(board)) == 2:
            score += 1
        elif np.trace(np.fliplr(board)) == -2:
            score -= 1

        return score

    def make_decision(self, board, num_iterations=100):
        """
        Makes a decision based on the current board state using MCMC.

        Args:
            board (numpy.ndarray): The current state of the board.
            num_iterations (int): The number of MCMC iterations to perform.

        Returns:
            tuple: The row and column of the selected move, or None if no valid moves are available.
        """
        valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
        if not valid_moves:
            return None

        best_move = None
        for _ in range(num_iterations):
            move = random.choice(valid_moves)
            if self._metropolis_hastings(board, move):
                best_move = move

        return best_move

    def report(self, result):
        """
        Updates the player's statistics based on the result of the game.

        Args:
            result (int): 1 if the player won, -1 if the player lost, 0 if it was a tie.
        """
        if result == 1:
            self._num_wins += 1
        elif result == -1:
            self._num_losses += 1
        else:
            self._num_ties += 1

    def get_stats(self):
        """Returns the player's current statistics."""
        return {
            'wins': self._num_wins,
            'losses': self._num_losses,
            'ties': self._num_ties
        }
