"""
Module for Adam's Minimax player for Drei Gewinnt (Tic Tac Toe).
"""

# External Modules
import numpy as np

class DGBotMinimax:
    """Represents a player who uses the minimax algorithm."""

    def __init__(self):
        """Initializes a minimax player."""
        self._num_wins = 0
        self._num_losses = 0
        self._num_ties = 0

    def make_decision(self, board):
        """
        Makes a move based on the minimax algorithm.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            tuple: The row and column of the selected move.
        """
        best_score = -np.inf
        best_move = None

        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    board[i][j] = 1
                    score = self._minimax(board, 0, False)
                    board[i][j] = 0

                    if score > best_score:
                        best_score = score
                        best_move = (i, j)

        return best_move

    def _minimax(self, board, depth, is_maximizing):
        """
        Recursive function for the minimax algorithm.

        Args:
            board (numpy.ndarray): The current state of the board.
            depth (int): The current depth of the recursion.
            is_maximizing (bool): True if it's the maximizing player's turn, False otherwise.

        Returns:
            int: The score of the current board state.
        """
        result = self._is_game_over(board)
        if result is not None:
            return result

        if is_maximizing:
            best_score = -np.inf
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        board[i][j] = 1
                        score = self._minimax(board, depth + 1, False)
                        board[i][j] = 0
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = np.inf
            for i in range(3):
                for j in range(3):
                    if board[i][j] == 0:
                        board[i][j] = -1
                        score = self._minimax(board, depth + 1, True)
                        board[i][j] = 0
                        best_score = min(score, best_score)
            return best_score

    def _is_game_over(self, board):
        """
        Checks if the game has ended.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            int: 1 if player 1 wins, -1 if player 2 wins, 0 if it's a tie, None if the game is not over.
        """
        # Check rows
        for row in board:
            if abs(sum(row)) == 3:
                return sum(row) // 3

        # Check columns
        for col in board.T:
            if abs(sum(col)) == 3:
                return sum(col) // 3

        # Check diagonals
        if abs(sum(np.diag(board))) == 3:
            return sum(np.diag(board)) // 3
        if abs(sum(np.diag(np.fliplr(board)))) == 3:
            return sum(np.diag(np.fliplr(board))) // 3

        # Check for a tie
        if np.count_nonzero(board) == 9:
            return 0

        return None

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
