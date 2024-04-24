import random

class DGBotRandom:
    """Represents a player who makes random moves."""

    def __init__(self):
        """Initializes a random player."""
        self._num_wins = 0
        self._num_losses = 0
        self._num_ties = 0

    def make_decision(self, board):
        """
        Makes a random valid move based on the current board state.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            tuple: The row and column of the selected move, or None if no valid moves are available.
        """
        valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
        if valid_moves:
            return random.choice(valid_moves)
        else:
            print("Random: No valid moves available.")
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
