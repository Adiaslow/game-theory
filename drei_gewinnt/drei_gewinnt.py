"""
Module for the Drei Gewinnt (Tic Tac Toe) game.
"""
# Internal Modules
from dg_bot_random import DGRandomBot

# External Modules
import numpy as np
import matplotlib.pyplot as plt
import random

class DreiGewinnt:
    """Represents a Drei Gewinnt (Tic Tac Toe) game."""

    def __init__(self, player1, player2):
        """
        Initializes a Drei Gewinnt game with two players.

        Args:
            player1 (object): The first player object.
            player2 (object): The second player object.
        """
        self._board = np.zeros((3, 3), dtype=int)
        self._player1 = player1
        self._player2 = player2
        self._player1_score = []
        self._player2_score = []

    def get_board(self):
        """Returns the current state of the board."""
        return self._board

    def make_move(self, row, col, player):
        """
        Makes a move on the board.

        Args:
            row (int): The row of the move (0, 1, or 2).
            col (int): The column of the move (0, 1, or 2).
            player (int): The player making the move (1 or -1).

        Returns:
            bool: True if the move is valid and the game has not ended, False otherwise.
        """
        if not self._is_valid_move(row, col):
            return False

        self._board[row][col] = player

        return not self._is_game_over()

    def _is_valid_move(self, row, col):
        """
        Checks if a move is valid.

        Args:
            row (int): The row of the move (0, 1, or 2).
            col (int): The column of the move (0, 1, or 2).

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False

        if self._board[row][col] != 0:
            return False

        return True

    def _is_game_over(self):
        """
        Checks if the game has ended.

        Returns:
            int: 1 if player 1 wins, -1 if player 2 wins, 0 if it's a tie, None if the game is not over.
        """
        # Check rows
        for row in self._board:
            if abs(sum(row)) == 3:
                return sum(row) // 3

        # Check columns
        for col in self._board.T:
            if abs(sum(col)) == 3:
                return sum(col) // 3

        # Check diagonals
        if abs(sum(np.diag(self._board))) == 3:
            return sum(np.diag(self._board)) // 3
        if abs(sum(np.diag(np.fliplr(self._board)))) == 3:
            return sum(np.diag(np.fliplr(self._board))) // 3

        # Check for a tie
        if np.count_nonzero(self._board) == 9:
            return 0

        return None

    def run_game(self):
        """
        Runs a single game of Drei Gewinnt.

        Returns:
            int: 1 if player 1 wins, -1 if player 2 wins, 0 if it's a tie.
        """
        self._board = np.zeros((3, 3), dtype=int)
        players = [self._player1, self._player2]
        random.shuffle(players)
        current_player = players[0]
        other_player = players[1]

        while True:
            move = current_player.make_decision(self._board)
            if move is None:
                # No valid moves available, game ends in a tie
                self._player1.report(0)
                self._player2.report(0)
                return 0

            row, col = move
            if not self.make_move(row, col, 1 if current_player == self._player1 else -1):
                continue

            game_over = self._is_game_over()
            if game_over is not None:
                if game_over == 1:
                    self._player1.report(1)
                    self._player2.report(-1)
                    return 1
                elif game_over == -1:
                    self._player1.report(-1)
                    self._player2.report(1)
                    return -1
                else:
                    self._player1.report(0)
                    self._player2.report(0)
                    return 0

            current_player, other_player = other_player, current_player

    def run_iterations(self, num_iterations):
        """
        Runs multiple iterations of the game.

        Args:
            num_iterations (int): The number of iterations to run.
        """
        for _ in range(num_iterations):
            result = self.run_game()
            self._player1_score.append(self._player1_score[-1] + result if self._player1_score else result)
            self._player2_score.append(self._player2_score[-1] - result if self._player2_score else -result)

    def plot_scores(self):
        """Plots the scores of both players over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self._player1_score) + 1), self._player1_score, label='Player 1')
        plt.plot(range(1, len(self._player2_score) + 1), self._player2_score, label='Player 2')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Player Scores over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    player1 = DGRandomBot()
    player2 = DGRandomBot()

    game = DreiGewinnt(player1, player2)
    game.run_iterations(100)
    game.plot_scores()

    print("Player 1 stats:", player1.get_stats())
    print("Player 2 stats:", player2.get_stats())
if __name__ == '__main__':
    main()
