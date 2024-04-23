"""
Module for the Drei Gewinnt (Tic Tac Toe) game.
"""
# Internal Modules
from pd_bot_random import PDRandomBot

import random
import matplotlib.pyplot as plt

class PrisonersDilemma:
    """Represents the Prisoner's Dilemma game."""

    def __init__(self, player1, player2, payoff_matrix):
        """
        Initializes the Prisoner's Dilemma game.

        Args:
            player1 (object): The first player object.
            player2 (object): The second player object.
            payoff_matrix (dict): The payoff matrix for the game.
        """
        self._player1 = player1
        self._player2 = player2
        self._payoff_matrix = payoff_matrix
        self._player1_score = []
        self._player2_score = []

    def run_game(self):
        """
        Runs a single round of the Prisoner's Dilemma game.

        Returns:
            tuple: The payoffs for player 1 and player 2.
        """
        move1 = self._player1.make_decision()
        move2 = self._player2.make_decision()
        payoff1, payoff2 = self._payoff_matrix[(move1, move2)]
        self._player1.report(payoff1)
        self._player2.report(payoff2)
        return payoff1, payoff2

    def run_iterations(self, num_iterations):
        """
        Runs multiple iterations of the game.

        Args:
            num_iterations (int): The number of iterations to run.
        """
        for _ in range(num_iterations):
            payoff1, payoff2 = self.run_game()
            self._player1_score.append(payoff1)
            self._player2_score.append(payoff2)

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

class RandomPlayer:
    """Represents a player who makes random moves."""

    def __init__(self):
        """Initializes a random player."""
        self._score = 0

    def make_decision(self):
        """Makes a random move (cooperate or defect)."""
        return random.choice(['C', 'D'])

    def report(self, payoff):
        """Updates the player's score based on the payoff."""
        self._score += payoff

    def get_score(self):
        """Returns the player's current score."""
        return self._score

def main():
    payoff_matrix = {
        ('C', 'C'): (3, 3),  # Both players cooperate
        ('C', 'D'): (0, 5),  # Player 1 cooperates, Player 2 defects
        ('D', 'C'): (5, 0),  # Player 1 defects, Player 2 cooperates
        ('D', 'D'): (1, 1)   # Both players defect
    }

    player1 = RandomPlayer()
    player2 = RandomPlayer()

    game = PrisonersDilemma(player1, player2, payoff_matrix)
    game.run_iterations(100)
    game.plot_scores()

    print("Player 1 score:", player1.get_score())
    print("Player 2 score:", player2.get_score())

if __name__ == '__main__':
    main()
