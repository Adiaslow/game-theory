import random

class PDRandomBot:
    """Represents a player who makes random moves in the Prisoner's Dilemma game."""

    def __init__(self):
        """Initializes a random player."""
        self._num_wins = 0
        self._num_losses = 0
        self._score = 0

    def make_decision(self):
        """Makes a random move (cooperate or defect)."""
        return random.choice(['C', 'D'])

    def report(self, payoff):
        """
        Updates the player's statistics based on the payoff received.

        Args:
            payoff (int): The payoff received by the player.
        """
        self._score += payoff
        if payoff > 0:
            self._num_wins += 1
        else:
            self._num_losses += 1

    def get_stats(self):
        """Returns the player's current statistics."""
        return {
            'wins': self._num_wins,
            'losses': self._num_losses,
            'score': self._score
        }
