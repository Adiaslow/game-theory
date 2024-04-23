"""
Module for Adam's Multilayer Perceptron (MLP) player for Drei Gewinnt (Tic Tac Toe).
"""

# External Modules
import itertools
from joblib import dump, load
import numpy as np
import os
import pickle
import random
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

class Neuron:
    """Represents a single neuron in a neural network."""
    def __init__(self, num_inputs):
        """
        Initializes a neuron with random weights and bias.

        Args:
            num_inputs (int): The number of input values.
        """
        self._weights = np.random.rand(num_inputs)
        self._bias = random.random()
        self._num_inputs = num_inputs

    def get_weights(self):
        """Returns the weights of the neuron."""
        return self._weights

    def get_bias(self):
        """Returns the bias of the neuron."""
        return self._bias

    def set_weights(self, weights):
        """Sets the weights of the neuron."""
        self._weights = weights

    def set_bias(self, bias):
        """Sets the bias of the neuron."""
        self._bias = bias

    def _sigmoid(self, x):
        """Calculates the sigmoid activation function.

        Args:
            x (numpy.ndarray): The input values.

        Returns:
            numpy.ndarray: The output values.
        """
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Calculates the softmax
        Args:
            x (numpy.ndarray): The input values.

        Returns:
            numpy.ndarray: The output values.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def output(self, x):
        """Calculates the output of the neuron.

        Args:
            x (numpy.ndarray): The input values.

        Returns:
            float: The output of the neuron.
        """
        return self._sigmoid(np.dot(x, self._weights) + self._bias)

class MultilayerPerceptron:
    """Represents a Multilayer Perceptron."""
    def __init__(self, num_inputs=9, num_outputs=1, num_hidden_layers=2, num_neurons_per_layer=9):
        """Initializes the Multilayer Perceptron.

        Args:
            num_inputs (int): The number of input neurons.
            num_outputs (int): The number of output neurons.
            num_hidden_layers (int): The number of hidden layers.
            num_neurons_per_layer (int): The number of neurons per hidden layer.
        """
        self._layers = []
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._num_hidden_layers = num_hidden_layers
        self._num_neurons_per_layer = num_neurons_per_layer

        # Initialize the input layer
        self._layers.append([Neuron(num_inputs) for _ in range(num_neurons_per_layer)])

        # Initialize the hidden layers
        for _ in range(num_hidden_layers):
            self._layers.append([Neuron(num_neurons_per_layer) for _ in range(num_neurons_per_layer)])

        # Initialize the output layer
        self._layers.append([Neuron(num_neurons_per_layer) for _ in range(num_outputs)])

    def _sigmoid(self, x):
        """Calculates the sigmoid activation function.

        Args:
            x (numpy.ndarray): The input values.

        Returns:
            numpy.ndarray: The output values.
        """
        return 1 / (1 + np.exp(-x[..., np.newaxis]))

    def _forward_propagation(self, x):
        """Performs forward propagation through the neural network.

        Args:
            x (numpy.ndarray): The input values.

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        # Flatten the input data
        x = x.flatten()

        for layer in self._layers:
            layer_outputs = []
            for neuron in layer:
                layer_outputs.append(neuron.output(x))
            x = np.array(layer_outputs)

        return x

        for layer in self._layers:
            outputs = []
            for neuron in layer:
                outputs.append(neuron.output(x))
            x = np.array(outputs)
        return x

    def _mean_squared_error(self, y_true, y_pred):
        """Calculates the mean squared error between the true and predicted values.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.

        Returns:
            float: The mean squared error.
        """
        return np.mean((y_true - y_pred) ** 2)

    def _get_weights(self):
        """Returns the current weights of the neural network."""
        weights = []
        for layer in self._layers:
            layer_weights = []
            for neuron in layer:
                layer_weights.append((neuron.get_weights(), neuron.get_bias()))
            weights.append(layer_weights)
        return weights

    def _set_weights(self, weights):
        """Sets the weights of the neural network."""
        for layer, layer_weights in zip(self._layers, weights):
            for neuron, (neuron_weights, neuron_bias) in zip(layer, layer_weights):
                neuron.set_weights(neuron_weights)
                neuron.set_bias(neuron_bias)

    def train(self, x_train, y_train, epochs=100, learning_rate=0.01, batch_size=32, validation_data=None, patience=10):
        """Trains the neural network on the given training data with early stopping.

        Args:
            x_train (numpy.ndarray): The input training data.
            y_train (numpy.ndarray): The target training data.
            epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for updating the weights and biases.
            batch_size (int): The number of samples to process in each batch.
            validation_data (tuple): A tuple of (x_val, y_val) for validation data. (optional)
            patience (int): The number of epochs with no improvement to wait before early stopping. (optional)
        """
        num_samples = len(x_train)
        best_weights = None  # Store weights of the best model
        best_validation_loss = float('inf')  # Initialize with high value
        early_stop_counter = 0
        for epoch in tqdm(range(epochs), desc="Training Model"):
            for i in range(0, num_samples, batch_size):
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                for x, y in zip(batch_x, batch_y):
                    # Forward propagation for single data point
                    output = self._forward_propagation(x)

                    # Backpropagation for single data point
                    error = self._mean_squared_error(y, output)
                    delta = error * output * (1 - output)

                    # Update weights and biases
                    for i in range(len(self._layers) - 1, -1, -1):
                        if i == len(self._layers) - 1:
                            # Output layer
                            layer_weights = np.array([neuron.get_weights() for neuron in self._layers[i]])
                            layer_biases = np.array([neuron.get_bias() for neuron in self._layers[i]])
                            layer_outputs = np.array([neuron.output(x) for neuron in self._layers[i-1]])
                            layer_weights += learning_rate * np.dot(delta[:, np.newaxis], layer_outputs[:, np.newaxis].T)
                            layer_biases += learning_rate * delta
                            for j, neuron in enumerate(self._layers[i]):
                                neuron.set_weights(layer_weights[j])
                                neuron.set_bias(layer_biases[j])
                        else:
                            # Hidden layers
                            layer_weights = np.array([neuron.get_weights() for neuron in self._layers[i]])
                            layer_biases = np.array([neuron.get_bias() for neuron in self._layers[i]])
                            error = np.dot(delta, self._layers[i+1][:, np.newaxis].get_weights().T)
                            layer_outputs = np.array([neuron.output(x) for neuron in self._layers[i]])
                            delta_hidden = error * layer_outputs * (1 - layer_outputs)
                            layer_weights += learning_rate * np.dot(delta_hidden.T, np.array([neuron.output(x) for neuron in self._layers[i-1]])[:, np.newaxis])
                            layer_biases += learning_rate * delta_hidden.squeeze()
                            for j, neuron in enumerate(self._layers[i]):
                                neuron.set_weights(layer_weights[j])
                                neuron.set_bias(layer_biases[j])

            # Validation (if validation data provided)
            if validation_data:
                validation_loss = self._evaluate(validation_data[0], validation_data[1])
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_weights = self._get_weights()  # Store the best weights
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss")
                        self._set_weights(best_weights)  # Restore best weights
                        return

        # Training finished, restore best weights if applicable
        if validation_data and best_weights is not None:
            self._set_weights(best_weights)

    def _evaluate(self, x_val, y_val):
        """Evaluates the model on the validation data.

        Args:
            x_val (numpy.ndarray): The input validation data.
            y_val (numpy.ndarray): The target validation data.

        Returns:
            float: The validation loss (mean squared error).
        """
        # Forward propagation on validation data
        y_pred = np.array([self._forward_propagation(x) for x in x_val])

        # Calculate validation loss
        validation_loss = self._mean_squared_error(y_val, y_pred)

        return validation_loss

    def _predict(self, board):
        """Runs inference on the given board state to determine the next move.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            tuple: The row and column of the next move.
        """
        # Flatten the board state into a 1D array
        input_data = board.flatten()

        # Perform forward propagation to get the network's output
        output = self._forward_propagation(input_data)

        # Find the index of the maximum output value
        move_index = np.argmax(output)

        # Convert the move index to row and column
        row = move_index // 3
        col = move_index % 3

        return row, col

    def save_model(self, filename):
        """Saves the model to a file.

        Args:
            filename (str): The name of the file to save the model to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filename):
        """Loads a model from a file.

        Args:
            filename (str): The name of the file to load the model from.

        Returns:
            MultilayerPerceptron: The loaded model.

        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

class DGDataGenerator:
    """Generates training data for the Drei Gewinnt model."""

    def __init__(self):
        """Initializes the data generator."""
        print("Generating all possible Tic Tac Toe boards...")
        self._boards = self._generate_all_boards()
        print(f"{len(self._boards)} boards generated.")
        print("Evaluating all possible Tic Tac Toe boards...")
        self._scores = self._evaluate_boards(self._boards)

    def _generate_all_possible_boards(self, board, player):
        """Generates all possible board states for the given player.

        Args:
            board (numpy.ndarray): The current state of the board.
            player (int): The player making the move.

        Returns:
            list: A list of all possible board states.
        """
        boards = []
        empty_positions = np.argwhere(board == 0)

        if len(empty_positions) == 0:
            boards.append(board)
        else:
            for pos in empty_positions:
                new_board = board.copy()
                new_board[pos[0], pos[1]] = player
                next_player = -player
                boards.extend(self._generate_all_possible_boards(new_board, next_player))

        return boards

    def _generate_all_boards(self):
        """Generates all possible valid Tic Tac Toe boards.

        Returns:
            list: A list of all possible board states.
        """
        empty_board = np.zeros((3, 3), dtype=int)
        boards = self._generate_all_possible_boards(empty_board, 1)
        flattened_boards = [board.flatten() for board in boards]
        return flattened_boards

    def _evaluate_board(self, board):
        """
        Evaluates the score of the current board state.

        Args:
            board (numpy.ndarray): The current state of the board.

        Returns:
            float: The score of the board state.
        """
        board = board.reshape((3, 3))
        score = 0

        # Check rows
        row_sums = np.sum(board, axis=1)
        score += np.count_nonzero(row_sums == 2) ^ np.count_nonzero(row_sums == -2)

        # Check columns
        col_sums = np.sum(board, axis=0)
        score += np.count_nonzero(col_sums == 2) ^ np.count_nonzero(col_sums == -2)

        # Check diagonals
        diag_sum1 = np.trace(board)
        diag_sum2 = np.trace(np.fliplr(board))
        score += (diag_sum1 == 2) ^ (diag_sum1 == -2) + (diag_sum2 == 2) ^ (diag_sum2 == -2)


        return score

    def _evaluate_boards(self, boards):
        """
        Scores the given list of boards.

        Args:
            boards (list): A list of board states.

        Returns:
            list: A list of scores for each board state.
        """
        return [self._evaluate_board(board) for board in boards]

class DGAdamMLP:
    """Represents a Drei Gewinnt player that uses Adam's Multilayer Perceptron algorithm."""
    def __init__(self):
        """
        Initializes the Drei Gewinnt player.
        """
        self._board = np.zeros((3, 3), dtype=int)

        if os.path.exists('drei_gewinnt/dg_adam_mlp_model.pkl'):
            try:
                self.network = load('drei_gewinnt/dg_adam_mlp_model.pkl')
                print("Loaded model.")
            except (EOFError, ValueError):
                print("Error loading model. Please check the model file.")
                raise
        else:
            print("No existing model found. Please train and save the model first.")
            raise FileNotFoundError("Model file not found.")

        self._num_wins = 0
        self._num_losses = 0
        self._num_ties = 0

    def make_decision(self, board):
            """
            Makes a move based on the current board state using the Multilayer Perceptron.

            Args:
                board (numpy.ndarray): The current state of the board.

            Returns:
                tuple: The row and column of the selected move, or None if no valid moves are available.
            """
            valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
            if valid_moves:
                flattened_board = board.flatten()
                scores = self.network.predict([flattened_board])[0]
                best_move_index = np.argmax(scores)
                row, col = valid_moves[best_move_index]
                return row, col
            else:
                print("MLP: No valid moves available.")
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

def main():
    """Main function for training the model."""

    # Create a data generator
    data_generator = DGDataGenerator()

    # Create a player
    player = DGAdamMLP()
    x_train, x_test, y_train, y_test = train_test_split(data_generator._boards, data_generator._scores, test_size=0.2, random_state=42)

    # Train the player
    player.network.fit(x_train, y_train)

    # Evaluate the player
    score = player.network.score(x_test, y_test)
    print(f"Test score: {score}")

    # Save the player
    with open('dg_adam_mlp_model.pkl', 'wb') as f:
        dump(player.network, 'dg_adam_mlp_model.pkl')

if __name__ == '__main__':
    main()
