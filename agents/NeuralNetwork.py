import numpy as np


class NeuralNetwork:
    """The first 10 elements of output mean move TOP, move TR, move R, move BR, move BOTTOM, move BL, move LEFT, move TL, wait, back, and the final element means light."""

    def __init__(self, input_size=63, hidden_size=10, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_1 = np.random.normal(0, 1.5, size=(self.hidden_size, self.input_size))
        self.b_1 = np.zeros((self.hidden_size, 1))
        self.w_2 = np.random.normal(0, 1.5, size=(self.output_size, self.hidden_size))
        self.b_2 = np.zeros((self.output_size, 1))

    def load(self, filename):
        weights = np.loadtxt(filename)

        offset = 0
        size = self.input_size * self.hidden_size
        self.w_1 = weights[offset : offset + size].reshape(
            (self.hidden_size, self.input_size)
        )
        offset += size

        size = self.hidden_size
        self.b_1 = weights[offset : offset + size].reshape((self.hidden_size, 1))
        offset += size

        size = self.output_size * self.hidden_size
        self.w_2 = weights[offset : offset + size].reshape(
            (self.output_size, self.hidden_size)
        )
        offset += size

        size = self.output_size
        self.b_2 = weights[offset : offset + size].reshape((self.output_size, 1))
        offset += size


    def save(self, filename):
        np.savetxt(
            filename,
            np.concatenate(
                (
                    self.w_1.flatten(),
                    self.b_1.flatten(),
                    self.w_2.flatten(),
                    self.b_2.flatten(),
                )
            ),
        )

    def load_from_numpy(self, weights):
        offset = 0
        size = self.input_size * self.hidden_size
        self.w_1 = weights[offset : offset + size].reshape(
            (self.hidden_size, self.input_size)
        )
        offset += size

        size = self.hidden_size
        self.b_1 = weights[offset : offset + size].reshape((self.hidden_size, 1))
        offset += size

        size = self.output_size * self.hidden_size
        self.w_2 = weights[offset : offset + size].reshape(
            (self.output_size, self.hidden_size)
        )
        offset += size

        size = self.output_size
        self.b_2 = weights[offset : offset + size].reshape((self.output_size, 1))
        offset += size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_sum = np.sum(np.exp(x))
        return np.exp(x) / exp_sum

    def __call__(self, inputs):
        x = np.tanh(self.w_1 @ inputs + self.b_1)
        x = self.sigmoid(self.w_2 @ x + self.b_2)

        return x
    
    def size(self):
       return self.w_1.size + self.b_1.size + self.w_2.size + self.b_2.size
