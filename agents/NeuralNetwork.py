import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_1 = np.random.uniform(-1, 1, size=(self.hidden_size, self.input_size))
        self.b_1 = np.zeros((self.hidden_size, 1))
        self.w_2 = np.random.uniform(-1, 1, size=(self.output_size, self.hidden_size))
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

    def save(self, filename):
        # print(self.w_1.size + self.b_1.size + self.w_2.size + self.b_2.size)
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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, inputs):
        x = self.sigmoid(self.w_1 @ inputs + self.b_1)
        x = self.sigmoid(self.w_2 @ x + self.b_2)

        return x
