import numpy as np
from agents.NeuralNetwork import NeuralNetwork
from utils import simulate, seabed_security

model = NeuralNetwork(15 * 12 + 3 * 2)

test = np.random.normal(0, 1, (11, 1))

test[:-1] = model.softmax(test[:-1])
test[-1] = model.sigmoid(test[-1])

print(test)
print(np.argmax(test[:-1]))
