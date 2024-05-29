import numpy as np
from agents.NeuralNetwork import NeuralNetwork
from utils import simulate, seabed_security

model = NeuralNetwork(18 * 12 + 4 * 2, 32, 3)

model.save("dev.txt")
