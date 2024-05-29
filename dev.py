import numpy as np
from agents.NeuralNetwork import NeuralNetwork
from utils import simulate, seabed_security

model = NeuralNetwork(18 * 12 + 4 * 2)

test = np.random.normal(0, 1, 18 * 12 + 4 * 2).reshape((-1, 1))

model.save("dev.txt")
model.load("dev.txt")

result = model(test)

print(model.w_1.size + model.b_1.size + model.w_2.size + model.b_2.size +  model.w_3.size + model.b_3.size)

print(result)


