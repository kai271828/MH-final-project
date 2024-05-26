from neural import NeuralNetwork

model = NeuralNetwork(18 * 12 + 4 * 2, 100, 3)

model.save("weight.txt")