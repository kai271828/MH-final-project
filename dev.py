import numpy as np
from agents.NeuralNetwork import NeuralNetwork
from utils import simulate, seabed_security

test_func = seabed_security("simulate.jar", "agents/level1.py", "starterAIs/SS_Starter.py", "", str(1), str(9527), difference_mode=False, verbose=False, parallel=True)
result = test_func([np.random.normal(-1, 1, size=22803) for num in range(20)])

print(result)