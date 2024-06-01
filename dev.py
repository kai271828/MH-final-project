import numpy as np
from agents.Policy import Policy
from utils import simulate, seabed_security

policy = Policy()
read_from_file = Policy(dm_hidden_size=128)

print(read_from_file.num_parameters)

# print(policy.weights)

# for k in policy.weights.keys():
#     print(k)

policy.save("dev.txt")
read_from_file.load("dev.txt")

# print(policy.weights["dm_w_2"] == read_from_file.weights["dm_w_2"])
# print(policy.weights["act_w_2"] == read_from_file.weights["act_w_2"])

