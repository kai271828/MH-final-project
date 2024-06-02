import numpy as np
from agents.Policy import Policy
from utils import simulate, seabed_security

policy = Policy()

policy.load("result/level2_baseline/1-th_original.txt")

# print(policy.weights)

# for k in policy.weights.keys():
#     print(k)

policy.save("dev.txt")

# print(policy.weights["dm_w_2"] == read_from_file.weights["dm_w_2"])
# print(policy.weights["act_w_2"] == read_from_file.weights["act_w_2"])

