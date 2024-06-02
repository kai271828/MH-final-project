import numpy as np


class Policy:
    """
    decision_maker((fish_info, scan_record, my_pos, foe_pos)) = fish_index or go to surface
    actor((selected_fish_radar, my_drone_info)) = move eight directions or wait
    """

    def __init__(
        self, 
        dm_input_size=136,
        dm_hidden_size=64, 
        dm_output_size=13, 
        act_input_size=7,
        act_hidden_size=32, 
        act_output_size=9, 
    ):
        
        self.size_list = [
            dm_input_size, 
            dm_hidden_size, 
            dm_output_size, 
            act_input_size, 
            act_hidden_size, 
            act_output_size
        ]

        self.weights = {}

        self.weights["dm_w_1"] = np.random.normal(0, 1.5, size=(self.size_list[1], self.size_list[0]))
        self.weights["dm_b_1"] = np.zeros((self.size_list[1], 1))
        self.weights["dm_w_2"] = np.random.normal(0, 1.5, size=(self.size_list[1], self.size_list[1]))
        self.weights["dm_b_2"] = np.zeros((self.size_list[1], 1))
        self.weights["dm_w_3"] = np.random.normal(0, 1.5, size=(self.size_list[2], self.size_list[1]))
        self.weights["dm_b_3"] = np.zeros((self.size_list[2], 1))

        self.weights["act_w_1"] = np.random.normal(0, 1.5, size=(self.size_list[4], self.size_list[3]))
        self.weights["act_b_1"] = np.zeros((self.size_list[4], 1))
        self.weights["act_w_2"] = np.random.normal(0, 1.5, size=(self.size_list[5], self.size_list[4]))
        self.weights["act_b_2"] = np.zeros((self.size_list[5], 1))


    def load(self, filename):
        weights = np.loadtxt(filename)

        offset = 0

        for k in self.weights.keys():
           self.weights[k] = weights[offset : offset + self.weights[k].size].reshape(
                self.weights[k].shape
           )
           offset += self.weights[k].size

    def save(self, filename, precision=2):
        np.savetxt(
            filename,
            np.concatenate(
                [weight.flatten() for weight in self.weights.values()]
            ),
            newline=",", 
            fmt=f"%.{precision}f",
        )

    def load_from_numpy(self, weights):
        offset = 0

        for k in self.weights.keys():
           self.weights[k] = weights[offset : offset + self.weights[k].size].reshape(
                self.weights[k].shape
           )
           offset += self.weights[k].size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_sum = np.sum(np.exp(x))
        return np.exp(x) / exp_sum

    def decision_maker(self, inputs):
        x = np.tanh(self.weights["dm_w_1"] @ inputs + self.weights["dm_b_1"])
        x = np.tanh(self.weights["dm_w_2"] @ x + self.weights["dm_b_2"])
        x = self.softmax(self.weights["dm_w_3"] @ x + self.weights["dm_b_3"])
        return x
    
    def actor(self, inputs):
        x = self.sigmoid(self.weights["act_w_1"] @ inputs + self.weights["act_b_1"])
        x = self.softmax(self.weights["act_w_2"] @ x + self.weights["act_b_2"])
        return x

    
    @property
    def num_parameters(self):
       return sum([weight.size for weight in self.weights.values()])
