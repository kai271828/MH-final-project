import sys
import argparse
import numpy as np

from NeuralNetwork import NeuralNetwork

# for each creature: [color one hot(4), type one hot(3), visible, scaned one hot[me, foe](2), x, y, vx, vy, radar one hot(4)] 18 dim
# for each drone: [drone_x, drone_y, emergency, battery] 4 dim


class NeuralNetwork:
    def __init__(self, input_size, hidden1_size=32, hidden2_size=32, output_size=3):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.w_1 = np.random.normal(0, 1.5, size=(self.hidden1_size, self.input_size))
        self.b_1 = np.zeros((self.hidden1_size, 1))
        self.w_2 = np.random.normal(0, 1.5, size=(self.hidden2_size, self.hidden1_size))
        self.b_2 = np.zeros((self.hidden2_size, 1))
        self.w_3 = np.random.normal(0, 1.5, size=(self.output_size, self.hidden2_size))
        self.b_3 = np.zeros((self.output_size, 1))

    def load(self, filename):
        weights = np.loadtxt(filename)

        offset = 0
        size = self.input_size * self.hidden1_size
        self.w_1 = weights[offset : offset + size].reshape(
            (self.hidden1_size, self.input_size)
        )
        offset += size

        size = self.hidden1_size
        self.b_1 = weights[offset : offset + size].reshape((self.hidden1_size, 1))
        offset += size

        size = self.hidden2_size * self.hidden1_size
        self.w_2 = weights[offset : offset + size].reshape(
            (self.hidden2_size, self.hidden1_size)
        )
        offset += size

        size = self.hidden2_size
        self.b_2 = weights[offset : offset + size].reshape((self.hidden2_size, 1))
        offset += size

        size = self.output_size * self.hidden2_size
        self.w_3 = weights[offset : offset + size].reshape(
            (self.output_size, self.hidden2_size)
        )
        offset += size

        size = self.output_size
        self.b_3 = weights[offset : offset + size].reshape((self.output_size, 1))

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
                    self.w_3.flatten(),
                    self.b_3.flatten(),
                )
            ),
        )

    def load_from_numpy(self, weights):
        offset = 0
        size = self.input_size * self.hidden1_size
        self.w_1 = weights[offset : offset + size].reshape(
            (self.hidden1_size, self.input_size)
        )
        offset += size

        size = self.hidden1_size
        self.b_1 = weights[offset : offset + size].reshape((self.hidden1_size, 1))
        offset += size

        size = self.hidden2_size * self.hidden1_size
        self.w_2 = weights[offset : offset + size].reshape(
            (self.hidden2_size, self.hidden1_size)
        )
        offset += size

        size = self.hidden2_size
        self.b_2 = weights[offset : offset + size].reshape((self.hidden2_size, 1))
        offset += size

        size = self.output_size * self.hidden2_size
        self.w_3 = weights[offset : offset + size].reshape(
            (self.output_size, self.hidden2_size)
        )
        offset += size

        size = self.output_size
        self.b_3 = weights[offset : offset + size].reshape((self.output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, inputs):
        x = self.sigmoid(self.w_1 @ inputs + self.b_1)
        x = self.sigmoid(self.w_2 @ x + self.b_2)
        x = self.sigmoid(self.w_3 @ x + self.b_3)

        return x


def reset_creatures(creatures):
    for i in creatures.keys():
        creatures[i]["visible"] = 0
        creatures[i]["x"] = -1
        creatures[i]["y"] = -1
        creatures[i]["vx"] = -1
        creatures[i]["vy"] = -1
        creatures[i]["radar"] = [0, 0, 0, 0]


def conver2vector(creatures, drones):
    vector = []
    for i in sorted(creatures.keys()):
        vector.extend(creatures[i]["color"])
        vector.extend(creatures[i]["type"])
        vector.append(creatures[i]["visible"])
        vector.extend(creatures[i]["scaned"])
        vector.append(creatures[i]["x"])
        vector.append(creatures[i]["y"])
        vector.append(creatures[i]["vx"])
        vector.append(creatures[i]["vy"])
        vector.extend(creatures[i]["radar"])

    for i in sorted(drones.keys()):
        vector.append(drones[i]["x"])
        vector.append(drones[i]["y"])
        vector.append(drones[i]["emergency"])
        vector.append(drones[i]["battery"])

    return np.array(vector).reshape((-1, 1))


def act(action, light, x, y, units):
    def clip(value, maximum=9999, minimum=0):
        return max(min(value, maximum), minimum)

    if action == 0:
        # move T
        des_x = x
        des_y = y - 600
    elif action == 1:
        # move TR
        des_x = x + 300
        des_y = y - 300
    elif action == 2:
        # move R
        des_x = x
        des_y = y + 600
    elif action == 3:
        # move BR
        des_x = x + 300
        des_y = y + 300
    elif action == 4:
        # move B
        des_x = x
        des_y = y + 600
    elif action == 5:
        # move BL
        des_x = x - 300
        des_y = y + 300
    elif action == 6:
        # move L
        des_x = x - 600
        des_y = y
    elif action == 7:
        # move TL
        des_x = x - 300
        des_y = y - 300
    elif action == 8:
        # wait
        print(f"WAIT {light}")
        return

    # MOVE <x> <y> <light (1|0)> | WAIT <light (1|0)>
    print(f"MOVE {clip(des_x)} {clip(des_y)} {light}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    type=str,
    default="",
    help="weights file",
)
args = parser.parse_args()


model = NeuralNetwork(18 * 12 + 4 * 2)

if args.weights:
    model.load(args.weights)


pos_encoding = {"TL": 0, "TR": 1, "BL": 2, "BR": 3}
units = 10000
action_lock = False

creature_count = int(input())
creatures = {}
for i in range(creature_count):
    creature_id, color, _type = [int(j) for j in input().split()]
    creature_dict = {
        "color": [0, 0, 0, 0],
        "type": [0, 0, 0],
        "visible": 0,
        "scaned": [0, 0],
        "x": -1,
        "y": -1,
        "vx": -1,
        "vy": -1,
        "radar": [0, 0, 0, 0],
    }
    creature_dict["color"][color] = 1
    creature_dict["type"][_type] = 1
    creatures[creature_id] = creature_dict

drones = {}

# game loop
while True:

    reset_creatures(creatures)

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for i in range(my_scan_count):
        creature_id = int(input())

        creatures[creature_id]["scaned"][0] = 1

    foe_scan_count = int(input())
    for i in range(foe_scan_count):
        creature_id = int(input())

        creatures[creature_id]["scaned"][1] = 1

    my_drone_count = int(input())
    for i in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]

        drones[drone_id] = {
            "x": drone_x / units,
            "y": drone_y / units,
            "emergency": emergency,
            "battery": battery,
        }

    foe_drone_count = int(input())
    for i in range(foe_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]

        drones[drone_id] = {
            "x": drone_x / units,
            "y": drone_y / units,
            "emergency": emergency,
            "battery": battery,
        }

    drone_scan_count = int(input())
    for i in range(drone_scan_count):
        drone_id, creature_id = [int(j) for j in input().split()]
        if drone_id == my_drone_id:
            creatures[creature_id]["my_record"][0] = 1
        else:
            creatures[creature_id]["foe_record"][0] = 1

    visible_creature_count = int(input())
    for i in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = [
            int(j) for j in input().split()
        ]
        creatures[creature_id]["visible"] = 1
        creatures[creature_id]["x"] = creature_x / units
        creatures[creature_id]["y"] = creature_y / units
        creatures[creature_id]["vx"] = creature_vx / units
        creatures[creature_id]["vy"] = creature_vy / units

    radar_blip_count = int(input())
    print(f"radar_blip_count: {radar_blip_count}", file=sys.stderr, flush=True)
    for i in range(radar_blip_count):
        inputs = input().split()
        drone_id = int(inputs[0])
        creature_id = int(inputs[1])
        radar = inputs[2]

        creatures[creature_id]["radar"][pos_encoding[radar]] = 1

    for i in range(my_drone_count):

        if action_lock:
            if drones[my_drone_id]["y"] <= 500:
                action_lock = False
            else:
                print(f"MOVE {drones[my_drone_id]['x']} 0 0")
                continue

        inputs = conver2vector(creatures, drones)

        outputs = model(inputs).flatten()

        action = np.argmax(outputs[:-1])

        if action == 10:
            action_lock = True
            print(f"MOVE {drones[my_drone_id]['x']} 0 0")
            continue

        light = 1 if outputs[-1] > 0.5 else 0

        act(action, light, drones[my_drone_id]["x"], drones[my_drone_id]["y"], units)
