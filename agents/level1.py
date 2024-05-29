import sys
import argparse
import numpy as np

from NeuralNetwork import NeuralNetwork

# for each creature: [color one hot(4), type one hot(3), visible, scaned one hot[me, foe](2), x, y, vx, vy, radar one hot(4)] 18 dim
# for each drone: [drone_x, drone_y, emergency, battery] 4 dim


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
    for i in creatures.keys():
        vector.extend(creatures[i]["color"])
        vector.extend(creatures[i]["type"])
        vector.append(creatures[i]["visible"])
        vector.extend(creatures[i]["scaned"])
        vector.append(creatures[i]["x"])
        vector.append(creatures[i]["y"])
        vector.append(creatures[i]["vx"])
        vector.append(creatures[i]["vy"])
        vector.extend(creatures[i]["radar"])

    for i in drones.keys():
        vector.append(drones[i]["x"])
        vector.append(drones[i]["y"])
        vector.append(drones[i]["emergency"])
        vector.append(drones[i]["battery"])

    return np.array(vector).reshape((-1, 1))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    type=str,
    default="",
    help="weights file",
)
args = parser.parse_args()

model = NeuralNetwork(18 * 12 + 4 * 2, 100, 3)
if args.weights:
    model.load(args.weights)


pos_encoding = {"TL": 0, "TR": 1, "BL": 2, "BR": 3}
units = 10000

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
            "x": drone_x,
            "y": drone_y,
            "emergency": emergency,
            "battery": battery,
        }

    foe_drone_count = int(input())
    for i in range(foe_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]

        drones[drone_id] = {
            "x": drone_x,
            "y": drone_y,
            "emergency": emergency,
            "battery": battery,
        }

    drone_scan_count = int(input())
    for i in range(drone_scan_count):
        drone_id, creature_id = [int(j) for j in input().split()]
        print(f"{drone_id}, {creature_id}", file=sys.stderr, flush=True)

    visible_creature_count = int(input())
    for i in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = [
            int(j) for j in input().split()
        ]
        creatures[creature_id]["visible"] = 1
        creatures[creature_id]["x"] = creature_x
        creatures[creature_id]["y"] = creature_y
        creatures[creature_id]["vx"] = creature_vx
        creatures[creature_id]["vy"] = creature_vy

    radar_blip_count = int(input())
    print(f"radar_blip_count: {radar_blip_count}", file=sys.stderr, flush=True)
    for i in range(radar_blip_count):
        inputs = input().split()
        drone_id = int(inputs[0])
        creature_id = int(inputs[1])
        radar = inputs[2]

        creatures[creature_id]["radar"][pos_encoding[radar]] = 1

    for i in range(my_drone_count):

        inputs = conver2vector(creatures, drones)

        outputs = model(inputs).flatten()

        outputs[2] = 1 if outputs[2] > 0.5 else 0

        # MOVE <x> <y> <light (1|0)> | WAIT <light (1|0)>
        print(
            f"MOVE {int(units * outputs[0])} {int(units * outputs[1])} {int(outputs[2])}"
        )
