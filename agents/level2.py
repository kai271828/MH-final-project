import sys
import argparse
import numpy as np

from NeuralNetwork import NeuralNetwork

# for each creature: [color one hot(4), type one hot(3), my_record[scaned, saved](2), foe_record[scaned, saved](2), radar one hot(4)] 15 dim
# for each drone: [drone_x, drone_y, battery] 3 dim


def reset_creatures(creatures):
    for i in creatures.keys():
        creatures[i]["radar"] = [0, 0, 0, 0]


def conver2vector(creatures, drones):
    vector = []

    for i in sorted(drones.keys()):
        vector.append(drones[i]["x"])
        vector.append(drones[i]["y"])
        # vector.append(drones[i]["emergency"])
        vector.append(drones[i]["battery"])

    for i in sorted(creatures.keys()):
        vector.extend(creatures[i]["color"])
        vector.extend(creatures[i]["type"])
        vector.extend(creatures[i]["my_record"])
        vector.extend(creatures[i]["foe_record"])
        vector.extend(creatures[i]["radar"])

    return np.array(vector).reshape((-1, 1))


def auto_route(visible, x, y):
    for i in range(len(visible)):
        visible[i]["distance"] = np.sqrt(
            (x - visible[i]["x"]) ** 2 + (y - visible[i]["y"]) ** 2
        )
    visible.sort(key=lambda e: e["distance"])

    des_x = -1
    des_y = -1

    for i in range(len(visible)):
        if creatures[visible[i]["id"]]["my_record"][0] == 0:
            des_x = visible[i]["x"] + visible[i]["vx"]
            des_y = visible[i]["y"] + visible[i]["vy"]
            break

    if des_x == -1 or des_y == -1:
        return False
    else:
        print(f"MOVE {int(des_x)} {int(des_y)} 1")
        return True


parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    type=str,
    default="",
    help="weights file",
)
args = parser.parse_args()


model = NeuralNetwork(15 * 12 + 4 * 2)

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
        "my_record": [0, 0],
        "foe_record": [0, 0],
        "radar": [0, 0, 0, 0],
    }
    creature_dict["color"][color] = 1
    creature_dict["type"][_type] = 1
    creatures[creature_id] = creature_dict

drones = {}

# game loop
while True:

    reset_creatures(creatures)
    visible = []

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for i in range(my_scan_count):
        creature_id = int(input())

        creatures[creature_id]["my_record"][1] = 1

    foe_scan_count = int(input())
    for i in range(foe_scan_count):
        creature_id = int(input())

        creatures[creature_id]["foe_record"][1] = 1

    my_drone_count = int(input())
    for i in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]
        my_drone_id = drone_id

        drones[drone_id] = {
            "x": drone_x,
            "y": drone_y,
            # "emergency": emergency,
            "battery": battery,
        }

    foe_drone_count = int(input())
    for i in range(foe_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]
        foe_drone_id = drone_id

        drones[drone_id] = {
            "x": drone_x,
            "y": drone_y,
            # "emergency": emergency,
            "battery": battery,
        }

    drone_scan_count = int(input())
    for i in range(drone_scan_count):
        drone_id, creature_id = [int(j) for j in input().split()]
        if drone_id == my_drone_id:
            creatures[creature_id]["my_record"][0] = 1
        else:
            creatures[creature_id]["my_record"][0] = 1

    visible_creature_count = int(input())
    for i in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = [
            int(j) for j in input().split()
        ]
        visible.append(
            {
                "id": creature_id,
                "x": creature_x,
                "y": creature_y,
                "vx": creature_vx,
                "vy": creature_vy,
            }
        )

    radar_blip_count = int(input())
    print(f"radar_blip_count: {radar_blip_count}", file=sys.stderr, flush=True)
    for i in range(radar_blip_count):
        inputs = input().split()
        drone_id = int(inputs[0])
        creature_id = int(inputs[1])
        radar = inputs[2]

        creatures[creature_id]["radar"][pos_encoding[radar]] = 1

    for item in sorted(creatures.items()):
        print(
            item,
            file=sys.stderr,
            flush=True,
        )

    for i in range(my_drone_count):

        if action_lock:
            if drones[my_drone_id]["y"] <= 500:
                action_lock = False
            else:
                print(f"MOVE {drones[my_drone_id]['x']} 0 0")

        inputs = conver2vector(creatures, drones)

        outputs = model(inputs).flatten()

        action = np.argmax(outputs[:-1])
        print(
            f"action: {action}, {drones[my_drone_id]['x']}, {drones[my_drone_id]['y']}",
            file=sys.stderr,
            flush=True,
        )

        if action == 9:
            action_lock = True
            print(f"MOVE {drones[my_drone_id]['x']} 0 0")

        light = 1 if outputs[-1] > 0.5 else 0

        act(action, light, drones[my_drone_id]["x"], drones[my_drone_id]["y"], units)
