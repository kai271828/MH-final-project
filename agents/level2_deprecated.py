import sys
import argparse
import numpy as np

from NeuralNetwork import NeuralNetwork

# for each creature: [color one hot(4), type one hot(3), my_record[scaned, saved](2), foe_record[scaned, saved](2), radar one hot(4)] 15 dim
# for each drone: [drone_x, drone_y, battery] 3 dim

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    type=str,
    default="",
    help="weights file",
)
try:
    args = parser.parse_args(args=[])
    
except SystemExit as e:
    print(f"Error parsing arguments: {e}")


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

    des_x = visible[0]["x"] + visible[0]["vx"]
    des_y = visible[0]["y"] + visible[0]["vy"]

    print(f"MOVE {int(des_x)} {int(des_y)} 1")


def act(action, light, x, y, units):
    def clip(value, maximum=units-1, minimum=0):
        return max(min(int(value), maximum), minimum)

    x *= units
    y *= units

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


model = NeuralNetwork(15 * 12 + 3 * 2, 50, 50, 11) # NeuralNetwork(15 * 12 + 3 * 2)

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
            "x": drone_x / units,
            "y": drone_y / units,
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
            "x": drone_x / units,
            "y": drone_y / units,
            # "emergency": emergency,
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
        if creatures[creature_id]["my_record"][0] == 0:
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

    for item in sorted(drones.items()):
        print(
            item,
            file=sys.stderr,
            flush=True,
        )
    print(
            f"visible: {visible}",
            file=sys.stderr,
            flush=True,
        )

    for i in range(my_drone_count):

        if action_lock:
            if drones[my_drone_id]["y"] <= 500:
                action_lock = False
            else:
                print(f"MOVE {drones[my_drone_id]['x']} 0 0")
                continue

        if visible_creature_count > 0:
            auto_route(
                visible, drones[my_drone_id]["x"], drones[my_drone_id]["y"]
            )
            continue

        inputs = conver2vector(creatures, drones)

        outputs = model(inputs).flatten()

        action = np.argmax(outputs[:-1])

        print(
            f"action: {action}",
            file=sys.stderr,
            flush=True,
        )

        if action == 9:
            action_lock = True
            print(f"MOVE {int(drones[my_drone_id]['x'])} 0 0")
            continue

        light = 1 if outputs[-1] > 0.5 else 0

        act(action, light, drones[my_drone_id]["x"], drones[my_drone_id]["y"], units)
