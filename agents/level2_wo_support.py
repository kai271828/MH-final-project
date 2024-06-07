import sys
import argparse
import numpy as np

from Policy import Policy

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
        creatures[i]["radar"] = [[0, 0, 0, 0], [0, 0, 0, 0]]

def create_dm_inputs(creatures, drones, my_drone_id, units):

    vector = []
    vector.append(drones[my_drone_id]["x"] / units)
    vector.append(drones[my_drone_id]["y"] / units)

    for i in foe_drone_ids:
        vector.append(drones[i]["x"] / units)
        vector.append(drones[i]["y"] / units)

    for i in sorted(creatures.keys())[:12]:
        vector.extend(creatures[i]["color"])
        vector.extend(creatures[i]["type"])
        vector.extend(creatures[i]["my_record"])
        vector.extend(creatures[i]["foe_record"])

    return np.array(vector).reshape((-1, 1))

def create_act_inputs(creatures, target_id, drones, my_drone_id, units):

    vector = []
    vector.append(drones[my_drone_id]["x"] / units)
    vector.append(drones[my_drone_id]["y"] / units)
    vector.append(drones[my_drone_id]["battery"] / 30)

    vector.extend(creatures[target_id]["radar"][my_drone_ids.index(my_drone_id)])

    for i in sorted(creatures.keys())[12:]:
       
        vector.extend(creatures[i]["radar"][my_drone_ids.index(my_drone_id)])

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

def clip(value, maximum=9999, minimum=0):
    return max(min(int(value), maximum), minimum)

policy = Policy()

creature_count = int(input())
creatures = {}
for i in range(creature_count):
    creature_id, color, _type = [int(j) for j in input().split()]
    creature_dict = {
        "color": [0, 0, 0, 0],
        "type": [0, 0, 0],
        "my_record": [0, 0],
        "foe_record": [0, 0],
        "radar": [[0, 0, 0, 0], [0, 0, 0, 0]],
    }
    
    if _type == -1:
        creature_dict["type"] = -1
    else:
        creature_dict["type"][_type] = 1
        creature_dict["color"][color] = 1

    creatures[creature_id] = creature_dict

pos_encoding = {"TL": 0, "TR": 1, "BL": 2, "BR": 3}
units = 10000
action_lock = [False, False]
need_decision = [True, True]
targets = [2, 3]
same_target_turn = np.array([0, 0])
my_drone_ids = []
foe_drone_ids = []
# penalty = np.array([0] * 12)
# pos_penalty = np.array([0, 0])



drones = {}

# game loop
while True:

    same_target_turn += 1
    reset_creatures(creatures)
    visible = []

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())

    for i in range(my_scan_count):
        creature_id = int(input())
        creatures[creature_id]["my_record"][1] = 1
        creatures[creature_id]["my_record"][0] = 0
        # penalty[creature_id - 2] = 10

    foe_scan_count = int(input())

    for i in range(foe_scan_count):
        creature_id = int(input())

        creatures[creature_id]["foe_record"][1] = 1
        creatures[creature_id]["foe_record"][0] = 0

    my_drone_count = int(input())
    for i in range(my_drone_count):
        drone_id, drone_x, drone_y, emergency, battery = [
            int(j) for j in input().split()
        ]

        if not drone_id in my_drone_ids:
            my_drone_ids.append(drone_id)

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

        if not drone_id in foe_drone_ids:
            foe_drone_ids.append(drone_id)

        drones[drone_id] = {
            "x": drone_x,
            "y": drone_y,
            "emergency": emergency,
            "battery": battery,
        }

    drone_scan_count = int(input())
    for i in range(drone_scan_count):
        drone_id, creature_id = [int(j) for j in input().split()]
        if drone_id in my_drone_ids:
            creatures[creature_id]["my_record"][0] = 1
        else:
            creatures[creature_id]["foe_record"][0] = 1

    visible_creature_count = int(input())
    for i in range(visible_creature_count):
        creature_id, creature_x, creature_y, creature_vx, creature_vy = [
            int(j) for j in input().split()
        ]
        if creatures[creature_id]["my_record"][0] == 0 and creatures[creature_id]["my_record"][1] == 0:
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

        creatures[creature_id]["radar"][my_drone_ids.index(drone_id)][pos_encoding[radar]] = 1

    for item in sorted(drones.items()):
        print(
            item,
            file=sys.stderr,
            flush=True,
        )
    for item in sorted(creatures.items()):
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

        if visible:
            auto_route(
                visible, drones[my_drone_ids[i]]["x"], drones[my_drone_ids[i]]["y"]
            )
            continue

        if action_lock[i]:
            if drones[my_drone_ids[i]]["y"] <= 500:
                action_lock[i] = False
                need_decision[i] = True
            else:
                print(f"MOVE {int(drones[my_drone_ids[i]]['x'])} 0 0")
                continue


        
        if (not need_decision[i]) and creatures[targets[i]]["my_record"][0] == 1:
            need_decision[i] = True

        if need_decision[i]:

            print(
                f"run policy.decision_maker",
                file=sys.stderr,
                flush=True,
            )

            same_target_turn[i] = 0

            inputs = create_dm_inputs(creatures, drones, my_drone_ids[i], units)

            outputs = policy.decision_maker(inputs).flatten()

            # pos_penalty[i] = 1 if drones[my_drone_ids[i]]["y"] <= 500 else 0

            # print(
            #     f"penalty: {penalty}",
            #     file=sys.stderr,
            #     flush=True,
            # )

            # outputs[:-1] -= penalty
            # outputs[-1] -= pos_penalty[i]


            targets[i] = np.argmax(outputs) + 2 # fish ids start from 2

            if targets[i] == 14 or creatures[targets[i]]["my_record"][0] == 1:
                print(f"MOVE {int(drones[my_drone_ids[i]]['x'])} 0 0")
                action_lock[i] = True
                continue
            else:
                need_decision[i] = False
                # penalty[targets[i] - 2] = 1

        print(
            f"target: {targets[i]}",
            file=sys.stderr,
            flush=True,
        )

        radar = creatures[targets[i]]["radar"][i]

        # {"TL": 0, "TR": 1, "BL": 2, "BR": 3}
        des_x = drones[my_drone_ids[i]]["x"]
        des_y = drones[my_drone_ids[i]]["y"]
        light = 1 if np.random.random() < (same_target_turn[i] / 20) else 0

        if radar[0]:
            des_x -= 600
            des_y -= 600
        elif radar[1]:
            des_x += 600
            des_y -= 600
        elif radar[2]:
            des_x -= 600
            des_y += 600
        elif radar[3]:
            des_x += 600
            des_y += 600
            
            
        print(f"MOVE {clip(des_x)} {clip(des_y)} {light}")