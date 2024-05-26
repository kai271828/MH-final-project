import numpy as np
import subprocess


def sphere_function(array):
    return np.sum([element**2 for element in array])


def simulate(jar_path, agent1, agent2, level, seed):
    
    args = ['java', '-jar', jar_path, agent1, agent2, level, seed]

    try:
        subprocess.run(args, capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def seabed_security(jar_path, agent1, agent2, level, seed):
    def fitness(weight):
        simulate(jar_path, agent1, agent2, level, seed)
        with open("cache.txt", 'r') as file:
            lines = file.readlines()
            score1 = int(lines[0].strip())
            score2 = int(lines[1].strip())

        return score1, score2

    return fitness
