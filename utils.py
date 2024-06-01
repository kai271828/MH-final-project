import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def sphere_function(array):
    return np.sum([element**2 for element in array])


def simulate(jar_path, agent1, weight1, agent2, weight2, level, seed):
    
    args = ['java', '-jar', jar_path, agent1, weight1, agent2, weight2, level, seed]

    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def seabed_security(jar_path, agent, opponent, opponent_weight_file, level, seed, difference_mode=False, verbose=False, parallel=False):
    def fitness(weight, weight_file="cache/weight.txt"):
        
        np.savetxt(weight_file, weight.flatten())
        score1_list = []
        score2_list = []


        for _ in range(3):

            _seed = int(np.random.uniform(-10000, 10000) * seed)

            result = simulate(jar_path, agent, weight_file, opponent, opponent_weight_file, str(level), str(_seed))

            scores = result.stderr.split("\n")
            score1_list.append(int(scores[-3]))
            score2_list.append(int(scores[-2]))

            if int(scores[-3]) == -1:
                print(result.stdout)

        fitness = 0.5 * np.min(score1_list) - 0.5 * np.min(score2_list) if difference_mode else np.min(score1_list)

        if verbose:
                print(f"agent's score: {score1_list},\topponent's score: {score2_list},\tfitness: {fitness}")

        return fitness if not parallel else (int(weight_file.split("_")[-1].split(".")[0]), fitness)
    
    def parallel_fitness(weights, max_workers=8):
        scores = {}
        weight_files = [f"cache/multithread_weight_{num}.txt" for num in range(len(weights))]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = {executor.submit(fitness, weight, weight_file): (weight, weight_file) for weight, weight_file in zip(weights, weight_files)}
            for future in as_completed(tasks):
                try:
                    result = future.result()
                    scores[result[0]] = result[1]
                except Exception as e:
                    print(f'An error occurred: {e}')

        return scores

    return parallel_fitness if parallel else fitness
