import argparse
import numpy as np
from tqdm.auto import tqdm

from utils import sphere_function, seabed_security


class EvolutionStrategy:
    def __init__(
        self,
        num_parents,
        dim,
        mutation_strength=0.1,
        self_adaptive=True,
        selection_type="all",
        optimization="max",
        fitness_function=None,
    ):
        assert fitness_function is not None, "You have to set fitness_function."
        assert selection_type in [
            "all",
            "offsprings",
        ], "You need to set selection_type as 'all' or 'offsprings'."
        assert optimization in [
            "min",
            "max",
        ], "You need to set optimization as 'min' or 'max'."
        self.num_parents = num_parents
        self.dim = dim
        self.mutation_strength = mutation_strength
        self.selection_type = selection_type
        self.fitness_function = fitness_function
        self.self_adaptive = self_adaptive
        self.learning_rate_1 = 1 / np.sqrt(self.dim)
        self.learning_rate_2 = 1 / np.sqrt(2 * np.sqrt(self.dim))
        self.optimization = optimization

        if self.self_adaptive:

            weights = [
                np.concatenate(
                    (
                        np.random.uniform(0, 1, self.dim),
                        np.random.uniform(-1, 1, self.dim),
                    ),
                )
                for i in range(self.num_parents)
            ]
            self.parents = [
                {"weight": weight, "score": self.fitness_function(weight[self.dim :])}
                for weight in weights
            ]
        else:
            weights = [
                np.random.uniform(-1, 1, self.dim) for i in range(self.num_parents)
            ]
            self.parents = [
                {"weight": weight, "score": self.fitness_function(weight)}
                for weight in weights
            ]

    def _marriage(self):
        a, b = np.random.choice(self.num_parents, 2, replace=False)
        return a, b

    def _recombine(self, a, b):
        return (self.parents[a]["weight"].copy() + self.parents[b]["weight"].copy()) / 2

    def _mutate(self, child):
        if self.self_adaptive:
            for i in range(len(child)):
                if i < self.dim:
                    child[i] = max(
                        child[i]
                        * np.exp(
                            self.learning_rate_1 * np.random.normal(0, 1)
                            + self.learning_rate_2 * np.random.normal(0, 1)
                        ),
                        1e-6,
                    )
                else:
                    child[i] += child[-1] * np.random.normal(0, 1)

        else:

            for i in range(len(child)):
                child[i] += self.mutation_strength * np.random.normal(0, 1)

    def _selection(self, offsprings):
        if self.selection_type == "all":
            temp = self.parents + offsprings
            temp.sort(
                key=lambda p: p["score"],
                reverse=True if self.optimization == "max" else False,
            )

            return temp[: self.num_parents]
        elif self.selection_type == "offsprings":
            offsprings.sort(
                key=lambda p: p["score"],
                reverse=True if self.optimization == "max" else False,
            )

            return offsprings[: self.num_parents]
        else:
            raise ValueError

    def evolve(self, generation, num_offsprings, verbose=False):

        for g in tqdm(range(generation)):
            offsprings = []

            for l in range(num_offsprings):
                index1, index2 = self._marriage()
                child = self._recombine(index1, index2)

                self._mutate(child)

                offsprings.append(
                    {
                        "weight": child,
                        "score": (
                            self.fitness_function(child[self.dim :])
                            if self.self_adaptive
                            else self.fitness_function(child)
                        ),
                    }
                )

            self.parents = self._selection(offsprings)

        return self.parents


def main(args):
    level2dim = {"1": 22803}

    es = EvolutionStrategy(
        num_parents=args.num_parents,
        dim=level2dim[args.level],
        mutation_strength=args.mutation_strength,
        self_adaptive=args.self_adaptive,
        selection_type=args.selection_type,
        optimization=args.optimization,
        fitness_function=seabed_security(args.jar_path, args.agent1, args.agent2, args.level, args.seed),
    )

    result = es.evolve(args.generation, args.num_offsprings, verbose=args.verbose)

    for r in result:
        print(r)


def parse_args():
    parser = argparse.ArgumentParser(description="Evolution Strategy.")
    parser.add_argument(
        "--num_parents",
        type=int,
        default=10,
        help="µ in ES.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="The level of the Seabed Security.",
    )
    parser.add_argument(
        "--mutation_strength",
        type=float,
        default=0.1,
        help="Coefficient of mutation.",
    )
    parser.add_argument(
        "--self_adaptive",
        action="store_true",
        help="Whether or not to use self adaptive mutation strength.",
    )
    parser.add_argument(
        "--selection_type",
        type=str,
        default="all",
        help="Choose from {'all', 'offsprings'}",
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="max",
        help="Choose from {'min', 'max'}",
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=10,
        help="Number of generation to evolve.",
    )
    parser.add_argument(
        "--num_offsprings",
        type=int,
        default=10,
        help="λ in ES.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether or not to use verbose mode.",
    )
    parser.add_argument(
        "--agent1",
        type=str,
        default="agents/level1.py",
        help="The agent 1.",
    )
    parser.add_argument(
        "--agent2",
        type=str,
        default="agents/level1.py",
        help="The agent 2." ,
    )
    parser.add_argument(
        "--jar_path",
        type=str,
        default="MH-final-project.jar",
        help="The jar file used to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9527,
        help="The seed of simulation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
