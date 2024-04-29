import numpy as np
from pulp import LpVariable, LpInteger, LpProblem, LpMinimize, LpStatus, lpSum
from pulp.constants import LpStatusOptimal, LpStatusNotSolved
from ..core.relations import PositivePreference, NegativePreference, Indifference, Incomparible
from ..core.types import RankingModeType
from ..core.const import RankingMode
from ..core.visualize.graph import Graph
from collections import defaultdict
from itertools import permutations
from abc import ABC, abstractmethod

class Outranking(ABC):
    def __init__(self, credibility, scores, labels: list[str]):
        self.credibility = credibility.matrix
        self.size = credibility.get_size()
        self.scores = scores
        self.labels = labels
        self.problem = LpProblem("Maximize_support", LpMinimize)
        self.results = []

        self.upper_matrix_ids = np.triu_indices(self.size, 1)
        self.upper_matrix_ids = np.column_stack(self.upper_matrix_ids)

        self.unique_permutations = list(permutations(range(self.size), 3))

    def create_variable_matrix(self, name):
        return np.array([LpVariable(f"{name}_{i}_{k}", 0, 1, LpInteger) if i != k else 0 for i in range(self.size) for k in range(self.size)]).reshape((self.size, self.size))

    def solve(self, mode: RankingModeType, all_results: bool = False):
        if mode == "partial":
            self.problem = self.init_partial(self.problem)
        elif mode == "complete":
            self.problem = self.init_complete(self.problem)
        else:
            self.results = None
            raise ValueError("Invalid mode")
        
        self.results = self.solve_problem(self.problem, all_results)
        
        
    def solve_problem(self, problem, all_results):
        results = []
        if all_results:
            prev_objective_value = None
            while True:
                print(problem.constraints)
                problem.solve()
                if problem.status == LpStatusOptimal and (prev_objective_value is None or problem.objective.value() <= prev_objective_value):
                    prev_objective_value = problem.objective.value()
                    result_matrix = self.get_outranking(problem, "p")
                    results.append(result_matrix)
                    problem = self.get_new_constraints(problem)
                else:
                    break
        else:
            problem.solve()
            results.append(problem)

        return results

    @abstractmethod
    def init_partial(self, all_results: bool = False):
        pass

    @abstractmethod
    def init_complete(self, all_results: bool = False):
        pass

    def create_variables(self, relations: list[str]) -> dict:
        variables = dict()
        for relation in relations:
            variables[relation] = self.create_variable_matrix(relation)
        return variables
    
    def add_contraints(self, mode: RankingModeType, problem, variables, size, unique_permutations):
        if mode == RankingMode.PARTIAL:
            for i in range(size):
                for j in range(size):
                    if i != j:
                        problem += variables["outranking"][i][j] - variables["outranking"][j][i] <= variables["pp"][i][j], f"Positive preference [{i}-{j}]"
                        problem += variables["outranking"][j][i] - variables["outranking"][i][j] <= variables["pn"][i][j], f"Negative preference [{i}-{j}]"
                        problem += variables["outranking"][i][j] + variables["outranking"][j][i] - 1 <= variables["i"][i][j], f"Indifference [{i}-{j}]"
                        problem += 1 - variables["outranking"][i][j] - variables["outranking"][j][i] <= variables["r"][i][j], f"Incomparability [{i}-{j}]"
                        problem += variables["pp"][i][j] + variables["pn"][i][j] + variables["r"][i][j] + variables["i"][i][j] == 1, f"Only one relation [{i}, {j}]"

            for i, k, p in unique_permutations:
                problem += variables["outranking"][i][k] >= variables["outranking"][i][p] + variables["outranking"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

            return problem
        elif mode == RankingMode.COMPLETE:
            for i in range(size):
                for j in range(size):
                    if i != j:
                        problem += variables["p"][i][j] + variables["p"][j][i] >= 1, f"Weak preference [{i}-{j}]"
                        problem += variables["z"][i][j] == variables["p"][i][j] + variables["p"][j][i] - 1, f"Incomparability [{i}-{j}]"

            for i, k, p in unique_permutations:
                problem += variables["p"][i][k] >= variables["p"][i][p] + variables["p"][p][k] - 1.5, f"Transitivity [{i}-{k}-{p}]"

            return problem
        else:
            raise ValueError("Invalid mode")

    def verbose(self):
        print("Status:", LpStatus[self.problem.status])
        print()

        print(self.problem.constraints)

        print()

        vars = np.array([x.name.split("_") + [x.varValue] for x in self.problem.variables()])
        rels = list(set(vars[:,0]))
        matrices = defaultdict(lambda: np.eye(self.size), {rel: np.eye(self.size) for rel in rels})

        for rel, i, j, value in vars:
            matrices[rel][int(i)][int(j)] = value

        for key in matrices.keys():
            print(f"Matrix {key}:")
            print(matrices[key])
            print()

        print(f"Objective function: {self.problem.objective}")

    def get_outranking(self, problem, relation_array: str):
        variables = np.array([x.name.split("_") + [x.varValue] for x in problem.variables()])
        variables = variables[variables[:, 0] == relation_array]
        outranking = np.eye(self.size)
        for _, i, j, value in variables:
            outranking[int(i)][int(j)] = value
        return outranking
    
    def get_new_constraints(self, problem):
        final_values = []
        for var in problem.variables():
            if var.varValue == 1:
                final_values.append(var)
        problem += lpSum([var for var in final_values]) <= len(final_values) - 1
        return problem

    @staticmethod
    def get_preference(i: int, j: int):
        if i > j:
            return PositivePreference
        elif j > i:
            return NegativePreference
        elif i == j == 1:
            return Indifference
        else:
            return Incomparible
        
    def show_graph(self, all_results: bool = False):
        if all_results:
            for idx, result in enumerate(self.results):
                graph = Graph(result, self.labels)
                graph.show(f"temp_{idx}")
        else:
            graph = Graph(self.results[0], self.labels)
            graph.show("temp")

    def save_graph(self, all_results: bool = False, path: str = "temp"):
        if all_results:
            for idx, result in enumerate(self.results):
                graph = Graph(result, self.labels)
                graph.save(f"{path}_{idx}")
        else:
            graph = Graph(self.results[0], self.labels)
            graph.save(path)