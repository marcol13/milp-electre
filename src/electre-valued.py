import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, value, LpStatus
from itertools import product
from collections import defaultdict
from utils.const import score_dict, relations
from utils.utils import get_preference
from utils.visualize import visualize_ranking


class ValuedArray:
    def __init__(self, matrix_v):
        self.matrix_v = matrix_v
        self.shape = matrix_v.shape
        self.matrix_pp, self.matrix_pn, self.matrix_i = self.set_relation_matrices(matrix_v)
        # (a,b): {P+, P-, I, R}

    def set_relation_matrices(self, mat):
        upper_triangle = np.triu(mat)
        lower_triangle = np.tril(mat)

        matrix_pp = mat - lower_triangle + upper_triangle.T - np.diag(np.diag(upper_triangle))
        matrix_pn = mat - upper_triangle + lower_triangle.T - np.diag(np.diag(lower_triangle))
        matrix_i = 1 - mat - mat.T

        return matrix_pp, matrix_pn, matrix_i

    def create_variable_matrix(self, name):
        return np.array(
            [
                LpVariable(f"{name}_{i}_{k}", 0, 1, LpInteger) if i != k else 0
                for i in range(self.shape[0])
                for k in range(self.shape[1])
            ]
        ).reshape(self.shape)

    def distance_func(self, rel1, rel2):
        return score_dict[relations[rel1]][relations[rel2]]

    def solve_partial(self, verbose=False, visualize=True):
        prob = LpProblem("max support", LpMinimize)

        r = self.create_variable_matrix("r")
        rel_pp = self.create_variable_matrix("PP")
        rel_pn = self.create_variable_matrix("PN")
        rel_i = self.create_variable_matrix("I")
        rel_r = self.create_variable_matrix("R")

        var_relations = [rel_pp, rel_pn, rel_i, rel_r]

        prob += lpSum(
            [
                rel_p[i][k] * rel[i][k] * self.distance_func(rel_index, rel_p_index)
                for i, k in product(range(self.shape[0]), range(self.shape[1]))
                if i < k
                for rel_index, rel in enumerate(var_relations)
                for rel_p_index, rel_p in enumerate(
                    [self.matrix_pp, self.matrix_pn, self.matrix_i]
                )
            ]
        )

        # func
        for i in range(self.shape[0]):
            for k in range(self.shape[1]):
                if i != k:
                    print("LOL")
                    print(i, k)
                    print(i, k, rel_pp[i][k] + rel_pn[i][k] + rel_r[i][k] + rel_i[i][k], r[i][k])
                    prob += (
                        r[i][k] - r[k][i] <= rel_pp[i][k],
                        f"Positive preference {i}-{k}",
                    )
                    prob += (
                        r[k][i] - r[i][k] <= rel_pn[i][k],
                        f"Negative preference {i}-{k}",
                    )
                    prob += (
                        r[i][k] + r[k][i] - 1 <= rel_i[i][k],
                        f"Indifference {i}-{k}",
                    )
                    prob += (
                        1 - r[i][k] - r[k][i] <= rel_r[i][k],
                        f"Incomparability {i}-{k}",
                    )
                    prob += (
                        rel_pp[i][k] + rel_pn[i][k] + rel_r[i][k] + rel_i[i][k] == 1,
                        f"Only one relation [{i}, {k}]",
                    )

        for t_i, t_k, t_p in product(
            range(self.shape[0]), range(self.shape[0]), range(self.shape[0])
        ):
            if t_i != t_k and t_k != t_p and t_i != t_p:
                prob += (
                    r[t_i][t_k] >= r[t_i][t_p] + r[t_p][t_k] - 1.5,
                    f"Transition {t_i}-{t_k}-{t_p}",
                )

        prob.solve()

        if verbose:
            print("Status:", LpStatus[prob.status])
            print()

            vars = np.array(
                [x.name.split("_") + [x.varValue] for x in prob.variables()]
            )
            print(vars)
            rels = list(set(vars[:, 0]))
            matrices = defaultdict(
                lambda: np.eye(self.shape[0]),
                {rel: np.eye(self.shape[0]) for rel in rels},
            )

            for rel, i, j, value in vars:
                matrices[rel][int(i)][int(j)] = value

            for key in matrices.keys():
                print(f"Matrix {key}:")
                print(matrices[key])
                print()

            print(f"Objective function: {prob.objective}")

        if(visualize):
            vars = np.array([x.name.split("_") + [x.varValue] for x in prob.variables()])
            rels = list(set(vars[:,0]))
            matrices = defaultdict(lambda: np.eye(self.shape[0]), {rel: np.eye(self.shape[0]) for rel in rels})

            for rel, i, j, value in vars:
                matrices[rel][int(i)][int(j)] = value

            visualize_ranking(matrices["r"])

    def solve_complete(self, verbose=False, visualize=True):
        prob = LpProblem("max support", LpMinimize)

        r = self.create_variable_matrix("r")
        rel_z = self.create_variable_matrix("Z")

        # prob += lpSum([r[i][k] * self.distance_func(self.matrix_a, 0, i, k) + r[k][i] * self.distance_func(self.matrix_a, 1, i, k) + rel_z[i][k] * (self.distance_func(self.matrix_a, 2, i, k) - self.distance_func(self.matrix_a, 0, i, k) - self.distance_func(self.matrix_a, 1, i, k))  for i, k in product(range(self.shape[0]), range(self.shape[1])) if i < k])
        prob += lpSum(
            [
                rel_p[i][k] * (r[i][k] - rel_z[i][k]) * self.distance_func(0, rel_p_index) + (r[k][i] - rel_z[i][k]) * rel_p[i][k] * self.distance_func(1, rel_p_index) + rel_z[i][k] * rel_p[i][k] * self.distance_func(2, rel_p_index)
                for i, k in product(range(self.shape[0]), range(self.shape[1]))
                if i < k
                for rel_p_index, rel_p in enumerate(
                    [self.matrix_pp, self.matrix_pn, self.matrix_i]
                )
            ]
        )

        for i in range(self.shape[0]):
            for k in range(self.shape[1]):
                if i != k:
                    prob += r[i][k] + r[k][i] >= 1, f"Weak preference {i}-{k}"
                    prob += rel_z[i][k] == r[i][k] + r[k][i] - 1, f"Incomparability {i}-{k}"

        for t_i, t_k, t_p in product(range(self.shape[0]), range(self.shape[0]), range(self.shape[0])):
            if t_i != t_k and t_k != t_p and t_i != t_p:
                prob += r[t_i][t_k] >= r[t_i][t_p] + r[t_p][t_k] - 1.5, f"Transition {t_i}-{t_k}-{t_p}"

        prob.solve()

        if(verbose):
            print("Status:", LpStatus[prob.status])
            print()

            vars = np.array([x.name.split("_") + [x.varValue] for x in prob.variables()])
            rels = list(set(vars[:,0]))
            matrices = defaultdict(lambda: np.eye(self.shape[0]), {rel: np.eye(self.shape[0]) for rel in rels})

            for rel, i, j, value in vars:
                matrices[rel][int(i)][int(j)] = value

            for key in matrices.keys():
                print(f"Matrix {key}:")
                print(matrices[key])
                print()

            print(f"Objective function: {prob.objective}")

        if(visualize):
            vars = np.array([x.name.split("_") + [x.varValue] for x in prob.variables()])
            rels = list(set(vars[:,0]))
            matrices = defaultdict(lambda: np.eye(self.shape[0]), {rel: np.eye(self.shape[0]) for rel in rels})

            for rel, i, j, value in vars:
                matrices[rel][int(i)][int(j)] = value

            visualize_ranking(matrices["r"])



arr = np.array([[0, 0, 0.7], [1, 0, 0.2], [0.3, 0, 0]], dtype=np.float32)

# arr = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
# arr = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 1]], dtype=np.uint8)
# arr = np.array([[1,1,1,0,1,1,1,1,1,1],[0,1,0,0,1,1,1,0,1,1],[0,0,1,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,1],[0,1,1,0,1,1,1,0,0,1],[0,1,0,0,1,1,0,1,1,1],[0,1,0,0,1,0,0,0,1,1],[0,0,0,0,1,0,0,0,0,1]], dtype=np.uint8)
# visualize_ranking(arr)
a = ValuedArray(arr)
a.solve_partial(verbose=True, visualize=True)

a.solve_complete(verbose=True, visualize=True)
