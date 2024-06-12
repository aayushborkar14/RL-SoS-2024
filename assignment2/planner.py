import argparse
import random
import sys

import numpy as np
import pulp as pl


class MDPPlanner:
    def __init__(self, mdp_file, algorithm="hpi", policy=None):
        self.algorithm = algorithm
        self.policy = policy
        self.T = None
        self.R = None
        with open(mdp_file, "r") as f:
            for line in f:
                line = line.strip()
                match line.split(" ")[0]:
                    case "numStates":
                        self.S = int(line.split(" ")[1])
                    case "numActions":
                        self.A = int(line.split(" ")[1])
                    case "end":
                        self.end = list(map(int, line.split(" ")[1:]))
                    case "transition":
                        s1, a, s2 = map(int, line.split(" ")[1:4])
                        r, p = map(float, line.split(" ")[4:6])
                        if self.T is None:
                            self.T = np.zeros(
                                (self.S, self.A, self.S),
                                dtype=float,
                            )
                        if self.R is None:
                            self.R = np.zeros(
                                (self.S, self.A, self.S),
                                dtype=float,
                            )
                        self.T[s1, a, s2] = p
                        self.R[s1, a, s2] = r
                    case "mdptype":
                        self.mdptype = line.split(" ")[1]
                    case "discount":
                        self.gamma = float(line.split(" ")[2])

    def computePolicy(self):
        assert self.policy is not None
        with open(self.policy, "r") as f:
            p = np.zeros(self.S, dtype=int)
            for i, line in enumerate(f):
                if line == "":
                    continue
                p[i] = int(line.strip())
        V = np.linalg.inv(
            np.identity(self.S) - self.gamma * self.T[np.arange(self.S), p]
        ) @ np.sum(
            self.T[np.arange(self.S), p] * self.R[np.arange(self.S), p],
            axis=1,
            keepdims=True,
        )
        return V.reshape(self.S), p

    def solve(self):
        if policy is not None:
            return self.computePolicy()
        if self.algorithm == "vi":
            return self.valueIteration()
        if self.algorithm == "lp":
            return self.linearProgramming()
        return self.howardPolicyIteration()

    def valueIteration(self):
        V = np.zeros(self.S, dtype=float)
        assert self.R is not None and self.T is not None
        while True:
            V_new = np.max(
                np.sum(
                    self.T * (self.R + self.gamma * V.reshape(1, 1, self.S)), axis=2
                ),
                axis=1,
            )
            if np.abs(V_new - V).max() < 1e-10:
                V = V_new
                break
            V = V_new
        p = np.argmax(
            np.sum(self.T * (self.R + self.gamma * V.reshape(1, 1, self.S)), axis=2),
            axis=1,
        )
        return V, p

    def howardPolicyIteration(self):
        assert self.R is not None and self.T is not None
        p = np.random.randint(0, self.A, self.S)
        while True:
            V = np.linalg.inv(
                np.identity(self.S) - self.gamma * self.T[np.arange(self.S), p]
            ) @ np.sum(
                self.T[np.arange(self.S), p] * self.R[np.arange(self.S), p],
                axis=1,
                keepdims=True,
            )
            p_new = np.argmax(
                np.sum(self.T * (self.R + self.gamma * V.transpose()), axis=2),
                axis=1,
            )
            if np.array_equal(p, p_new):
                break
            p = p_new
        return V.reshape(self.S), p

    def linearProgramming(self):
        assert self.R is not None and self.T is not None
        prob = pl.LpProblem("MDP", pl.LpMinimize)
        Vlp = [pl.LpVariable(f"V{s}") for s in range(self.S)]
        prob += pl.lpSum(Vlp)
        for s in range(self.S):
            for a in range(self.A):
                prob += Vlp[s] >= pl.lpSum(
                    [
                        self.T[s, a, s2] * (self.R[s, a, s2] + self.gamma * Vlp[s2])
                        for s2 in range(self.S)
                    ]
                )
        if self.mdptype == "episodic":
            for s in self.end:
                prob += Vlp[s] == 0
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        V = np.zeros(self.S, dtype=float)
        for v in prob.variables():
            V[int(v.name[1:])] = v.varValue
        p = np.zeros(self.S, dtype=int)
        for s in range(self.S):
            p[s] = np.argmax(
                np.sum(self.T[s, :, :] * (self.R[s, :, :] + self.gamma * V), axis=1)
            )
        return V, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="planner.py")
    parser.add_argument("--mdp", type=str, required=False)
    parser.add_argument(
        "--algorithm",
        type=str,
        required=False,
        default="hpi",
        choices=["vi", "lp", "hpi"],
    )
    parser.add_argument("--policy", type=str, required=False, default=None)
    args = parser.parse_args()
    mdp_file = args.mdp
    algorithm = args.algorithm
    policy = args.policy
    mdp = MDPPlanner(mdp_file, algorithm, policy)
    V, p = mdp.solve()
    for s in range(V.shape[0]):
        print("%.6f %d" % (V[s], p[s]))
