"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, reward): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars


class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError


# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def ucb(p, t, u):
    return p + math.sqrt(2 * math.log(t) / u)


def KL(x, y):
    if x:
        return x * math.log(x / y) + (1 - x) * math.log((1 - x) / (1 - y))
    else:
        return -math.log(1 - y)


def ucb_kl_func(t, c, u):
    return (math.log(t) + c * math.log(math.log(t))) / u


def ucb_kl(p, t, c, u):
    l, r = p, 1
    q = (l + r) / 2  # initial guess
    while r - l > 1e-2:
        if KL(p, q) > ucb_kl_func(t, c, u):
            r = q
        else:
            l = q
        q = (l + r) / 2
    return q


# END EDITING HERE


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.p = np.zeros(num_arms)
        self.u = np.zeros(num_arms)
        self.t = 1
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        # if any arm has not been pulled yet, pull that arm
        if any(self.u == 0):
            self.t += 1
            return np.random.choice(np.where(self.u == 0)[0])
        # otherwise, pull the arm with the highest UCB value
        max_ucb, arm = -1, -1
        for i in range(self.num_arms):
            ucb_val = ucb(self.p[i], self.t, self.u[i])
            if ucb_val > max_ucb:
                max_ucb = ucb_val
                arm = i
        self.t += 1
        return arm
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.p[arm_index] = (self.p[arm_index] * self.u[arm_index] + reward) / (
            self.u[arm_index] + 1
        )
        self.u[arm_index] += 1
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.p = np.zeros(num_arms)
        self.u = np.zeros(num_arms)
        self.t = 1
        self.c = 3
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        # if any arm has not been pulled yet, pull that arm
        if any(self.u == 0):
            self.t += 1
            return np.random.choice(np.where(self.u == 0)[0])
        # otherwise, pull the arm with the highest UCB-KL value
        max_kl_ucb, arm = -1, -1
        for i in range(self.num_arms):
            kl_ucb_val = ucb_kl(self.p[i], self.t, self.c, self.u[i])
            if kl_ucb_val > max_kl_ucb:
                max_kl_ucb = kl_ucb_val
                arm = i
        self.t += 1
        return arm
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.p[arm_index] = (self.p[arm_index] * self.u[arm_index] + reward) / (
            self.u[arm_index] + 1
        )
        self.u[arm_index] += 1
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.s = np.zeros(num_arms)
        self.f = np.zeros(num_arms)
        # END EDITING HERE

    def give_pull(self):
        # START EDITING HERE
        return np.argmax(np.random.beta(self.s + 1, self.f + 1))
        # END EDITING HERE

    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward:
            self.s[arm_index] += 1
        else:
            self.f[arm_index] += 1
        # END EDITING HERE
