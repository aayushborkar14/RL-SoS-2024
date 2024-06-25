import time

import matplotlib.pyplot as plt

from task1 import KL_UCB

regrets = [
    97.50000000000247,
    122.66000000000285,
    150.5399999999909,
    161.91999999995136,
    182.51999999986205,
    190.53999999969682,
    223.8799999993397,
    244.95999999866058,
    238.33999999737796,
]
horizons = [2**i for i in range(10, 19)]
algorithm = KL_UCB
plt.plot(horizons, regrets)
plt.title("Regret vs Horizon")
plt.savefig(
    "task1-{}-{}.png".format(algorithm.__name__, time.strftime("%Y%m%d-%H%M%S"))
)
plt.clf()
