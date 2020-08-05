import numpy as np
import matplotlib.pyplot as plt

from Artificial_bee_colony import *

#from matplotlib.style import use



def simulate(obj_function, colony_size=30, n_iter=10, max_trials=10, simulations=1):
    itr = range(n_iter)
    values = np.zeros(n_iter)
    box_optimal = []
    for _ in range(simulations):
        optimizer = ABeeC(obj_function=ObjectiveFunction(name='Sphere', dim=30, minf=-10.0, maxf=10.0),
                        colony_size=colony_size, n_iter=n_iter, max_trials=max_trials)
        optimizer.optimize()
        values += np.array(optimizer.optimality_tracking)
        box_optimal.append(optimizer.optimal_solution.fitness)
#        print(optimizer.optimal_solution.pos)
    values /= simulations

#    plt.plot(itr, values, lw=0.5, label=obj_function)
    plt.legend(loc='upper right')


def main():
#    plt.figure(figsize=(8, 7))
    simulate('Sphere')
#    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
#    plt.xticks(rotation=45)
#    plt.show()


if __name__ == '__main__':
    main()
